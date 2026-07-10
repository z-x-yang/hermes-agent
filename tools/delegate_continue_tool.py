from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional

from tools.registry import registry
from tools.subagent_sessions import (
    RetainedSubagentSession,
    claim_retained_subagent_session,
    release_retained_subagent_session,
    update_retained_history,
)


DELEGATE_CONTINUE_SCHEMA = {
    "name": "delegate_continue",
    "description": (
        "Continue a short-lived retained subagent by agent_id. The original "
        "subagent type, workspace hint, model/provider metadata, role, and "
        "capability ceiling are preserved; callers cannot choose new tools."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "agent_id returned by a retained delegate_task result.",
            },
            "prompt": {
                "type": "string",
                "description": "Follow-up instruction for the same retained subagent session.",
            },
            "scheduling": {
                "type": "string",
                "enum": ["auto", "foreground", "background"],
                "description": "Whether to wait, background, or use the retained subagent type default.",
            },
        },
        "required": ["agent_id", "prompt"],
    },
}


def _tool_error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


class _ContinuationInterruptBridge:
    """Thread-safe interrupt handle for one lazily-built continuation child."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._child = None
        self._interrupt_requested = False
        self._interrupt_delivered = False

    @staticmethod
    def _signal_child(child) -> None:
        interrupt = getattr(child, "interrupt", None)
        if callable(interrupt):
            interrupt()
        else:
            child._interrupt_requested = True

    def register(self, child) -> None:
        with self._lock:
            self._child = child
            should_interrupt = self._interrupt_requested and not self._interrupt_delivered
            if should_interrupt:
                self._interrupt_delivered = True
        if should_interrupt:
            self._signal_child(child)

    def __call__(self) -> None:
        with self._lock:
            self._interrupt_requested = True
            child = self._child
            should_interrupt = child is not None and not self._interrupt_delivered
            if should_interrupt:
                self._interrupt_delivered = True
        if should_interrupt:
            self._signal_child(child)


def _build_continuation_child(
    record: RetainedSubagentSession,
    *,
    prompt: str,
    parent_agent,
):
    """Rebuild a child agent from retained non-secret metadata.

    Credentials are deliberately resolved from the current trusted parent/config
    state; the retained record never carries API keys.
    """

    from tools.delegate_tool import (
        _build_child_agent,
        _load_config,
        _normalize_role,
        _prepare_delegation_credentials_config,
        _resolve_delegation_credentials,
    )
    from tools.subagent_profiles import get_subagent_profile, resolve_profile_config

    cfg = _load_config()
    subagent_type = record.subagent_type or "general-purpose"
    profile = get_subagent_profile(subagent_type)
    resolved_profile = resolve_profile_config(subagent_type, cfg)

    parent_provider = str(getattr(parent_agent, "provider", "") or "")
    record_provider = str(record.provider or "")
    child_cfg = _prepare_delegation_credentials_config(
        cfg,
        model=record.model or resolved_profile.model,
        provider=record_provider or resolved_profile.provider,
        provider_changed=bool(record_provider and record_provider != parent_provider),
    )
    if record_provider and record_provider != parent_provider:
        creds = _resolve_delegation_credentials(child_cfg, parent_agent)
    else:
        # Same provider as the live parent: inherit current trusted credentials
        # and endpoint from the parent, but preserve the retained model name.
        creds = {
            "model": child_cfg.get("model"),
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": [],
        }

    effective_role = _normalize_role(record.role)
    default_max_iter = cfg.get("max_iterations")
    try:
        max_iterations = int(default_max_iter) if default_max_iter is not None else 50
    except (TypeError, ValueError):
        max_iterations = 50

    return _build_child_agent(
        task_index=0,
        goal=prompt,
        context=None,
        toolsets=None,
        model=creds.get("model") or record.model or resolved_profile.model,
        max_iterations=max_iterations,
        task_count=1,
        parent_agent=parent_agent,
        override_provider=creds.get("provider"),
        override_base_url=creds.get("base_url"),
        override_api_key=creds.get("api_key"),
        override_api_mode=creds.get("api_mode"),
        override_acp_command=creds.get("command"),
        override_acp_args=creds.get("args"),
        role=effective_role,
        profile=profile,
        workspace_path_override=record.workspace_path,
    )


def _remove_active_child(parent_agent, child) -> None:
    if not hasattr(parent_agent, "_active_children"):
        return
    try:
        lock = getattr(parent_agent, "_active_children_lock", None)
        if lock:
            with lock:
                if child in parent_agent._active_children:
                    parent_agent._active_children.remove(child)
        elif child in parent_agent._active_children:
            parent_agent._active_children.remove(child)
    except Exception:
        pass


def _run_continuation_entry(
    record: RetainedSubagentSession,
    prompt: str,
    parent_agent,
    *,
    child_run_timeout_seconds: Optional[float] = None,
    interrupt_bridge: Optional[_ContinuationInterruptBridge] = None,
) -> dict[str, Any]:
    start = time.monotonic()
    child = None
    try:
        child = _build_continuation_child(record, prompt=prompt, parent_agent=parent_agent)
        if interrupt_bridge is not None:
            interrupt_bridge.register(child)
        payload = _build_continue_payload(prompt)

        def _run_child_conversation():
            return child.run_conversation(
                user_message=payload,
                conversation_history=list(record.conversation_history),
                task_id=f"delegation-continue-{record.agent_id}-{int(time.time())}",
            )

        if child_run_timeout_seconds is None:
            result = _run_child_conversation()
        else:
            from tools.delegate_tool import _run_child_conversation_with_timeout

            result, timeout_entry = _run_child_conversation_with_timeout(
                child=child,
                run_callable=_run_child_conversation,
                timeout_seconds=child_run_timeout_seconds,
                task_index=0,
                goal=prompt,
                child_start=start,
            )
            if timeout_entry is not None:
                timeout_entry.pop("task_index", None)
                timeout_entry.pop("_child_role", None)
                timeout_entry["agent_id"] = record.agent_id
                timeout_entry["model"] = getattr(child, "model", record.model)
                timeout_entry["provider"] = getattr(child, "provider", record.provider)
                timeout_entry["subagent_type"] = record.subagent_type
                timeout_entry["role"] = getattr(child, "_delegate_role", record.role)
                return timeout_entry
            assert result is not None
        messages = result.get("messages") if isinstance(result, dict) else None
        if isinstance(messages, list):
            update_retained_history(record.agent_id, list(messages))
        summary = (result or {}).get("final_response") or ""
        status = "completed" if summary else "failed"
        entry: dict[str, Any] = {
            "status": status,
            "agent_id": record.agent_id,
            "summary": summary,
            "api_calls": (result or {}).get("api_calls", 0),
            "duration_seconds": round(time.monotonic() - start, 2),
            "model": getattr(child, "model", record.model),
            "provider": getattr(child, "provider", record.provider),
            "subagent_type": record.subagent_type,
            "role": getattr(child, "_delegate_role", record.role),
        }
        if status != "completed":
            entry["error"] = (result or {}).get("error") or "Subagent did not produce a response."
        return entry
    except Exception as exc:  # noqa: BLE001 - tool must fail closed as JSON
        return {
            "status": "error",
            "agent_id": record.agent_id,
            "error": f"{type(exc).__name__}: {exc}",
            "summary": None,
            "api_calls": 0,
            "duration_seconds": round(time.monotonic() - start, 2),
        }
    finally:
        if child is not None:
            _remove_active_child(parent_agent, child)
            try:
                if hasattr(child, "close"):
                    child.close()
            except Exception:
                pass


def _build_continue_payload(prompt: str) -> str:
    from tools.delegate_tool import _build_child_task_payload

    return _build_child_task_payload(prompt, None)


def _combined_for_async(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "results": [entry],
        "total_duration_seconds": entry.get("duration_seconds", 0),
    }


def _unwrap_foreground_payload(payload: str) -> str:
    try:
        data = json.loads(payload)
        results = data.get("results")
        if isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict):
            return json.dumps(results[0], ensure_ascii=False)
    except Exception:
        pass
    return payload


def delegate_continue(
    agent_id: str,
    prompt: str,
    scheduling: str = "auto",
    *,
    parent_agent=None,
) -> str:
    if parent_agent is None:
        return _tool_error("delegate_continue requires a parent agent context.")
    if scheduling not in {"auto", "foreground", "background"}:
        return _tool_error(f"Invalid scheduling: {scheduling}")

    parent_session_id = str(getattr(parent_agent, "session_id", "") or "")
    if not parent_session_id:
        return _tool_error("delegate_continue requires a non-empty parent session id.")
    if not isinstance(prompt, str) or not prompt.strip():
        return _tool_error("prompt is required")

    try:
        record = claim_retained_subagent_session(agent_id)
    except (KeyError, RuntimeError) as exc:
        return _tool_error(str(exc))

    if not record.parent_session_id:
        release_retained_subagent_session(agent_id)
        return _tool_error("delegate_continue requires a non-empty parent session id.")
    if record.parent_session_id != parent_session_id:
        release_retained_subagent_session(agent_id)
        return _tool_error("agent_id does not belong to this parent session")

    from tools.delegate_tool import (
        _get_max_async_children,
        _load_config,
        _resolve_foreground_timeouts,
        _resolve_scheduling,
    )

    is_subagent = getattr(parent_agent, "_delegate_depth", 0) > 0
    try:
        delivery_mode = _resolve_scheduling(
            record.subagent_type,
            scheduling,
            is_batch=False,
            is_subagent=is_subagent,
        )
    except ValueError as exc:
        release_retained_subagent_session(agent_id)
        return _tool_error(str(exc))

    foreground_wait_timeout_seconds: Optional[float] = None
    child_run_timeout_seconds: Optional[float] = None
    if delivery_mode == "foreground":
        try:
            cfg = _load_config()
            (
                foreground_wait_timeout_seconds,
                child_run_timeout_seconds,
            ) = _resolve_foreground_timeouts(record.subagent_type, cfg)
        except Exception as exc:
            release_retained_subagent_session(agent_id)
            return _tool_error(f"Cannot resolve continuation timeouts: {exc}")

    start = time.monotonic()
    interrupt_bridge = _ContinuationInterruptBridge()

    def _runner() -> dict[str, Any]:
        try:
            entry = _run_continuation_entry(
                record,
                prompt,
                parent_agent,
                child_run_timeout_seconds=child_run_timeout_seconds,
                interrupt_bridge=interrupt_bridge,
            )
            return _combined_for_async(entry)
        finally:
            release_retained_subagent_session(agent_id)

    # Nested continuations run inline; they cannot own async delivery.
    if is_subagent:
        combined = _runner()
        return _unwrap_foreground_payload(json.dumps(combined, ensure_ascii=False))

    try:
        from gateway.session_context import async_delivery_supported
        async_ok = async_delivery_supported()
    except Exception:
        async_ok = True
    if not async_ok:
        combined = _runner()
        entry = combined["results"][0]
        entry["note"] = (
            f"scheduling={delivery_mode} cannot detach on this endpoint "
            "(stateless HTTP API), so the retained subagent ran synchronously."
        )
        return json.dumps(entry, ensure_ascii=False)

    from tools.async_delegation import (
        dispatch_async_delegation_batch,
        wait_for_async_delegation,
    )
    from tools.approval import get_current_session_key

    try:
        dispatch = dispatch_async_delegation_batch(
            goals=[f"Continue retained subagent {record.agent_id}"],
            context=None,
            toolsets=None,
            role=record.role,
            model=record.model,
            session_key=get_current_session_key(default=""),
            runner=_runner,
            interrupt_fn=interrupt_bridge,
            max_async_children=_get_max_async_children(),
            initial_delivery_mode=(
                "foreground_waiting" if delivery_mode == "foreground" else "background"
            ),
        )
    except Exception as exc:
        release_retained_subagent_session(agent_id)
        return _tool_error(f"Failed to dispatch retained subagent continuation: {exc}")

    if dispatch.get("status") != "dispatched":
        combined = _runner()
        entry = combined["results"][0]
        entry["note"] = (
            "The background delegation pool was at capacity, so the retained "
            "subagent continuation ran synchronously and the result is included."
        )
        return json.dumps(entry, ensure_ascii=False)

    if delivery_mode == "foreground":
        payload = wait_for_async_delegation(
            dispatch,
            timeout_seconds=float(foreground_wait_timeout_seconds or 0),
        )
        if payload is not None:
            return _unwrap_foreground_payload(payload)
        return json.dumps(
            {
                "status": "backgrounded_after_foreground_timeout",
                "mode": "background",
                "delegation_id": dispatch["delegation_id"],
                "agent_id": record.agent_id,
                "duration_seconds": round(time.monotonic() - start, 2),
                "note": (
                    "Foreground wait timed out. The same retained subagent "
                    "continuation is still running and will re-enter on completion."
                ),
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "status": "dispatched",
            "mode": "background",
            "delegation_id": dispatch["delegation_id"],
            "agent_id": record.agent_id,
            "note": (
                "Retained subagent continuation is running in the background. "
                "Its full result re-enters the conversation when it finishes."
            ),
        },
        ensure_ascii=False,
    )


registry.register(
    name="delegate_continue",
    toolset="delegation",
    schema=DELEGATE_CONTINUE_SCHEMA,
    handler=lambda args, **kw: delegate_continue(
        agent_id=args.get("agent_id", ""),
        prompt=args.get("prompt", ""),
        scheduling=args.get("scheduling", "auto"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=lambda: True,
    emoji="↪️",
)
