from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Callable


@dataclass(frozen=True)
class SummaryRuntime:
    """Lightweight bridge from the compressor to the active main-model runtime."""

    provider: str
    model: str
    api_mode: str
    base_url: str
    reasoning_effort: str | None
    context_limit_tokens: int | None
    tools_included: bool
    main_api_calls_in_process: int
    summary_runtime_shape: str | None
    summary_runtime_toolset_source: str | None
    build_kwargs: Callable[[list[dict[str, Any]], int], dict[str, Any]]
    fingerprint_prefix: Callable[[list[dict[str, Any]]], str]
    invoke: Callable[[dict[str, Any]], Any]
    estimate_request_tokens: Callable[[dict[str, Any]], int]
    activate_fallback: Callable[[BaseException], bool] | None = None
    fallback_attempt_budget: int = 0


_NON_PREFIX_REQUEST_FIELDS = frozenset({
    "include",
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
    "metadata",
    "n",
    "parallel_tool_calls",
    "response_format",
    "service_tier",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "timeout",
    "tool_choice",
    "top_k",
    "top_p",
    "user",
})


def fingerprint_cache_visible_prefix(
    api_kwargs: dict[str, Any],
    *,
    lineage: tuple[str, ...] = (),
) -> str:
    """Hash one successful request's reusable prompt prefix and API lineage."""
    prompt = {
        key: value
        for key, value in api_kwargs.items()
        if key not in _NON_PREFIX_REQUEST_FIELDS
        and not (
            isinstance(key, str)
            and key.startswith("__")
            and key.endswith("__")
        )
    }
    if not prompt:
        return ""
    projection = {
        "lineage": [str(value or "").strip().rstrip("/") for value in lineage],
        "prompt": prompt,
    }
    payload = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_summary_runtime(agent: Any) -> SummaryRuntime:
    """Return a main-runtime bridge for append-cached compression summary calls."""
    from agent.chat_completion_helpers import (
        estimate_request_context_tokens,
        interruptible_api_call,
        prepare_provider_visible_messages,
    )

    def _build_kwargs(messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        old_ephemeral = getattr(agent, "_ephemeral_max_output_tokens", None)
        try:
            agent._ephemeral_max_output_tokens = max_tokens
            provider_messages = prepare_provider_visible_messages(
                agent,
                messages,
                copy_messages=True,
            )
            system_prompt = getattr(agent, "_cached_system_prompt", None)
            if isinstance(system_prompt, str) and system_prompt:
                ephemeral_system = getattr(agent, "ephemeral_system_prompt", None)
                if isinstance(ephemeral_system, str) and ephemeral_system:
                    system_prompt = (system_prompt + "\n\n" + ephemeral_system).strip()
            if (
                isinstance(system_prompt, str)
                and system_prompt
                and not (
                    provider_messages
                    and isinstance(provider_messages[0], dict)
                    and provider_messages[0].get("role") == "system"
                )
            ):
                provider_messages = [
                    {"role": "system", "content": system_prompt}
                ] + provider_messages
            prefill_messages = getattr(agent, "prefill_messages", None) or []
            if prefill_messages:
                system_offset = (
                    1
                    if provider_messages
                    and isinstance(provider_messages[0], dict)
                    and provider_messages[0].get("role") == "system"
                    else 0
                )
                provider_messages[system_offset:system_offset] = [
                    message.copy()
                    for message in prefill_messages
                    if isinstance(message, dict)
                ]
            if getattr(agent, "_use_prompt_caching", False):
                from agent.prompt_caching import apply_anthropic_cache_control

                provider_messages = apply_anthropic_cache_control(
                    provider_messages,
                    cache_ttl=getattr(agent, "_cache_ttl", "5m"),
                    native_anthropic=bool(
                        getattr(agent, "_use_native_cache_layout", False)
                    ),
                )
            api_kwargs = agent._build_api_kwargs(provider_messages)
            try:
                transport = agent._get_transport()
                api_kwargs = transport.preflight_kwargs(
                    api_kwargs,
                    allow_stream=False,
                )
            except (AttributeError, NotImplementedError):
                pass
            return api_kwargs
        finally:
            agent._ephemeral_max_output_tokens = old_ephemeral

    @contextmanager
    def _suppress_main_stream_callbacks():
        """Keep internal summarizer deltas out of user-facing streams."""
        stream_delta_callback = getattr(agent, "stream_delta_callback", None)
        stream_callback = getattr(agent, "_stream_callback", None)
        reasoning_callback = getattr(agent, "reasoning_callback", None)
        try:
            agent.stream_delta_callback = None
            agent._stream_callback = None
            agent.reasoning_callback = None
            yield
        finally:
            agent.stream_delta_callback = stream_delta_callback
            agent._stream_callback = stream_callback
            agent.reasoning_callback = reasoning_callback

    def _invoke(api_kwargs: dict[str, Any]) -> Any:
        with _suppress_main_stream_callbacks():
            return interruptible_api_call(agent, api_kwargs)

    def _fingerprint_prefix(messages: list[dict[str, Any]]) -> str:
        return fingerprint_cache_visible_prefix(
            _build_kwargs(messages, 1),
            lineage=(
                str(getattr(agent, "provider", "") or ""),
                str(getattr(agent, "model", "") or ""),
                str(getattr(agent, "api_mode", "") or ""),
                str(getattr(agent, "base_url", "") or ""),
                str(getattr(agent, "api_key", "") or ""),
            ),
        )

    def _activate_fallback(exc: BaseException) -> bool:
        activator = getattr(agent, "_try_activate_fallback", None)
        if not callable(activator):
            return False
        reason = None
        try:
            from agent.error_classifier import classify_api_error

            if isinstance(exc, Exception):
                reason = classify_api_error(
                    exc,
                    provider=str(getattr(agent, "provider", "") or ""),
                    model=str(getattr(agent, "model", "") or ""),
                ).reason
        except Exception:
            reason = None
        try:
            return bool(activator(reason))
        except Exception:
            return False

    fallback_chain = getattr(agent, "_fallback_chain", None) or []
    fallback_index = int(getattr(agent, "_fallback_index", 0) or 0)
    fallback_budget = max(0, len(fallback_chain) - fallback_index)

    return SummaryRuntime(
        provider=getattr(agent, "provider", "") or "",
        model=getattr(agent, "model", "") or "",
        api_mode=getattr(agent, "api_mode", "") or "",
        base_url=getattr(agent, "base_url", "") or "",
        reasoning_effort=getattr(agent, "reasoning_effort", None),
        context_limit_tokens=getattr(getattr(agent, "context_compressor", None), "context_length", None),
        tools_included=bool(getattr(agent, "tools", None)),
        main_api_calls_in_process=int(getattr(agent, "session_api_calls", 0) or 0),
        summary_runtime_shape=getattr(agent, "_summary_runtime_shape", None),
        summary_runtime_toolset_source=getattr(agent, "_summary_runtime_toolset_source", None),
        build_kwargs=_build_kwargs,
        fingerprint_prefix=_fingerprint_prefix,
        invoke=_invoke,
        estimate_request_tokens=estimate_request_context_tokens,
        activate_fallback=_activate_fallback,
        fallback_attempt_budget=fallback_budget,
    )


def apply_summary_tool_choice_none(
    api_kwargs: dict[str, Any],
    api_mode: str,
) -> tuple[dict[str, Any], bool]:
    """Return kwargs with provider-appropriate no-tool choice when tools are present."""
    if "tools" not in api_kwargs:
        return api_kwargs, False
    if api_mode == "codex_responses":
        # Codex Responses already emits ``tool_choice: auto`` when tools are
        # present. Keep the main-runtime payload shape unchanged for cache reuse
        # and rely on the summary instruction + post-response tool-call
        # validation instead of sending a provider-rejected no-tool override.
        return api_kwargs, False

    updated = dict(api_kwargs)
    if api_mode == "anthropic_messages":
        updated["tool_choice"] = {"type": "none"}
    else:
        updated["tool_choice"] = "none"
    return updated, True


def extract_summary_response_content(response: Any) -> tuple[str, bool]:
    """Return response text and whether the model attempted a tool call."""
    def _get(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    if part:
                        parts.append(part)
                    continue
                part_type = str(_get(part, "type", "") or "").strip().lower()
                if part_type not in {"output_text", "text"}:
                    continue
                text = _get(part, "text", "")
                if not isinstance(text, str):
                    text = str(text or "")
                if text:
                    parts.append(text)
            return "\n".join(parts).strip()
        return str(content) if content else ""

    choices = getattr(response, "choices", None) or []
    if not choices:
        output = _get(response, "output") or []
        content_parts: list[str] = []
        if isinstance(output, list):
            for item in output:
                item_type = str(_get(item, "type", "") or "").strip().lower()
                if item_type in {"function_call", "custom_tool_call"}:
                    return "", True
                if item_type != "message":
                    continue
                text = _content_text(_get(item, "content", ""))
                if text:
                    content_parts.append(text)
        if content_parts:
            return "\n".join(content_parts).strip(), False
        output_text = _get(response, "output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip(), False
        return "", False
    message = getattr(choices[0], "message", None)
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")
    else:
        tool_calls = getattr(message, "tool_calls", None) or []
        content = getattr(message, "content", message)
    if tool_calls:
        return "", True
    return _content_text(content), False


def extract_summary_cache_stats(response: Any) -> dict[str, Any]:
    """Normalize provider cache usage fields for compression audit records."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {
            "reported": False,
            "read_tokens": None,
            "write_tokens": None,
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "hit_rate_provider_actual": None,
            "hit_rate_estimate": None,
        }

    def _get(obj: Any, name: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def _has(obj: Any, name: str) -> bool:
        if obj is None:
            return False
        if isinstance(obj, dict):
            return name in obj
        return hasattr(obj, name)

    def _first_present(*paths: tuple[Any, str]) -> tuple[Any, bool]:
        for obj, name in paths:
            if _has(obj, name):
                value = _get(obj, name)
                if value is not None:
                    return value, True
        return None, False

    def _int_or_none(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details")
    read_tokens, read_reported = _first_present(
        (details, "cached_tokens"),
        (details, "cache_read_input_tokens"),
        (usage, "cache_read_tokens"),
        (usage, "cache_read_input_tokens"),
    )
    write_tokens, write_reported = _first_present(
        (details, "cache_write_tokens"),
        (details, "cache_creation_tokens"),
        (usage, "cache_write_tokens"),
        (usage, "cache_creation_input_tokens"),
        (usage, "cache_creation_tokens"),
    )
    prompt_tokens, _prompt_reported = _first_present(
        (usage, "prompt_tokens"),
        (usage, "input_tokens"),
    )
    output_tokens, _output_reported = _first_present(
        (usage, "completion_tokens"),
        (usage, "output_tokens"),
    )
    read_int = _int_or_none(read_tokens) if read_reported else None
    write_int = _int_or_none(write_tokens) if write_reported else None
    prompt_int = _int_or_none(prompt_tokens)
    output_int = _int_or_none(output_tokens)
    hit_rate = None
    if read_int is not None and prompt_int:
        hit_rate = round(read_int / max(prompt_int, 1), 4)
    elif read_int == 0 and prompt_int is not None:
        hit_rate = 0.0
    return {
        "reported": read_int is not None or write_int is not None,
        "read_tokens": read_int,
        "write_tokens": write_int,
        # New explicit provider-actual denominator fields. Keep the legacy
        # hit_rate_estimate key for existing audit readers, but make the actual
        # denominator available so callers do not divide by rough estimates.
        "provider_input_tokens": prompt_int,
        "provider_output_tokens": output_int,
        "hit_rate_provider_actual": hit_rate,
        "hit_rate_estimate": hit_rate,
    }
