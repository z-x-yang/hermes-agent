"""Task 5 retained subagent sessions and delegate_continue tests."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tools.subagent_sessions import (
    RetainedSubagentSession,
    clear_retained_subagent_sessions,
    get_retained_subagent_session,
    retain_subagent_session,
)


def setup_function():
    clear_retained_subagent_sessions()


def _parent(session_id: str = "parent-1", *, enabled_toolsets=None):
    return SimpleNamespace(
        session_id=session_id,
        model="model-a",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="parent-secret-key",
        api_mode="chat_completions",
        enabled_toolsets=enabled_toolsets,
        valid_tool_names={
            "terminal",
            "process",
            "read_file",
            "write_file",
            "patch",
            "search_files",
            "web_search",
            "web_extract",
            "delegate_task",
            "delegate_continue",
        },
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _memory_manager=None,
        _session_db=None,
        _print_fn=None,
        tool_progress_callback=None,
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        _current_turn_id="turn-1",
        _current_api_request_id="req-1",
    )


def _record(**overrides):
    now = time.time()
    data = {
        "agent_id": "agent-1",
        "parent_session_id": "parent-1",
        "subagent_type": "general-purpose",
        "role": "leaf",
        "workspace_path": "/tmp/repo",
        "model": "model-a",
        "provider": "openrouter",
        "conversation_history": [{"role": "user", "content": "first"}],
        "created_at": now,
        "expires_at": now + 60,
    }
    data.update(overrides)
    return RetainedSubagentSession(**data)


class _CompletedRetainableChild:
    session_id = "child-session"
    model = "model-a"
    provider = "openrouter"
    _delegate_role = "leaf"
    _subagent_id = "sa-test"
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_reasoning_tokens = 0
    tool_progress_callback = None

    def run_conversation(self, **kwargs):
        return {
            "final_response": "done",
            "completed": True,
            "api_calls": 1,
            "messages": [
                {"role": "user", "content": kwargs["user_message"]},
                {"role": "assistant", "content": "done"},
            ],
        }

    def get_activity_summary(self):
        return {"api_call_count": 1, "current_tool": None, "max_iterations": 1}

    def close(self):
        pass


def test_retained_session_round_trip_and_ttl():
    record = _record()
    retain_subagent_session(record)
    assert get_retained_subagent_session("agent-1") == record


def test_expired_session_fails_closed():
    record = _record(agent_id="expired", created_at=time.time() - 10, expires_at=time.time() - 1)
    retain_subagent_session(record)
    with pytest.raises(KeyError, match="expired"):
        get_retained_subagent_session("expired")


def test_retained_session_metadata_does_not_persist_api_keys():
    field_names = {field.name for field in dataclasses.fields(RetainedSubagentSession)}
    assert "api_key" not in field_names
    assert "secret" not in field_names
    assert {"model", "provider", "subagent_type", "role", "workspace_path"}.issubset(field_names)


def test_delegate_continue_schema_is_narrow_and_registered():
    from tools.delegate_continue_tool import DELEGATE_CONTINUE_SCHEMA
    from tools.registry import registry
    from toolsets import TOOLSETS

    props = DELEGATE_CONTINUE_SCHEMA["parameters"]["properties"]
    assert set(props) == {"agent_id", "prompt", "scheduling"}
    for forbidden in {"subagent_type", "role", "toolsets", "max_iterations", "timeout", "retain_session"}:
        assert forbidden not in props
    assert DELEGATE_CONTINUE_SCHEMA["parameters"]["required"] == ["agent_id", "prompt"]
    assert "delegate_continue" in TOOLSETS["delegation"]["tools"]

    definitions = registry.get_definitions({"delegate_continue"})
    assert len(definitions) == 1
    assert definitions[0]["function"]["parameters"]["properties"] == props


def test_delegate_continue_reuses_history_and_updates_retained_record(monkeypatch):
    from tools.delegate_continue_tool import delegate_continue

    captured = {}

    class FakeChild:
        session_id = "continued-session"
        model = "model-a"
        provider = "openrouter"
        session_prompt_tokens = 3
        session_completion_tokens = 4
        session_estimated_cost_usd = 0.0

        def run_conversation(self, **kwargs):
            captured.update(kwargs)
            return {
                "final_response": "continued",
                "messages": kwargs["conversation_history"]
                + [
                    {"role": "user", "content": kwargs["user_message"]},
                    {"role": "assistant", "content": "continued"},
                ],
                "api_calls": 1,
                "completed": True,
            }

        def close(self):
            pass

    retain_subagent_session(_record(subagent_type="Explore"))
    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        lambda *_args, **_kwargs: FakeChild(),
    )

    result = json.loads(
        delegate_continue(
            agent_id="agent-1",
            prompt="continue the same investigation",
            scheduling="foreground",
            parent_agent=_parent(),
        )
    )

    assert result["status"] == "completed"
    assert result["agent_id"] == "agent-1"
    assert captured["conversation_history"] == [{"role": "user", "content": "first"}]
    assert "continue the same investigation" in captured["user_message"]
    updated = get_retained_subagent_session("agent-1")
    assert updated.subagent_type == "Explore"
    assert updated.conversation_history[-1] == {"role": "assistant", "content": "continued"}


def test_delegate_continue_rejects_concurrent_same_agent_without_losing_history(monkeypatch):
    from tools.delegate_continue_tool import delegate_continue

    run_started = threading.Event()
    different_agent_started = threading.Event()
    allow_finish = threading.Event()
    worker_threads = []
    seen_histories = []
    seen_histories_lock = threading.Lock()

    class BlockingChild:
        model = "model-a"
        provider = "openrouter"

        def run_conversation(self, **kwargs):
            with seen_histories_lock:
                seen_histories.append(list(kwargs["conversation_history"]))
                if len(seen_histories) == 2:
                    different_agent_started.set()
            run_started.set()
            assert allow_finish.wait(2)
            return {
                "final_response": "continued",
                "messages": kwargs["conversation_history"]
                + [{"role": "assistant", "content": "continued"}],
                "api_calls": 1,
            }

        def close(self):
            pass

    def fake_dispatch(*, runner, **_kwargs):
        worker = threading.Thread(target=runner)
        worker.start()
        worker_threads.append(worker)
        return {"status": "dispatched", "delegation_id": f"d-{len(worker_threads)}"}

    retain_subagent_session(_record())
    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        lambda *_args, **_kwargs: BlockingChild(),
    )
    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", fake_dispatch)

    first = json.loads(
        delegate_continue("agent-1", "first follow-up", "background", parent_agent=_parent())
    )
    assert first["status"] == "dispatched"
    assert run_started.wait(2)

    retain_subagent_session(_record(agent_id="agent-2"))
    different = json.loads(
        delegate_continue("agent-2", "parallel follow-up", "background", parent_agent=_parent())
    )
    assert different["status"] == "dispatched"
    assert different_agent_started.wait(2)

    second = json.loads(
        delegate_continue("agent-1", "racing follow-up", "background", parent_agent=_parent())
    )
    assert second == {"error": "Retained subagent continuation already in progress: agent-1"}

    allow_finish.set()
    for worker in worker_threads:
        worker.join(timeout=2)
        assert not worker.is_alive()

    third = json.loads(
        delegate_continue("agent-1", "after first", "background", parent_agent=_parent())
    )
    assert third["status"] == "dispatched"
    worker_threads[-1].join(timeout=2)
    assert not worker_threads[-1].is_alive()
    assert len(seen_histories) == 3
    assert seen_histories[2][-1] == {"role": "assistant", "content": "continued"}


def test_delegate_continue_releases_claim_after_runner_exception(monkeypatch):
    from tools.delegate_continue_tool import delegate_continue

    dispatch_count = 0

    def fake_dispatch(*, runner, **_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        with pytest.raises(RuntimeError, match="runner boom"):
            runner()
        return {"status": "dispatched", "delegation_id": f"d-{dispatch_count}"}

    retain_subagent_session(_record())
    monkeypatch.setattr(
        "tools.delegate_continue_tool._run_continuation_entry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("runner boom")),
    )
    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", fake_dispatch)

    first = json.loads(
        delegate_continue("agent-1", "fails", "background", parent_agent=_parent())
    )
    assert first["status"] == "dispatched"

    second = json.loads(
        delegate_continue("agent-1", "retry", "background", parent_agent=_parent())
    )
    assert second["status"] == "dispatched"
    assert dispatch_count == 2


@pytest.mark.parametrize("sync_path", ["nested", "stateless", "capacity"])
def test_delegate_continue_releases_claim_after_sync_fallbacks(monkeypatch, sync_path):
    from tools.delegate_continue_tool import delegate_continue

    monkeypatch.setattr(
        "tools.delegate_continue_tool._run_continuation_entry",
        lambda record, prompt, _parent_agent, **_kwargs: {
            "status": "completed",
            "agent_id": record.agent_id,
            "summary": prompt,
            "duration_seconds": 0,
        },
    )
    parent = _parent()
    if sync_path == "nested":
        parent._delegate_depth = 1
    elif sync_path == "stateless":
        monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: False)
    else:
        monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
        monkeypatch.setattr(
            "tools.async_delegation.dispatch_async_delegation_batch",
            lambda **_kwargs: {"status": "rejected", "error": "capacity"},
        )

    retain_subagent_session(_record())
    scheduling = "auto" if sync_path == "nested" else "background"
    for prompt in ("first", "second"):
        result = json.loads(
            delegate_continue("agent-1", prompt, scheduling, parent_agent=parent)
        )
        assert result["status"] == "completed"


def test_delegate_continue_releases_claim_when_dispatch_raises(monkeypatch):
    from tools.delegate_continue_tool import delegate_continue

    dispatch_count = 0

    def fake_dispatch(**_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        if dispatch_count == 1:
            raise RuntimeError("dispatch boom")
        return {"status": "rejected", "error": "capacity"}

    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", fake_dispatch)
    monkeypatch.setattr(
        "tools.delegate_continue_tool._run_continuation_entry",
        lambda record, prompt, _parent_agent, **_kwargs: {
            "status": "completed",
            "agent_id": record.agent_id,
            "summary": prompt,
            "duration_seconds": 0,
        },
    )
    retain_subagent_session(_record())

    first = json.loads(
        delegate_continue("agent-1", "first", "background", parent_agent=_parent())
    )
    assert first["error"] == (
        "Failed to dispatch retained subagent continuation: dispatch boom"
    )

    second = json.loads(
        delegate_continue("agent-1", "retry", "background", parent_agent=_parent())
    )
    assert second["status"] == "completed"


def test_claimed_retained_session_survives_midflight_ttl_then_expires(monkeypatch):
    from tools.subagent_sessions import (
        claim_retained_subagent_session,
        release_retained_subagent_session,
        update_retained_history,
    )

    clock = {"now": 100.0}
    monkeypatch.setattr("tools.subagent_sessions.time.time", lambda: clock["now"])
    retain_subagent_session(
        _record(created_at=100.0, expires_at=101.0, conversation_history=[])
    )

    claimed = claim_retained_subagent_session("agent-1")
    assert claimed.agent_id == "agent-1"
    clock["now"] = 200.0
    update_retained_history("agent-1", [{"role": "assistant", "content": "late"}])
    release_retained_subagent_session("agent-1")

    with pytest.raises(KeyError, match="expired"):
        get_retained_subagent_session("agent-1")


def test_foreground_continue_applies_run_cap_but_background_does_not(monkeypatch):
    from tools.delegate_continue_tool import delegate_continue

    seen_run_caps = []
    payloads = []

    def fake_run_entry(
        record,
        prompt,
        parent_agent,
        *,
        child_run_timeout_seconds=None,
    ):
        seen_run_caps.append(child_run_timeout_seconds)
        return {
            "status": "completed",
            "agent_id": record.agent_id,
            "summary": prompt,
            "duration_seconds": 0,
        }

    def fake_dispatch(*, runner, **_kwargs):
        payloads.append(json.dumps(runner(), ensure_ascii=False))
        return {"status": "dispatched", "delegation_id": f"d-{len(payloads)}"}

    monkeypatch.setattr("tools.delegate_continue_tool._run_continuation_entry", fake_run_entry)
    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", fake_dispatch)
    monkeypatch.setattr(
        "tools.async_delegation.wait_for_async_delegation",
        lambda *_args, **_kwargs: payloads[-1],
    )
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_foreground_timeouts",
        lambda *_args, **_kwargs: (10, 0.25),
    )

    retain_subagent_session(_record(agent_id="foreground-agent"))
    foreground = json.loads(
        delegate_continue(
            "foreground-agent",
            "foreground",
            "foreground",
            parent_agent=_parent(),
        )
    )
    assert foreground["status"] == "completed"

    retain_subagent_session(_record(agent_id="background-agent"))
    background = json.loads(
        delegate_continue(
            "background-agent",
            "background",
            "background",
            parent_agent=_parent(),
        )
    )
    assert background["status"] == "dispatched"
    assert seen_run_caps == [0.25, None]


def test_run_continuation_entry_interrupts_on_foreground_run_cap(monkeypatch):
    from tools.delegate_continue_tool import _run_continuation_entry

    interrupted = threading.Event()

    class HangingChild:
        model = "model-a"
        provider = "openrouter"

        def run_conversation(self, **_kwargs):
            interrupted.wait(2)
            return {"final_response": "too late", "messages": [], "api_calls": 1}

        def get_activity_summary(self):
            return {"api_call_count": 1, "current_tool": None, "max_iterations": 1}

        def interrupt(self):
            interrupted.set()

        def close(self):
            pass

    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        lambda *_args, **_kwargs: HangingChild(),
    )

    entry = _run_continuation_entry(
        _record(),
        "foreground cap",
        _parent(),
        child_run_timeout_seconds=0.05,
    )

    assert entry["status"] == "timeout"
    assert entry["exit_reason"] == "timeout"
    assert "0.05s" in entry["error"]
    assert interrupted.is_set()


def test_delegate_continue_requires_owner_parent_session():
    from tools.delegate_continue_tool import delegate_continue

    retain_subagent_session(_record(parent_session_id="other-parent"))
    result = json.loads(
        delegate_continue(
            agent_id="agent-1",
            prompt="continue",
            scheduling="foreground",
            parent_agent=_parent("parent-1"),
        )
    )
    assert "error" in result
    assert "does not belong" in result["error"]


def test_delegate_continue_rejects_empty_parent_session_ownership():
    from tools.delegate_continue_tool import delegate_continue

    retain_subagent_session(_record())
    empty_parent_result = json.loads(
        delegate_continue(
            agent_id="agent-1",
            prompt="continue",
            scheduling="foreground",
            parent_agent=_parent(""),
        )
    )
    assert "error" in empty_parent_result
    assert "non-empty parent session" in empty_parent_result["error"]

    clear_retained_subagent_sessions()
    retain_subagent_session(_record(parent_session_id=""))
    empty_record_result = json.loads(
        delegate_continue(
            agent_id="agent-1",
            prompt="continue",
            scheduling="foreground",
            parent_agent=_parent("parent-1"),
        )
    )
    assert "error" in empty_record_result
    assert "non-empty parent session" in empty_record_result["error"]


def test_delegate_continue_rejects_invalid_scheduling():
    from tools.delegate_continue_tool import delegate_continue

    retain_subagent_session(_record())
    result = json.loads(
        delegate_continue(
            agent_id="agent-1",
            prompt="continue",
            scheduling="later",
            parent_agent=_parent(),
        )
    )
    assert result["error"] == "Invalid scheduling: later"


def test_build_continuation_child_preserves_explore_capability_ceiling(monkeypatch):
    from tools.delegate_continue_tool import _build_continuation_child
    from toolsets import TOOLSETS

    created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = "child-session"
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self.api_mode = kwargs["api_mode"]
            self._session_init_model_config = {}
            names = []
            for toolset in kwargs.get("enabled_toolsets") or []:
                names.extend(TOOLSETS.get(toolset, {}).get("tools", []))
            self.valid_tool_names = set(names)
            self.tools = [
                {"type": "function", "function": {"name": name, "parameters": {}}}
                for name in sorted(self.valid_tool_names)
            ]
            created.append(self)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    record = _record(subagent_type="Explore", role="leaf")
    child = _build_continuation_child(
        record,
        prompt="same investigation",
        parent_agent=_parent(enabled_toolsets=["terminal", "file", "web", "delegation"]),
    )

    assert child is created[0]
    assert getattr(child, "_subagent_tool_policy", None) is not None
    assert "read_file" in child.valid_tool_names
    assert "search_files" in child.valid_tool_names
    assert "write_file" not in child.valid_tool_names
    assert "patch" not in child.valid_tool_names
    assert "terminal" not in child.valid_tool_names
    assert "delegate_continue" not in child.valid_tool_names
    assert child._delegate_role == "leaf"
    assert child._subagent_profile.name == "Explore"
    assert "/tmp/repo" in child.kwargs["ephemeral_system_prompt"]


def test_build_continuation_child_isolates_retained_provider_transport(monkeypatch):
    from tools.delegate_continue_tool import _build_continuation_child

    captured = {}
    monkeypatch.setattr(
        "tools.delegate_tool._load_config",
        lambda: {
            "model": "global-model",
            "provider": "custom",
            "base_url": "https://old-global.example/v1",
            "api_key": "old-global-key",
            "api_mode": "chat_completions",
            "command": "old-global-acp",
            "args": ["--old"],
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "openrouter",
            "model": "retained-model",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "current-openrouter-key",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
        },
    )

    def fake_build_child_agent(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", fake_build_child_agent)

    parent = _parent()
    parent.provider = "custom"
    parent.base_url = "https://old-global.example/v1"
    parent.api_key = "old-global-key"
    record = _record(provider="openrouter", model="retained-model")

    _build_continuation_child(record, prompt="continue", parent_agent=parent)

    assert captured["override_provider"] == "openrouter"
    assert captured["override_base_url"] == "https://openrouter.ai/api/v1"
    assert captured["override_api_key"] == "current-openrouter-key"
    assert captured["override_api_mode"] == "chat_completions"
    assert captured["override_acp_command"] is None
    assert captured["override_acp_args"] == []


def test_run_single_child_retains_completed_general_purpose_session(monkeypatch):
    import tools.delegate_tool as dt

    class FakeChild:
        session_id = "child-session"
        model = "model-a"
        provider = "openrouter"
        _delegate_role = "leaf"
        _subagent_id = "sa-test"
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_estimated_cost_usd = 0.0
        session_reasoning_tokens = 0
        tool_progress_callback = None

        def run_conversation(self, **kwargs):
            return {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
                "messages": [
                    {"role": "user", "content": kwargs["user_message"]},
                    {"role": "assistant", "content": "done"},
                ],
            }

        def get_activity_summary(self):
            return {"api_call_count": 1, "current_tool": None, "max_iterations": 1}

        def close(self):
            pass

    monkeypatch.setattr(dt, "_get_retained_session_ttl", lambda: 60)
    monkeypatch.setattr(dt, "_get_max_retained_subagents", lambda: 64)

    entry = dt._run_single_child(
        task_index=0,
        goal="implement",
        child=FakeChild(),
        parent_agent=_parent(),
        context=None,
        child_timeout_override=30,
        retain_session=True,
        subagent_type="general-purpose",
        role="leaf",
        workspace_path="/tmp/repo",
    )

    assert entry["status"] == "completed"
    assert entry["agent_id"] == "child-session"
    record = get_retained_subagent_session("child-session")
    assert record.subagent_type == "general-purpose"
    assert record.role == "leaf"
    assert record.workspace_path == "/tmp/repo"
    assert record.conversation_history[-1] == {"role": "assistant", "content": "done"}


def test_run_single_child_does_not_retain_without_parent_session_id(monkeypatch):
    import tools.delegate_tool as dt

    monkeypatch.setattr(dt, "_get_retained_session_ttl", lambda: 60)
    monkeypatch.setattr(dt, "_get_max_retained_subagents", lambda: 64)

    entry = dt._run_single_child(
        task_index=0,
        goal="stateless request",
        child=_CompletedRetainableChild(),
        parent_agent=_parent(""),
        context=None,
        child_timeout_override=30,
        retain_session=True,
        subagent_type="general-purpose",
        role="leaf",
        workspace_path="/tmp/repo",
    )

    assert entry["status"] == "completed"
    assert "agent_id" not in entry
    assert "retained_until" not in entry
    with pytest.raises(KeyError, match="Unknown"):
        get_retained_subagent_session("child-session")


def test_live_agent_invoke_tool_dispatches_delegate_continue_with_parent_agent():
    from agent.agent_runtime_helpers import invoke_tool

    captured = {}

    class FakeAgent:
        session_id = "parent-live"
        valid_tool_names = {"delegate_continue"}
        enabled_toolsets = {"delegation"}
        disabled_toolsets = None
        _context_engine_tool_names = set()
        _memory_manager = None
        _current_turn_id = "turn-1"
        _current_api_request_id = "req-1"

        def _dispatch_delegate_continue(self, args):
            captured["self"] = self
            captured["args"] = dict(args)
            return json.dumps({"status": "completed", "agent_id": args["agent_id"]})

    agent = FakeAgent()
    result = json.loads(
        invoke_tool(
            agent,
            "delegate_continue",
            {"agent_id": "agent-1", "prompt": "continue", "scheduling": "foreground"},
            "task-1",
        )
    )

    assert result == {"status": "completed", "agent_id": "agent-1"}
    assert captured["self"] is agent
    assert captured["args"]["prompt"] == "continue"
