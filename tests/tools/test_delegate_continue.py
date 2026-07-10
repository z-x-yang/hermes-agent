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
