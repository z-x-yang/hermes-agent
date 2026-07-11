"""Task 5 retained subagent sessions and delegate_continue tests."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from types import SimpleNamespace

import pytest

from agent.subagent_tool_policy import ToolNamePolicy
from tools.registry import registry
from tools.subagent_sessions import (
    RetainedSubagentSession,
    clear_retained_subagent_sessions,
    get_retained_subagent_session,
    retain_subagent_session,
)
from tools.tool_effects import ResultRetention, ToolEffect, build_authority_snapshot


def setup_function():
    clear_retained_subagent_sessions()


def _set_authority(agent, names) -> None:
    import model_tools  # noqa: F401

    identities = {
        identity
        for name in names
        if isinstance((identity := registry.resolved_policy_identity(name)), str)
    }
    agent._parent_tool_authority_snapshot = build_authority_snapshot(
        identities, registry_generation=registry._generation
    )


def _attach_policy(agent, names, profile_name: str):
    names = frozenset(names)
    _set_authority(agent, names)
    allowed_effects = None
    if profile_name in {"Explore", "Plan"}:
        allowed_effects = frozenset(
            {ToolEffect.READ_LOCAL, ToolEffect.READ_REMOTE}
        )
    agent._subagent_tool_policy = ToolNamePolicy(
        allowed_names=names,
        allowed_effects=allowed_effects,
        authority_snapshot=agent._parent_tool_authority_snapshot,
        profile_name=profile_name,
    )
    return agent


def _parent(session_id: str = "parent-1", *, enabled_toolsets=None):
    parent = SimpleNamespace(
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
    _set_authority(parent, parent.valid_tool_names)
    return parent


def test_retained_session_has_no_role_field():
    assert "role" not in RetainedSubagentSession.__dataclass_fields__


def _record(**overrides):
    now = time.time()
    from agent.subagent_governance import load_governance_snapshot
    import model_tools  # noqa: F401

    governance = load_governance_snapshot()
    effective_names = frozenset(
        {
            "terminal",
            "process",
            "read_file",
            "write_file",
            "patch",
            "search_files",
            "web_search",
            "web_extract",
        }
    )
    original_identities = frozenset(
        identity
        for name in effective_names
        if isinstance((identity := registry.resolved_policy_identity(name)), str)
    )
    data = {
        "agent_id": "agent-1",
        "parent_session_id": "parent-1",
        "subagent_type": "general-purpose",
        "workspace_path": "/tmp",
        "model": "model-a",
        "provider": "openrouter",
        "conversation_history": [{"role": "user", "content": "first"}],
        "created_at": now,
        "expires_at": now + 60,
        "effective_allowed_tool_names": effective_names,
        "profile_id": governance.profile_id,
        "canonical_profile_home": str(governance.profile_home.resolve()),
        "original_policy_identities": original_identities,
        "original_governance_fingerprint": governance.fingerprint,
    }
    data.update(overrides)
    return RetainedSubagentSession(**data)


class _CompletedRetainableChild:
    session_id = "child-session"
    model = "model-a"
    provider = "openrouter"
    _subagent_id = "sa-test"
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_reasoning_tokens = 0
    tool_progress_callback = None
    valid_tool_names = {"read_file"}

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


def test_retained_session_history_is_deeply_isolated():
    nested_history = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "original"}],
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "/tmp/a"}}}
            ],
        }
    ]
    retain_subagent_session(_record(conversation_history=nested_history))

    nested_history[0]["content"][0]["text"] = "mutated source"
    nested_history[0]["tool_calls"][0]["function"]["arguments"]["path"] = "/tmp/source"
    fetched = get_retained_subagent_session("agent-1")
    fetched.conversation_history[0]["content"][0]["text"] = "mutated result"
    fetched.conversation_history[0]["tool_calls"][0]["function"]["arguments"]["path"] = "/tmp/result"

    stored = get_retained_subagent_session("agent-1")
    assert stored.conversation_history[0]["content"][0]["text"] == "original"
    assert stored.conversation_history[0]["tool_calls"][0]["function"]["arguments"] == {
        "path": "/tmp/a"
    }


def test_retained_session_capacity_fails_closed_while_every_record_is_claimed():
    import tools.subagent_sessions as sessions
    from tools.subagent_sessions import (
        claim_retained_subagent_session,
        release_retained_subagent_session,
    )

    retain_subagent_session(_record(agent_id="in-flight"), max_records=1)
    claim_retained_subagent_session("in-flight")

    with pytest.raises(
        RuntimeError,
        match=r"Retained subagent session capacity reached \(1 records\)",
    ):
        retain_subagent_session(_record(agent_id="new"), max_records=1)

    assert len(sessions._records) == 1
    assert get_retained_subagent_session("in-flight").agent_id == "in-flight"
    with pytest.raises(KeyError, match="Unknown"):
        get_retained_subagent_session("new")

    release_retained_subagent_session("in-flight")
    retain_subagent_session(_record(agent_id="new"), max_records=1)

    assert len(sessions._records) == 1
    assert get_retained_subagent_session("new").agent_id == "new"
    with pytest.raises(KeyError, match="Unknown"):
        get_retained_subagent_session("in-flight")


def test_oversized_initial_retention_fails_closed():
    history = [{"role": "user", "content": "x" * 512}]
    with pytest.raises(RuntimeError, match="exceeds retained transcript byte budget"):
        retain_subagent_session(
            _record(agent_id="oversized", conversation_history=history),
            max_total_bytes=128,
        )

    with pytest.raises(KeyError, match="Unknown"):
        get_retained_subagent_session("oversized")


def test_retained_transcript_aggregate_byte_budget_prunes_only_as_needed():
    from tools.subagent_sessions import retained_subagent_transcript_bytes

    first_history = [{"role": "user", "content": "a" * 80}]
    second_history = [{"role": "user", "content": "b" * 80}]
    first_bytes = len(
        json.dumps(
            first_history, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )
    second_bytes = len(
        json.dumps(
            second_history, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )
    budget = first_bytes + second_bytes - 1

    retain_subagent_session(
        _record(agent_id="first", conversation_history=first_history),
        max_total_bytes=budget,
    )
    retain_subagent_session(
        _record(agent_id="second", conversation_history=second_history),
        max_total_bytes=budget,
    )

    assert retained_subagent_transcript_bytes() <= budget
    with pytest.raises(KeyError, match="Unknown"):
        get_retained_subagent_session("first")
    assert get_retained_subagent_session("second").agent_id == "second"


def test_retained_transcript_byte_pruning_never_evicts_claimed_record():
    from tools.subagent_sessions import (
        claim_retained_subagent_session,
        release_retained_subagent_session,
    )

    history = [{"role": "user", "content": "c" * 80}]
    record_bytes = len(
        json.dumps(
            history, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )
    retain_subagent_session(
        _record(agent_id="claimed", conversation_history=history),
        max_total_bytes=record_bytes * 2,
    )
    claim_retained_subagent_session("claimed")
    try:
        with pytest.raises(RuntimeError, match="all retained sessions are in flight"):
            retain_subagent_session(
                _record(agent_id="new", conversation_history=history),
                max_total_bytes=record_bytes * 2 - 1,
            )
        assert get_retained_subagent_session("claimed").agent_id == "claimed"
        with pytest.raises(KeyError, match="Unknown"):
            get_retained_subagent_session("new")
    finally:
        release_retained_subagent_session("claimed")


def test_retained_transcript_budget_is_config_authoritative_and_not_model_facing(
    monkeypatch,
):
    from hermes_cli.config import DEFAULT_CONFIG
    import tools.delegate_tool as delegate_tool
    from tools.delegate_continue_tool import DELEGATE_CONTINUE_SCHEMA

    assert DEFAULT_CONFIG["delegation"]["max_retained_subagent_bytes"] == 16777216
    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {"max_retained_subagent_bytes": 321},
    )
    assert delegate_tool._get_max_retained_subagent_bytes() == 321

    from tools.delegate_tool import DELEGATE_TASK_SCHEMA

    assert (
        "max_retained_subagent_bytes"
        not in DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    )
    assert (
        "max_retained_subagent_bytes"
        not in DELEGATE_CONTINUE_SCHEMA["parameters"]["properties"]
    )


def test_successful_oversized_continuation_drops_retention_but_keeps_result(
    monkeypatch,
):
    from tools.delegate_continue_tool import _run_continuation_entry, delegate_continue

    class OversizedContinuationChild:
        model = "model-a"
        provider = "openrouter"

        def run_conversation(self, **kwargs):
            return {
                "final_response": "successful result survives",
                "messages": kwargs["conversation_history"]
                + [{"role": "assistant", "content": "z" * 1024}],
                "api_calls": 1,
            }

        def close(self):
            pass

    retain_subagent_session(_record())
    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        lambda *_args, **_kwargs: OversizedContinuationChild(),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._get_max_retained_subagent_bytes",
        lambda: 128,
    )

    entry = _run_continuation_entry(
        get_retained_subagent_session("agent-1"),
        "continue successfully",
        _parent(),
    )

    assert entry["status"] == "completed"
    assert entry["summary"] == "successful result survives"
    assert entry["retention_dropped"] is True
    assert entry["note"] == (
        "Successful continuation exceeded the retained transcript byte budget; "
        "the result is preserved, but agent-1 is no longer resumable."
    )
    retry = json.loads(
        delegate_continue("agent-1", "retry", "foreground", parent_agent=_parent())
    )
    assert retry == {
        "error": (
            "Successful continuation exceeded the retained transcript byte budget; "
            "the result is preserved, but agent-1 is no longer resumable."
        )
    }


def test_expired_session_fails_closed():
    record = _record(agent_id="expired", created_at=time.time() - 10, expires_at=time.time() - 1)
    retain_subagent_session(record)
    with pytest.raises(KeyError, match="expired"):
        get_retained_subagent_session("expired")


def test_retained_session_metadata_does_not_persist_api_keys():
    field_names = {field.name for field in dataclasses.fields(RetainedSubagentSession)}
    assert "api_key" not in field_names
    assert "secret" not in field_names
    assert {"model", "provider", "subagent_type", "workspace_path"}.issubset(
        field_names
    )
    assert "role" not in field_names


def test_delegate_continue_schema_is_narrow_and_registered():
    from tools.delegate_continue_tool import DELEGATE_CONTINUE_SCHEMA
    from tools.registry import registry
    from toolsets import TOOLSETS

    props = DELEGATE_CONTINUE_SCHEMA["parameters"]["properties"]
    assert set(props) == {"agent_id", "prompt", "scheduling"}
    assert props["scheduling"]["enum"] == ["auto", "foreground", "background"]
    for forbidden in {
        "subagent_type",
        "role",
        "toolsets",
        "model",
        "provider",
        "max_iterations",
        "timeout",
        "foreground_wait_timeout_seconds",
        "child_run_timeout_seconds",
        "max_foreground_wait_timeout_seconds",
        "on_foreground_wait_timeout",
        "retain_session",
    }:
        assert forbidden not in props
    assert DELEGATE_CONTINUE_SCHEMA["parameters"]["required"] == ["agent_id", "prompt"]
    description = DELEGATE_CONTINUE_SCHEMA["description"]
    assert "instead of spawning a new child" in description
    assert "completed delegate_task result returned an agent_id" in description
    assert "same retained history" in description
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
    assert updated.status == "completed"
    assert updated.updated_at > updated.created_at
    assert updated.conversation_history[-1] == {"role": "assistant", "content": "continued"}


def test_continuation_reloads_modified_current_active_governance(monkeypatch, tmp_path):
    import agent.subagent_governance as governance
    import tools.delegate_continue_tool as continue_tool
    from tools.delegate_tool import _build_child_system_prompt

    memories = tmp_path / "memories"
    memories.mkdir()
    (tmp_path / "SOUL.md").write_text("SOUL-V1\n", encoding="utf-8")
    (memories / "MEMORY.md").write_text("MEMORY-CANARY\n", encoding="utf-8")
    (memories / "USER.md").write_text("USER-CANARY\n", encoding="utf-8")
    monkeypatch.setattr(governance, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(governance, "get_active_profile_name", lambda: "active-test")

    prompts = []
    fingerprints = []

    def fake_build_child_agent(*_args, **kwargs):
        snapshot = kwargs["governance_snapshot"]
        fingerprints.append(snapshot.fingerprint)
        prompts.append(
            _build_child_system_prompt(
                profile=kwargs["profile"],
                allow_delegation=False,
                workspace_path=kwargs["workspace_path_override"],
                child_depth=1,
                max_spawn_depth=1,
                governance_snapshot=snapshot,
            )
        )
        return _attach_policy(
            SimpleNamespace(valid_tool_names={"read_file"}),
            {"read_file"},
            kwargs["profile"].name,
        )

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", fake_build_child_agent)
    record = _record(
        subagent_type="Explore",
        workspace_path=str(tmp_path),
        effective_allowed_tool_names=frozenset({"read_file"}),
        profile_id="active-test",
        canonical_profile_home=str(tmp_path.resolve()),
    )
    parent = _parent()

    continue_tool._build_continuation_child(record, prompt="first", parent_agent=parent)
    (tmp_path / "SOUL.md").write_text("SOUL-V2-MODIFIED\n", encoding="utf-8")
    continue_tool._build_continuation_child(record, prompt="second", parent_agent=parent)

    assert "SOUL-V1\n" in prompts[0]
    assert "SOUL-V2-MODIFIED\n" in prompts[1]
    assert "SOUL-V1\n" not in prompts[1]
    assert fingerprints[0] != fingerprints[1]


def test_continuation_rejects_profile_or_canonical_home_drift_before_child_build(
    monkeypatch,
):
    import tools.delegate_continue_tool as continue_tool

    record = _record(profile_id="retained-profile")
    latest = SimpleNamespace(
        profile_id="different-profile",
        profile_home=record.canonical_profile_home,
        fingerprint="b" * 64,
    )
    backend_calls = []
    monkeypatch.setattr(continue_tool, "load_governance_snapshot", lambda: latest)
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent",
        lambda *_args, **_kwargs: backend_calls.append(True),
    )

    with pytest.raises(ValueError, match="profile changed"):
        continue_tool._build_continuation_child(
            record,
            prompt="continue",
            parent_agent=_parent(),
        )
    assert backend_calls == []


def test_continuation_intersects_original_and_current_exact_policy_identities(
    monkeypatch,
):
    import tools.delegate_continue_tool as continue_tool

    current = continue_tool.load_governance_snapshot()
    kept = "policy:" + "1" * 64
    original_only = "policy:" + "2" * 64
    replacement = "policy:" + "3" * 64
    record = _record(
        effective_allowed_tool_names=frozenset({"read_file"}),
        original_policy_identities=frozenset({kept, original_only}),
    )
    child = SimpleNamespace(valid_tool_names={"read_file"})
    child._subagent_tool_policy = ToolNamePolicy(
        allowed_names=frozenset({"read_file"}),
        allowed_effects=None,
        authority_snapshot=build_authority_snapshot(
            {kept, replacement}, registry_generation=registry._generation
        ),
        profile_name="general-purpose",
    )
    monkeypatch.setattr(continue_tool, "load_governance_snapshot", lambda: current)
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda *_a, **_k: child)

    rebuilt = continue_tool._build_continuation_child(
        record,
        prompt="continue",
        parent_agent=_parent(),
    )

    assert rebuilt._subagent_tool_policy.authority_snapshot.policy_identities == {
        kept
    }


def test_continuation_runtime_pins_retained_workspace_and_clears_task_override(
    monkeypatch, tmp_path
):
    import tools.file_tools as file_tools
    import tools.terminal_tool as terminal_tool
    from tools.delegate_continue_tool import _run_continuation_entry

    workspace_a = (tmp_path / "workspace-a").resolve()
    workspace_b = (tmp_path / "workspace-b").resolve()
    workspace_a.mkdir()
    workspace_b.mkdir()
    monkeypatch.chdir(workspace_b)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    seen = {}

    class WorkspaceCheckingChild:
        model = "model-a"
        provider = "openrouter"

        def run_conversation(self, **kwargs):
            task_id = kwargs["task_id"]
            seen["task_id"] = task_id
            seen["cwd"] = terminal_tool._task_env_overrides[task_id]["cwd"]
            seen["resolved_relative"] = str(
                file_tools._resolve_path_for_task("marker.txt", task_id=task_id)
            )
            return {
                "final_response": "continued in retained workspace",
                "messages": kwargs["conversation_history"]
                + [{"role": "assistant", "content": "continued"}],
                "api_calls": 1,
            }

        def close(self):
            pass

    retain_subagent_session(_record(workspace_path=str(workspace_a)))
    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        lambda *_args, **_kwargs: WorkspaceCheckingChild(),
    )

    entry = _run_continuation_entry(
        get_retained_subagent_session("agent-1"),
        "continue in A",
        _parent(),
    )

    assert entry["status"] == "completed"
    assert seen["task_id"].startswith("delegation-continue-agent-1-")
    assert seen["cwd"] == str(workspace_a)
    assert seen["resolved_relative"] == str(workspace_a / "marker.txt")
    assert seen["task_id"] not in terminal_tool._task_env_overrides


def test_delegate_continue_invalid_workspace_fails_closed_and_invalidates(monkeypatch):
    from tools.delegate_continue_tool import delegate_continue

    missing = "/definitely/missing/hermes-retained-workspace"
    retain_subagent_session(_record(workspace_path=missing))
    first = json.loads(
        delegate_continue("agent-1", "continue", "foreground", parent_agent=_parent())
    )
    assert first["error"] == (
        "Retained subagent workspace is invalid or unavailable: "
        f"{missing}"
    )

    second = json.loads(
        delegate_continue("agent-1", "retry", "foreground", parent_agent=_parent())
    )
    assert second == first


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


def test_background_continue_interrupt_all_reaches_child_and_releases_claim(monkeypatch):
    import tools.async_delegation as async_delegation
    from tools.delegate_continue_tool import delegate_continue
    from tools.subagent_sessions import (
        claim_retained_subagent_session,
        release_retained_subagent_session,
    )

    run_started = threading.Event()
    interrupted = threading.Event()
    interrupt_calls = []

    class InterruptibleChild:
        model = "model-a"
        provider = "openrouter"
        _interrupt_requested = False

        def run_conversation(self, **_kwargs):
            run_started.set()
            assert interrupted.wait(2)
            return {"final_response": "", "messages": [], "api_calls": 1}

        def interrupt(self):
            self._interrupt_requested = True
            interrupt_calls.append(True)
            interrupted.set()

        def close(self):
            pass

    child = InterruptibleChild()
    async_delegation._reset_for_tests()
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        lambda *_args, **_kwargs: child,
    )
    retain_subagent_session(_record())

    try:
        result = json.loads(
            delegate_continue("agent-1", "keep going", "background", parent_agent=_parent())
        )
        assert result["status"] == "dispatched"
        assert run_started.wait(2)

        with async_delegation._records_lock:
            async_record = async_delegation._records[result["delegation_id"]]
            assert callable(async_record["interrupt_fn"])
            done_event = async_record["done_event"]

        assert async_delegation.interrupt_all("test") == 1
        assert interrupted.wait(2)
        assert done_event.wait(2)
        assert child._interrupt_requested is True
        assert len(interrupt_calls) == 1

        claimed = claim_retained_subagent_session("agent-1")
        assert claimed.agent_id == "agent-1"
        release_retained_subagent_session("agent-1")
    finally:
        interrupted.set()
        async_delegation._reset_for_tests()


def test_background_continue_latches_interrupt_until_child_is_built(monkeypatch):
    import tools.async_delegation as async_delegation
    from tools.delegate_continue_tool import delegate_continue

    build_entered = threading.Event()
    allow_build = threading.Event()
    interrupted = threading.Event()
    interrupt_calls = []
    done_event = None

    class InterruptibleChild:
        model = "model-a"
        provider = "openrouter"
        _interrupt_requested = False

        def run_conversation(self, **_kwargs):
            assert self._interrupt_requested is True
            return {"final_response": "", "messages": [], "api_calls": 1}

        def interrupt(self):
            self._interrupt_requested = True
            interrupt_calls.append(True)
            interrupted.set()

        def close(self):
            pass

    child = InterruptibleChild()

    def blocking_build(*_args, **_kwargs):
        build_entered.set()
        assert allow_build.wait(2)
        return child

    async_delegation._reset_for_tests()
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        blocking_build,
    )
    retain_subagent_session(_record())

    try:
        result = json.loads(
            delegate_continue("agent-1", "keep going", "background", parent_agent=_parent())
        )
        assert result["status"] == "dispatched"
        assert build_entered.wait(2)

        with async_delegation._records_lock:
            async_record = async_delegation._records[result["delegation_id"]]
            assert callable(async_record["interrupt_fn"])
            done_event = async_record["done_event"]

        assert async_delegation.interrupt_all("test pre-build") == 1
        assert not interrupted.is_set()
        allow_build.set()

        assert interrupted.wait(2)
        assert done_event.wait(2)
        assert child._interrupt_requested is True
        assert len(interrupt_calls) == 1
    finally:
        allow_build.set()
        interrupted.set()
        if done_event is not None:
            done_event.wait(2)
        async_delegation._reset_for_tests()


@pytest.mark.parametrize("scheduling", ["background", "foreground"])
def test_top_level_async_continue_is_never_parent_lifecycle_owned(
    monkeypatch, scheduling
):
    import tools.async_delegation as async_delegation
    from tools.delegate_continue_tool import delegate_continue

    run_started = threading.Event()
    allow_finish = threading.Event()
    bridge_interrupt = threading.Event()
    parent_lifecycle_interrupt = threading.Event()
    registration_flags = []
    responses = []
    child = None

    class BlockingChild:
        model = "model-a"
        provider = "openrouter"

        def run_conversation(self, **kwargs):
            run_started.set()
            assert allow_finish.wait(2)
            return {
                "final_response": "continued",
                "messages": kwargs["conversation_history"]
                + [{"role": "assistant", "content": "continued"}],
                "api_calls": 1,
            }

        def interrupt(self, *args):
            if args:
                parent_lifecycle_interrupt.set()
            else:
                bridge_interrupt.set()
            allow_finish.set()

        def close(self):
            pass

    child = BlockingChild()

    def fake_build(*_args, parent_agent, register_with_parent=True, **_kwargs):
        registration_flags.append(register_with_parent)
        if register_with_parent:
            parent_agent._active_children.append(child)
        return child

    async_delegation._reset_for_tests()
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        fake_build,
    )
    if scheduling == "foreground":
        monkeypatch.setattr(
            "tools.delegate_tool._resolve_foreground_timeouts",
            lambda *_args, **_kwargs: (60, 60),
        )
    parent = _parent()
    retain_subagent_session(_record())

    caller = threading.Thread(
        target=lambda: responses.append(
            json.loads(
                delegate_continue(
                    "agent-1",
                    "keep going",
                    scheduling,
                    parent_agent=parent,
                )
            )
        )
    )
    caller.start()
    try:
        assert run_started.wait(2)
        assert child not in parent._active_children

        for active_child in list(parent._active_children):
            active_child.interrupt("parent lifecycle release")
        assert not parent_lifecycle_interrupt.is_set()

        assert async_delegation.interrupt_all("async registry owner") == 1
        assert bridge_interrupt.wait(2)
    finally:
        allow_finish.set()
        caller.join(timeout=2)
        with async_delegation._records_lock:
            futures = [record["future"] for record in async_delegation._records.values()]
        for future in futures:
            future.result(timeout=2)
        async_delegation._reset_for_tests()

    assert not caller.is_alive()
    assert registration_flags == [False]
    assert responses


@pytest.mark.parametrize("sync_path", ["nested", "stateless"])
def test_sync_continue_fallbacks_remain_parent_attached_while_running(
    monkeypatch, sync_path
):
    from tools.delegate_continue_tool import delegate_continue

    run_started = threading.Event()
    allow_finish = threading.Event()
    registration_flags = []
    responses = []
    parent = _parent()

    class BlockingChild:
        model = "model-a"
        provider = "openrouter"

        def run_conversation(self, **kwargs):
            run_started.set()
            assert allow_finish.wait(2)
            return {
                "final_response": "continued inline",
                "messages": kwargs["conversation_history"]
                + [{"role": "assistant", "content": "continued inline"}],
                "api_calls": 1,
            }

        def close(self):
            pass

    child = BlockingChild()

    def fake_build(*_args, parent_agent, register_with_parent=True, **_kwargs):
        registration_flags.append(register_with_parent)
        if register_with_parent:
            parent_agent._active_children.append(child)
        return child

    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        fake_build,
    )
    if sync_path == "nested":
        parent._delegate_depth = 1
        scheduling = "auto"
    elif sync_path == "stateless":
        monkeypatch.setattr(
            "gateway.session_context.async_delivery_supported", lambda: False
        )
        scheduling = "background"
    else:
        monkeypatch.setattr(
            "gateway.session_context.async_delivery_supported", lambda: True
        )
        monkeypatch.setattr(
            "tools.async_delegation.dispatch_async_delegation_batch",
            lambda **_kwargs: {"status": "rejected", "error": "capacity"},
        )
        scheduling = "background"

    retain_subagent_session(_record())
    caller = threading.Thread(
        target=lambda: responses.append(
            json.loads(
                delegate_continue(
                    "agent-1",
                    "continue inline",
                    scheduling,
                    parent_agent=parent,
                )
            )
        )
    )
    caller.start()
    try:
        assert run_started.wait(2)
        assert registration_flags == [True]
        assert child in parent._active_children
    finally:
        allow_finish.set()
        caller.join(timeout=2)

    assert not caller.is_alive()
    assert child not in parent._active_children
    assert responses[0]["status"] == "completed"


@pytest.mark.parametrize("sync_path", ["nested", "stateless"])
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


def test_delegate_continue_capacity_rejection_runs_no_child_and_releases_claim(
    monkeypatch,
):
    from tools.delegate_continue_tool import delegate_continue

    executions = []
    dispatches = []

    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "tools.delegate_continue_tool._run_continuation_entry",
        lambda *_args, **_kwargs: executions.append(True),
    )

    def reject(**_kwargs):
        dispatches.append(True)
        return {"status": "rejected", "error": "capacity"}

    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch", reject
    )
    retain_subagent_session(_record())

    first = json.loads(
        delegate_continue("agent-1", "first", "background", parent_agent=_parent())
    )
    second = json.loads(
        delegate_continue("agent-1", "retry", "background", parent_agent=_parent())
    )

    assert first == second == {
        "status": "rejected",
        "mode": "background",
        "agent_id": "agent-1",
        "error": "capacity",
    }
    assert executions == []
    assert len(dispatches) == 2


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
    assert second == {
        "status": "rejected",
        "mode": "background",
        "agent_id": "agent-1",
        "error": "capacity",
    }


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
        interrupt_bridge=None,
        register_with_parent=True,
    ):
        assert interrupt_bridge is not None
        assert register_with_parent is False
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


def test_delegate_continue_foreground_parent_interrupt_returns_without_late_queue(
    monkeypatch, tmp_path
):
    import tools.async_delegation as async_delegation
    from tools.delegate_continue_tool import delegate_continue
    from tools.process_registry import process_registry

    run_started = threading.Event()
    allow_finish = threading.Event()
    interrupt_called = threading.Event()
    parent = _parent()
    parent._interrupt_requested = False
    response = []

    class InterruptibleChild:
        model = "model-a"
        provider = "openrouter"

        def run_conversation(self, **_kwargs):
            run_started.set()
            assert allow_finish.wait(2)
            return {
                "final_response": "too late",
                "messages": [{"role": "assistant", "content": "too late"}],
                "api_calls": 1,
            }

        def interrupt(self):
            interrupt_called.set()

        def close(self):
            pass

    async_delegation._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        lambda *_args, **_kwargs: InterruptibleChild(),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_foreground_timeouts",
        lambda *_args, **_kwargs: (60, 60),
    )
    retain_subagent_session(_record(workspace_path=str(tmp_path)))

    try:
        thread = threading.Thread(
            target=lambda: response.append(
                json.loads(
                    delegate_continue(
                        "agent-1",
                        "interrupt me",
                        "foreground",
                        parent_agent=parent,
                    )
                )
            )
        )
        thread.start()
        assert run_started.wait(1)
        parent._interrupt_requested = True
        thread.join(timeout=1)

        assert not thread.is_alive()
        assert response[0]["status"] == "interrupted"
        assert response[0]["error"] == "Foreground delegation interrupted by parent."
        assert interrupt_called.is_set()
        assert process_registry.completion_queue.empty()

        with async_delegation._records_lock:
            future = async_delegation._records[response[0]["delegation_id"]]["future"]
        allow_finish.set()
        future.result(timeout=2)
        assert process_registry.completion_queue.empty()
        retained = get_retained_subagent_session("agent-1")
        assert retained.conversation_history == [
            {"role": "user", "content": "first"}
        ]
    finally:
        allow_finish.set()
        async_delegation._reset_for_tests()
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


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


def test_foreground_run_cap_poisoned_session_stays_non_resumable_while_worker_lives(
    monkeypatch, tmp_path
):
    import tools.async_delegation as async_delegation
    from tools.delegate_continue_tool import delegate_continue

    run_started = threading.Event()
    release_worker = threading.Event()
    worker_exited = threading.Event()
    interrupt_called = threading.Event()

    class InterruptIgnoringChild:
        model = "model-a"
        provider = "openrouter"

        def run_conversation(self, **_kwargs):
            run_started.set()
            assert release_worker.wait(2)
            worker_exited.set()
            return {"final_response": "too late", "messages": [], "api_calls": 1}

        def get_activity_summary(self):
            return {"api_call_count": 1, "current_tool": None, "max_iterations": 1}

        def interrupt(self):
            interrupt_called.set()

        def close(self):
            pass

    async_delegation._reset_for_tests()
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        lambda *_args, **_kwargs: InterruptIgnoringChild(),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_foreground_timeouts",
        lambda *_args, **_kwargs: (1, 0.05),
    )
    retain_subagent_session(_record(workspace_path=str(tmp_path)))

    try:
        timed_out = json.loads(
            delegate_continue(
                "agent-1", "foreground cap", "foreground", parent_agent=_parent()
            )
        )
        assert run_started.is_set()
        assert interrupt_called.is_set()
        assert timed_out["status"] == "timeout"
        assert timed_out["retention_dropped"] is True
        assert "no longer resumable" in timed_out["note"]
        assert not worker_exited.is_set()

        second = json.loads(
            delegate_continue(
                "agent-1", "must not race stale history", "foreground", parent_agent=_parent()
            )
        )
        assert second == {
            "error": (
                "Retained subagent session is no longer resumable after timeout: "
                "agent-1"
            )
        }
    finally:
        release_worker.set()
        assert worker_exited.wait(2)
        async_delegation._reset_for_tests()

    with pytest.raises(RuntimeError, match="no longer resumable after timeout"):
        get_retained_subagent_session("agent-1")


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


def test_build_continuation_child_warns_after_prior_file_writes(monkeypatch):
    from tools.delegate_continue_tool import _build_continuation_child

    child = _attach_policy(
        SimpleNamespace(
            valid_tool_names={"read_file"},
            tools=[
                {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {}},
                }
            ],
            ephemeral_system_prompt="base",
        ),
        {"read_file"},
        "general-purpose",
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent",
        lambda **_kwargs: child,
    )
    data = dataclasses.asdict(_record())
    data["files_written"] = ("/tmp/repo/changed.py",)

    built = _build_continuation_child(
        SimpleNamespace(**data),
        prompt="keep editing",
        parent_agent=_parent(),
    )

    assert "workspace may have changed" in built.ephemeral_system_prompt
    assert "verify the current diff/state" in built.ephemeral_system_prompt
    assert "/tmp/repo/changed.py" in built.ephemeral_system_prompt


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
            _set_authority(self, self.valid_tool_names)
            created.append(self)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    record = _record(subagent_type="Explore")
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
    assert not hasattr(child, "_delegate_role")
    assert child._subagent_profile.name == "Explore"
    assert "/tmp" in child.kwargs["ephemeral_system_prompt"]


def test_build_continuation_child_preserves_original_effective_tool_ceiling(monkeypatch):
    from tools.delegate_continue_tool import _build_continuation_child
    from toolsets import TOOLSETS

    class FakeAgent:
        def __init__(self, **kwargs):
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
            _set_authority(self, self.valid_tool_names)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    record = _record(
        subagent_type="general-purpose",
        effective_allowed_tool_names=frozenset({"read_file"}),
    )

    child = _build_continuation_child(
        record,
        prompt="continue narrowly",
        parent_agent=_parent(enabled_toolsets=["terminal", "file", "web"]),
    )

    assert child.valid_tool_names == {"read_file"}
    assert child._subagent_tool_policy.allowed_names == frozenset({"read_file"})


def test_build_continuation_child_intersects_exact_current_parent_tool_names(monkeypatch):
    from tools.delegate_continue_tool import _build_continuation_child
    from toolsets import TOOLSETS

    class FakeAgent:
        def __init__(self, **kwargs):
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
            _set_authority(self, self.valid_tool_names)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    parent = _parent(enabled_toolsets=["file"])
    parent.valid_tool_names = {"read_file"}
    _set_authority(parent, parent.valid_tool_names)
    record = _record(
        subagent_type="general-purpose",
        effective_allowed_tool_names=frozenset(
            {"read_file", "search_files", "write_file", "patch"}
        ),
    )

    child = _build_continuation_child(
        record,
        prompt="continue under the narrower live parent",
        parent_agent=parent,
    )

    assert child.valid_tool_names == {"read_file"}
    assert child._subagent_tool_policy.allowed_names == frozenset({"read_file"})


@pytest.mark.parametrize("subagent_type", [None, "general-purpose"])
@pytest.mark.parametrize(
    (
        "original_has_delegate_task",
        "parent_has_delegate_task",
        "orchestrator_enabled",
        "expected_delegation",
    ),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_continuation_delegation_requires_original_and_current_authority(
    monkeypatch,
    subagent_type,
    original_has_delegate_task,
    parent_has_delegate_task,
    orchestrator_enabled,
    expected_delegation,
):
    from tools.delegate_continue_tool import _build_continuation_child
    from toolsets import TOOLSETS

    class FakeAgent:
        def __init__(self, **kwargs):
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
            _set_authority(self, self.valid_tool_names)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        "tools.delegate_tool._get_orchestrator_enabled",
        lambda: orchestrator_enabled,
    )
    monkeypatch.setattr("tools.delegate_tool._get_max_spawn_depth", lambda: 2)
    retained_names = {"read_file", "delegate_continue"}
    if original_has_delegate_task:
        retained_names.add("delegate_task")
    parent_toolsets = ["file"]
    parent_names = {"read_file"}
    if parent_has_delegate_task:
        parent_toolsets.append("delegation")
        parent_names.add("delegate_task")
    parent = _parent(enabled_toolsets=parent_toolsets)
    parent.valid_tool_names = parent_names
    _set_authority(parent, parent.valid_tool_names)

    child = _build_continuation_child(
        _record(
            subagent_type=subagent_type,
            effective_allowed_tool_names=frozenset(retained_names),
        ),
        prompt="continue orchestration",
        parent_agent=parent,
    )

    expected_names = {"read_file"}
    if expected_delegation:
        expected_names.add("delegate_task")
    assert not hasattr(child, "_delegate_role")
    assert child.valid_tool_names == expected_names
    assert child._subagent_tool_policy.allowed_names == frozenset(expected_names)
    assert "delegate_continue" not in child.valid_tool_names


def test_build_continuation_child_rejects_missing_effective_tool_ceiling():
    from tools.delegate_continue_tool import _build_continuation_child

    with pytest.raises(ValueError, match="no original effective tool ceiling"):
        _build_continuation_child(
            _record(effective_allowed_tool_names=frozenset()),
            prompt="must fail closed",
            parent_agent=_parent(),
        )


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
        return _attach_policy(
            SimpleNamespace(valid_tool_names={"read_file"}),
            {"read_file"},
            kwargs["profile"].name,
        )

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
        _subagent_id = "sa-test"
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_estimated_cost_usd = 0.0
        session_reasoning_tokens = 0
        tool_progress_callback = None
        valid_tool_names = {"read_file"}

        def run_conversation(self, **kwargs):
            return {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
                "messages": [
                    {"role": "user", "content": kwargs["user_message"]},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "read-1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"x.py"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "read-1",
                        "content": "source",
                    },
                    {"role": "assistant", "content": "done"},
                ],
            }

        def get_activity_summary(self):
            return {"api_call_count": 1, "current_tool": None, "max_iterations": 1}

        def close(self):
            pass

    monkeypatch.setattr(dt, "_get_retained_session_ttl", lambda: 60)
    monkeypatch.setattr(dt, "_get_max_retained_subagents", lambda: 64)
    monkeypatch.setattr(
        dt.file_state,
        "writes_since",
        lambda *_args, **_kwargs: {"sa-test": ["/tmp/repo/changed.py"]},
    )
    governance = dt.load_governance_snapshot()
    child = _attach_policy(FakeChild(), {"read_file"}, "general-purpose")
    child._governance_diagnostics = {
        "profile_id": governance.profile_id,
        "profile_home": str(governance.profile_home),
        "fingerprint": governance.fingerprint,
        "total_bytes": governance.total_bytes,
    }

    entry = dt._run_single_child(
        task_index=0,
        description="implement",
        child=child,
        parent_agent=_parent(),
        prompt="perform the delegated task",
        child_timeout_override=30,
        retain_session=True,
        subagent_type="general-purpose",
        workspace_path="/tmp/repo",
    )

    assert entry["status"] == "completed"
    assert entry["retention_status"] == "retained"
    assert entry["agent_id"] == "child-session"
    record = get_retained_subagent_session("child-session")
    assert record.status == "completed"
    assert record.updated_at == record.created_at
    assert record.files_written == ("/tmp/repo/changed.py",)
    assert record.tool_trace_metadata[0][0] == "read_file"
    assert record.subagent_type == "general-purpose"
    assert record.workspace_path == "/tmp/repo"
    assert record.effective_allowed_tool_names == frozenset({"read_file"})
    assert record.profile_id == governance.profile_id
    assert record.canonical_profile_home == str(governance.profile_home.resolve())
    assert record.original_policy_identities == (
        child._subagent_tool_policy.authority_snapshot.policy_identities
    )
    assert record.original_governance_fingerprint == governance.fingerprint
    assert record.conversation_history[-1] == {"role": "assistant", "content": "done"}


def test_run_single_child_surfaces_retention_failure(monkeypatch):
    import tools.delegate_tool as dt
    import tools.subagent_sessions as sessions

    governance = dt.load_governance_snapshot()
    child = _attach_policy(
        _CompletedRetainableChild(), {"read_file"}, "general-purpose"
    )
    child._governance_diagnostics = {
        "profile_id": governance.profile_id,
        "profile_home": str(governance.profile_home),
        "fingerprint": governance.fingerprint,
        "total_bytes": governance.total_bytes,
    }
    monkeypatch.setattr(
        sessions,
        "retain_subagent_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("retained transcript capacity reached")
        ),
    )

    entry = dt._run_single_child(
        task_index=0,
        description="implement",
        child=child,
        parent_agent=_parent(),
        prompt="perform the delegated task",
        child_timeout_override=30,
        retain_session=True,
        subagent_type="general-purpose",
        workspace_path="/tmp/repo",
    )

    assert entry["status"] == "completed"
    assert entry["retention_status"] == "failed"
    assert entry["retention_error"] == "retained transcript capacity reached"
    assert "agent_id" not in entry
    assert "retained_until" not in entry


def test_initial_retained_history_projects_handle_only_tool_results(monkeypatch):
    import tools.delegate_tool as dt

    sensitive_body = "A" * 400 + "RAW_BODY_CANARY"

    class SensitiveChild(_CompletedRetainableChild):
        _subagent_tool_result_retention_by_call_id = {
            "sensitive-call": ResultRetention.HANDLE_ONLY
        }

        def run_conversation(self, **kwargs):
            return {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
                "messages": [
                    {"role": "user", "content": kwargs["user_message"]},
                    {
                        "role": "tool",
                        "name": "mcp_notion_ai_notion_ai_ask",
                        "tool_call_id": "sensitive-call",
                        "content": sensitive_body,
                    },
                    {"role": "assistant", "content": "done"},
                ],
            }

    governance = dt.load_governance_snapshot()
    child = _attach_policy(SensitiveChild(), {"read_file"}, "general-purpose")
    child._governance_diagnostics = {
        "profile_id": governance.profile_id,
        "profile_home": str(governance.profile_home),
        "fingerprint": governance.fingerprint,
        "total_bytes": governance.total_bytes,
    }
    dt._run_single_child(
        task_index=0,
        description="read notion",
        child=child,
        parent_agent=_parent(),
        prompt="perform the delegated task",
        child_timeout_override=30,
        retain_session=True,
        subagent_type="general-purpose",
        workspace_path="/tmp/repo",
    )

    stored_history = get_retained_subagent_session(
        "child-session"
    ).conversation_history
    tool_content = next(
        message["content"] for message in stored_history if message["role"] == "tool"
    )
    projection = json.loads(tool_content)
    assert "RAW_BODY_CANARY" not in tool_content
    assert projection["retention"] == "handle_only"
    assert projection["handle"].startswith("sha256:")


def test_continued_retained_history_projects_handle_only_tool_results(monkeypatch):
    from tools.delegate_continue_tool import _run_continuation_entry

    sensitive_body = "B" * 400 + "RAW_CONTINUATION_CANARY"

    class SensitiveContinuationChild:
        model = "model-a"
        provider = "openrouter"
        _subagent_tool_result_retention_by_call_id = {
            "continued-sensitive-call": ResultRetention.HANDLE_ONLY
        }

        def run_conversation(self, **kwargs):
            return {
                "final_response": "continued",
                "api_calls": 1,
                "messages": kwargs["conversation_history"]
                + [
                    {
                        "role": "tool",
                        "name": "mcp_apple_mail_get_message",
                        "tool_call_id": "continued-sensitive-call",
                        "content": sensitive_body,
                    },
                    {"role": "assistant", "content": "continued"},
                ],
            }

        def close(self):
            pass

    retain_subagent_session(_record())
    monkeypatch.setattr(
        "tools.delegate_continue_tool._build_continuation_child",
        lambda *_args, **_kwargs: SensitiveContinuationChild(),
    )
    _run_continuation_entry(
        get_retained_subagent_session("agent-1"),
        "continue reading mail",
        _parent(),
    )

    stored_history = get_retained_subagent_session("agent-1").conversation_history
    tool_content = next(
        message["content"] for message in stored_history if message["role"] == "tool"
    )
    projection = json.loads(tool_content)
    assert "RAW_CONTINUATION_CANARY" not in tool_content
    assert projection["retention"] == "handle_only"
    assert projection["handle"].startswith("sha256:")


def test_run_single_child_does_not_retain_without_parent_session_id(monkeypatch):
    import tools.delegate_tool as dt

    monkeypatch.setattr(dt, "_get_retained_session_ttl", lambda: 60)
    monkeypatch.setattr(dt, "_get_max_retained_subagents", lambda: 64)

    entry = dt._run_single_child(
        task_index=0,
        description="stateless request",
        child=_CompletedRetainableChild(),
        parent_agent=_parent(""),
        prompt="perform the delegated task",
        child_timeout_override=30,
        retain_session=True,
        subagent_type="general-purpose",
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
