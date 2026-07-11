from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_context_policy_exposes_explicit_error_type():
    from agent import subagent_context_policy

    assert issubclass(subagent_context_policy.ContextPolicyError, RuntimeError)


@pytest.mark.parametrize(
    "raw_routes",
    [
        {"project_id": "secret-route-value"},
        ("secret-route-entry",),
        ({7: "secret-route-value"},),
        ({"project_id": 7},),
    ],
)
def test_malformed_trusted_routes_fail_before_prompt_construction(raw_routes, tmp_path):
    from agent.subagent_context_policy import (
        ContextPolicyError,
        build_context_policy_capsule,
    )
    from tools.subagent_profiles import get_subagent_profile

    prompt_constructed = False
    with pytest.raises(ContextPolicyError) as raised:
        build_context_policy_capsule(
            profile=get_subagent_profile("general-purpose"),
            goal="untrusted goal",
            context="untrusted context",
            parent_agent=SimpleNamespace(_trusted_project_routes=raw_routes),
            workspace_path=str(tmp_path),
        )
        prompt_constructed = True

    assert prompt_constructed is False
    assert "secret-route" not in str(raised.value)
    assert "untrusted" not in str(raised.value)


def test_absent_trusted_route_seam_is_valid_even_for_dynamic_parent(tmp_path):
    from agent.subagent_context_policy import build_context_policy_capsule
    from tools.subagent_profiles import get_subagent_profile

    capsule = build_context_policy_capsule(
        profile=get_subagent_profile("Plan"),
        goal="untrusted goal",
        context="untrusted context",
        parent_agent=MagicMock(),
        workspace_path=str(tmp_path),
    )

    assert capsule.project_routes == ()
    assert capsule.must_query_project_memory is True


def test_project_facts_failure_is_an_explicit_safe_error(monkeypatch, tmp_path):
    from agent import coding_context
    from agent.subagent_context_policy import (
        ContextPolicyError,
        build_context_policy_capsule,
    )
    from tools.subagent_profiles import get_subagent_profile

    helper_secret = "secret helper failure detail"

    def fail_project_facts(_workspace_path):
        raise OSError(helper_secret)

    monkeypatch.setattr(coding_context, "project_facts_for", fail_project_facts)

    with pytest.raises(ContextPolicyError) as raised:
        build_context_policy_capsule(
            profile=get_subagent_profile("general-purpose"),
            goal="untrusted goal",
            context="untrusted context",
            parent_agent=SimpleNamespace(_trusted_project_routes=()),
            workspace_path=str(tmp_path),
        )

    assert helper_secret not in str(raised.value)
    assert "untrusted" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_delegate_worker_submission_preserves_contextvar_cwd(tmp_path):
    """delegate_task child workers must inherit the parent's session cwd ContextVar."""
    from agent.runtime_cwd import resolve_context_cwd
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.delegate_tool import _submit_with_context

    ctx_cwd = tmp_path / "ctx"
    ctx_cwd.mkdir()

    tokens = set_session_vars(cwd=str(ctx_cwd))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = _submit_with_context(executor, lambda: str(resolve_context_cwd()))
            assert future.result(timeout=5) == str(ctx_cwd)
    finally:
        clear_session_vars(tokens)


def test_run_single_child_registers_and_clears_child_terminal_override(tmp_path):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import delegate_tool as dt
    from tools.terminal_tool import _resolve_container_task_id, resolve_task_overrides

    ctx_cwd = tmp_path / "ctx"
    ctx_cwd.mkdir()
    observed: dict = {}

    class Child:
        _subagent_id = "sa-test"
        _parent_subagent_id = None
        _delegate_depth = 1
        _delegate_role = "leaf"
        _subagent_goal = "goal"
        _delegate_saved_tool_names = []
        _credential_pool = None
        tool_progress_callback = None
        model = "test-model"
        session_id = "child-session"
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_estimated_cost_usd = 0.0

        def get_activity_summary(self):
            return {
                "current_tool": None,
                "api_call_count": 0,
                "max_iterations": 1,
                "last_activity_desc": "idle",
            }

        def run_conversation(self, user_message, task_id=None, stream_callback=None):
            observed["task_id"] = task_id
            observed["container_task_id"] = _resolve_container_task_id(task_id)
            observed["overrides"] = resolve_task_overrides(task_id)
            return {
                "final_response": "ok",
                "completed": True,
                "messages": [],
                "api_calls": 0,
            }

        def close(self):
            observed["closed"] = True

    parent = SimpleNamespace(
        _current_task_id="parent-task",
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _touch_activity=lambda *_a, **_kw: None,
        session_id="parent-session",
        _current_turn_id="turn",
    )

    tokens = set_session_vars(cwd=str(ctx_cwd))
    try:
        result = dt._run_single_child(0, "goal", Child(), parent)
        assert result["status"] == "completed"
        assert observed["task_id"] == "sa-test"
        assert observed["container_task_id"] == "sa-test"
        assert observed["overrides"] == {
            "cwd": str(ctx_cwd),
            "_force_task_isolation": True,
        }
        assert resolve_task_overrides("sa-test") == {}
    finally:
        clear_session_vars(tokens)


def _install_delegate_task_stubs(monkeypatch, dt, summary_fn):
    class Parent:
        _delegate_depth = 0
        _current_task_id = "parent-task"
        _current_turn_id = "turn"
        _memory_manager = None
        _active_children = []
        _active_children_lock = threading.Lock()
        _delegate_spinner = None
        session_id = "parent-session"
        session_estimated_cost_usd = 0.0
        session_cost_source = "none"
        session_cost_status = "unknown"
        model = "test-model"
        provider = "test-provider"
        api_mode = "chat_completions"
        base_url = "http://test.local"
        enabled_toolsets = []
        valid_tool_names = []

        def _touch_activity(self, *_a, **_kw):
            pass

    monkeypatch.setattr(dt, "_load_config", lambda: {"max_iterations": 1})
    monkeypatch.setattr(dt, "_get_max_concurrent_children", lambda: 4)
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 2)
    monkeypatch.setattr(dt, "_get_orchestrator_enabled", lambda: False)
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda cfg, parent: {
            "model": "test-model",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )

    def fake_build_child_agent(task_index, **_kw):
        return SimpleNamespace(
            _delegate_role="leaf",
            session_id=f"child-{task_index}",
            _delegate_saved_tool_names=[],
        )

    monkeypatch.setattr(dt, "_build_child_agent", fake_build_child_agent)

    def fake_run_single_child(task_index, goal, child=None, parent_agent=None, **_kw):
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": summary_fn(),
            "api_calls": 0,
            "duration_seconds": 0,
            "_child_role": "leaf",
            "_child_cost_usd": 0.0,
        }

    monkeypatch.setattr(dt, "_run_single_child", fake_run_single_child)
    return Parent()


def test_delegate_task_batch_preserves_contextvar_cwd(tmp_path, monkeypatch):
    from agent.runtime_cwd import resolve_context_cwd
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import delegate_tool as dt

    env_cwd = tmp_path / "env"
    ctx_cwd = tmp_path / "ctx"
    env_cwd.mkdir()
    ctx_cwd.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(env_cwd))

    parent = _install_delegate_task_stubs(
        monkeypatch,
        dt,
        summary_fn=lambda: str(resolve_context_cwd()),
    )

    tokens = set_session_vars(cwd=str(ctx_cwd))
    try:
        raw = dt.delegate_task(
            tasks=[{"goal": "a"}, {"goal": "b"}],
            parent_agent=parent,
        )
        payload = json.loads(raw)
        assert [r["summary"] for r in payload["results"]] == [
            str(ctx_cwd),
            str(ctx_cwd),
        ]
    finally:
        clear_session_vars(tokens)


def test_delegate_task_background_runner_preserves_contextvar_cwd(tmp_path, monkeypatch):
    from agent.runtime_cwd import resolve_context_cwd
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import async_delegation
    from tools import delegate_tool as dt

    env_cwd = tmp_path / "env"
    ctx_cwd = tmp_path / "ctx"
    env_cwd.mkdir()
    ctx_cwd.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(env_cwd))

    captured: dict = {}
    parent = _install_delegate_task_stubs(
        monkeypatch,
        dt,
        summary_fn=lambda: str(resolve_context_cwd()),
    )

    def fake_dispatch_async_delegation_batch(**kwargs):
        runner = kwargs["runner"]
        with ThreadPoolExecutor(max_workers=1) as executor:
            captured["entry"] = executor.submit(runner).result(timeout=5)
        return {"status": "dispatched", "delegation_id": "d-test"}

    monkeypatch.setattr(
        async_delegation,
        "dispatch_async_delegation_batch",
        fake_dispatch_async_delegation_batch,
    )

    tokens = set_session_vars(cwd=str(ctx_cwd))
    try:
        raw = dt.delegate_task(goal="a", background=True, parent_agent=parent)
        payload = json.loads(raw)
        assert payload["status"] == "dispatched"
        assert captured["entry"]["results"][0]["summary"] == str(ctx_cwd)
    finally:
        clear_session_vars(tokens)


def test_context_policy_capsules_consume_only_trusted_routes_and_runtime_metadata(tmp_path):
    from agent.subagent_context_policy import build_context_policy_capsule
    from tools.subagent_profiles import get_subagent_profile

    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    parent_secret = "parent secret tool output"
    trusted_routes = ({"project_id": "project-1", "hub_id": "hub-1"},)
    parent = SimpleNamespace(
        _trusted_project_routes=trusted_routes,
        messages=[{"role": "tool", "content": parent_secret}],
        last_tool_output=parent_secret,
    )

    explore_capsule = build_context_policy_capsule(
        profile=get_subagent_profile("Explore"),
        goal="project-1 from untrusted goal",
        context=parent_secret,
        parent_agent=parent,
        workspace_path=str(tmp_path),
    )
    plan_capsule = build_context_policy_capsule(
        profile=get_subagent_profile("Plan"),
        goal="project-1 from untrusted goal",
        context=parent_secret,
        parent_agent=SimpleNamespace(_trusted_project_routes=()),
        workspace_path=str(tmp_path),
    )
    gp_capsule = build_context_policy_capsule(
        profile=get_subagent_profile("general-purpose"),
        goal="project-1 from untrusted goal",
        context=parent_secret,
        parent_agent=parent,
        workspace_path=str(tmp_path),
    )

    assert explore_capsule.policy == "lean"
    assert explore_capsule.project_routes == ()
    assert explore_capsule.workspace_metadata == {}
    assert plan_capsule.policy == "project_summary"
    assert plan_capsule.must_query_project_memory is True
    assert plan_capsule.project_routes == ()
    assert gp_capsule.policy == "normal"
    assert gp_capsule.project_routes == trusted_routes
    assert gp_capsule.must_query_project_memory is False
    assert gp_capsule.workspace_metadata["workspace_path"] == str(tmp_path.resolve())
    assert gp_capsule.workspace_metadata["repo_root"] == str(tmp_path.resolve())
    serialized = json.dumps(asdict(gp_capsule), ensure_ascii=False)
    assert "workspace_metadata" in serialized
    assert "project_id" in serialized
    assert parent_secret not in serialized

    from tools.delegate_tool import _build_child_system_prompt

    plan_prompt = _build_child_system_prompt(
        profile=get_subagent_profile("Plan"),
        role="leaf",
        workspace_path=str(tmp_path),
        child_depth=1,
        max_spawn_depth=1,
        context_policy_capsule=plan_capsule,
    )
    gp_prompt = _build_child_system_prompt(
        profile=get_subagent_profile("general-purpose"),
        role="leaf",
        workspace_path=str(tmp_path),
        child_depth=1,
        max_spawn_depth=1,
        context_policy_capsule=gp_capsule,
    )
    assert "project_routes" in plan_prompt
    assert "workspace_metadata" in gp_prompt
    assert parent_secret not in gp_prompt


def test_context_policy_capsule_maps_are_immutable_and_detached(tmp_path):
    from agent.subagent_context_policy import build_context_policy_capsule
    from tools.delegate_tool import _build_child_system_prompt
    from tools.subagent_profiles import get_subagent_profile

    (tmp_path / ".git").mkdir()
    parent_routes = [{"project_id": "project-1", "hub_id": "hub-1"}]
    capsule = build_context_policy_capsule(
        profile=get_subagent_profile("general-purpose"),
        goal="untrusted goal",
        context="untrusted context",
        parent_agent=SimpleNamespace(_trusted_project_routes=parent_routes),
        workspace_path=str(tmp_path),
    )

    def serialize_capsule():
        return json.dumps(asdict(capsule), ensure_ascii=False, sort_keys=True)

    def build_prompt():
        return _build_child_system_prompt(
            profile=get_subagent_profile("general-purpose"),
            role="leaf",
            workspace_path=str(tmp_path),
            child_depth=1,
            max_spawn_depth=1,
            context_policy_capsule=capsule,
        )

    serialized_before = serialize_capsule()
    prompt_before = build_prompt()
    parent_routes[0]["project_id"] = "mutated-parent-route"
    assert serialize_capsule() == serialized_before

    with pytest.raises(TypeError):
        capsule.project_routes[0]["project_id"] = "mutated-capsule-route"
    with pytest.raises(TypeError):
        capsule.workspace_metadata["repo_root"] = "mutated-metadata"

    assert serialize_capsule() == serialized_before
    assert build_prompt() == prompt_before


def test_delegate_batch_reuses_one_snapshot_then_new_call_reloads(monkeypatch):
    from tools import delegate_tool as dt

    snapshots = [object(), object()]
    load_count = 0
    built_snapshots = []

    def fake_load_governance_snapshot():
        nonlocal load_count
        snapshot = snapshots[load_count]
        load_count += 1
        return snapshot

    parent = _install_delegate_task_stubs(monkeypatch, dt, summary_fn=lambda: "ok")
    original_fake_builder = dt._build_child_agent

    def capture_builder(*args, **kwargs):
        built_snapshots.append(kwargs["governance_snapshot"])
        return original_fake_builder(*args, **kwargs)

    monkeypatch.setattr(dt, "load_governance_snapshot", fake_load_governance_snapshot)
    monkeypatch.setattr(dt, "_build_child_agent", capture_builder)

    dt.delegate_task(
        tasks=[{"goal": "a"}, {"goal": "b"}],
        scheduling="foreground",
        parent_agent=parent,
    )
    dt.delegate_task(goal="later", scheduling="foreground", parent_agent=parent)

    assert load_count == 2
    assert built_snapshots[:2] == [snapshots[0], snapshots[0]]
    assert built_snapshots[2] is snapshots[1]
