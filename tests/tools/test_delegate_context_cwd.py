from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock


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

    def fake_run_single_child(
        task_index, description, child=None, parent_agent=None, **_kw
    ):
        try:
            return {
                "task_index": task_index,
                "status": "completed",
                "summary": summary_fn(),
                "api_calls": 0,
                "duration_seconds": 0,
                "_child_role": "leaf",
                "_child_cost_usd": 0.0,
            }
        finally:
            on_runner_finished = _kw.get("on_runner_finished")
            if on_runner_finished is not None:
                on_runner_finished()

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
            tasks=[{"description": "a", "prompt": "a"}, {"description": "b", "prompt": "b"}],
            parent_agent=parent,
        run_in_background=False)
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
        raw = dt.delegate_task(description="a", prompt="a", run_in_background=True, parent_agent=parent)
        payload = json.loads(raw)
        assert payload["status"] == "dispatched"
        assert captured["entry"]["results"][0]["summary"] == str(ctx_cwd)
    finally:
        clear_session_vars(tokens)
