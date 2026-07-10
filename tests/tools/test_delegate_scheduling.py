"""Task 4 scheduling and foreground/background delivery race tests."""

import json
import threading
import time

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry
from tools.async_delegation import (
    dispatch_async_delegation_batch,
    wait_for_async_delegation,
)
from tools.delegate_tool import _resolve_scheduling


@pytest.fixture(autouse=True)
def _clean_async_registry():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _dispatch(runner, injected, *, mode="foreground_waiting"):
    return dispatch_async_delegation_batch(
        goals=["test goal"],
        context="context",
        toolsets=None,
        role="leaf",
        model="test-model",
        session_key="session",
        runner=runner,
        max_async_children=3,
        initial_delivery_mode=mode,
        inject_fn=lambda event: injected.append(event),
    )


def _record(handle):
    with ad._records_lock:
        return ad._records[handle["delegation_id"]]


def _parent(depth=0):
    from unittest.mock import MagicMock

    parent = MagicMock()
    parent._delegate_depth = depth
    parent.session_id = "parent-session"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._memory_manager = None
    return parent


def _install_fake_delegate_runtime(monkeypatch, run_child):
    from unittest.mock import MagicMock
    import tools.delegate_tool as dt

    built = []

    def build_child(**kwargs):
        child = MagicMock()
        child._delegate_role = kwargs.get("role", "leaf")
        child._subagent_id = f"fake-{len(built)}"
        parent = kwargs.get("parent_agent")
        if parent is not None:
            parent._active_children.append(child)
        built.append(child)
        return child

    monkeypatch.setattr(dt, "_build_child_agent", build_child)
    monkeypatch.setattr(dt, "_run_single_child", run_child)
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda *_args, **_kwargs: {
            "model": "test-model",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": [],
        },
    )
    return built


def test_auto_preserves_legacy_background_default():
    assert _resolve_scheduling(None, "auto", is_batch=False, is_subagent=False) == "background"


def test_auto_uses_foreground_for_single_explore_and_plan():
    assert _resolve_scheduling("Explore", "auto", False, False) == "foreground"
    assert _resolve_scheduling("Plan", "auto", False, False) == "foreground"


def test_auto_uses_background_for_general_purpose_and_batches():
    assert _resolve_scheduling("general-purpose", "auto", False, False) == "background"
    assert _resolve_scheduling("Explore", "auto", True, False) == "background"


def test_generic_foreground_timeouts_use_global_then_general_purpose_defaults():
    from tools.delegate_tool import _resolve_foreground_timeouts

    assert _resolve_foreground_timeouts(None, {}) == (1800, 7200)
    assert _resolve_foreground_timeouts(
        None,
        {
            "foreground_wait_timeout_seconds": 222,
            "child_run_timeout_seconds": 333,
        },
    ) == (222, 333)


def test_nested_auto_and_foreground_stay_foreground_but_background_fails():
    assert _resolve_scheduling("general-purpose", "auto", False, True) == "foreground"
    assert _resolve_scheduling("Explore", "foreground", False, True) == "foreground"
    with pytest.raises(ValueError, match="Nested/orchestrator delegation must run foreground"):
        _resolve_scheduling("Explore", "background", False, True)


def test_completion_before_wait_is_claimed_without_async_injection():
    injected = []
    handle = _dispatch(
        lambda: {
            "results": [{"status": "completed", "summary": "done"}],
            "total_duration_seconds": 0.01,
        },
        injected,
    )
    record = _record(handle)
    assert record["done_event"].wait(timeout=2)

    payload = wait_for_async_delegation(handle, timeout_seconds=2)

    assert json.loads(payload)["results"][0]["summary"] == "done"
    assert injected == []
    assert wait_for_async_delegation(handle, timeout_seconds=0) is None


def test_timeout_before_completion_hands_same_future_to_background_once():
    injected = []
    gate = threading.Event()
    handle = _dispatch(
        lambda: (
            gate.wait(2),
            {
                "results": [{"status": "completed", "summary": "late"}],
                "total_duration_seconds": 0.02,
            },
        )[1],
        injected,
    )
    record = _record(handle)
    future = record["future"]

    assert wait_for_async_delegation(handle["delegation_id"], timeout_seconds=0.01) is None
    gate.set()
    future.result(timeout=2)

    assert len(injected) == 1
    assert injected[0]["delegation_id"] == handle["delegation_id"]
    assert injected[0]["results"][0]["summary"] == "late"
    assert wait_for_async_delegation(handle, timeout_seconds=0) is None


def test_concurrent_completion_and_timeout_has_exactly_one_delivery():
    for _ in range(25):
        injected = []
        gate = threading.Event()
        start = threading.Barrier(2)
        handle = _dispatch(
            lambda: (
                start.wait(timeout=2),
                gate.wait(2),
                {
                    "results": [{"status": "completed", "summary": "raced"}],
                    "total_duration_seconds": 0.01,
                },
            )[2],
            injected,
        )
        record = _record(handle)
        future = record["future"]
        claimed = []

        def waiter():
            start.wait(timeout=2)
            claimed.append(wait_for_async_delegation(handle, timeout_seconds=0))

        thread = threading.Thread(target=waiter)
        thread.start()
        gate.set()
        thread.join(timeout=2)
        future.result(timeout=2)

        foreground_deliveries = sum(payload is not None for payload in claimed)
        assert foreground_deliveries + len(injected) == 1
        assert wait_for_async_delegation(handle, timeout_seconds=0) is None


def test_parent_interrupt_claims_foreground_and_suppresses_late_delivery():
    injected = []
    allow_finish = threading.Event()
    interrupt_called = threading.Event()
    parent_interrupt = threading.Event()
    waiter_started = threading.Event()
    waiter_result = []

    handle = dispatch_async_delegation_batch(
        goals=["interruptible foreground"],
        context=None,
        toolsets=None,
        role="leaf",
        model="test-model",
        session_key="session",
        runner=lambda: (
            allow_finish.wait(2),
            {
                "results": [{"status": "completed", "summary": "too late"}],
                "total_duration_seconds": 0.01,
            },
        )[1],
        interrupt_fn=interrupt_called.set,
        max_async_children=3,
        initial_delivery_mode="foreground_waiting",
        inject_fn=lambda event: injected.append(event),
    )
    record = _record(handle)

    def wait_in_foreground():
        waiter_started.set()
        waiter_result.append(
            wait_for_async_delegation(
                handle,
                timeout_seconds=60,
                interrupt_requested=parent_interrupt.is_set,
            )
        )

    thread = threading.Thread(target=wait_in_foreground)
    thread.start()
    assert waiter_started.wait(1)
    parent_interrupt.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    interrupted_payload = json.loads(waiter_result[0])
    assert interrupted_payload == {
        "status": "interrupted",
        "mode": "foreground",
        "delegation_id": handle["delegation_id"],
        "error": "Foreground delegation interrupted by parent.",
        "note": "Late completion delivery is suppressed for this delegation.",
    }
    assert interrupt_called.is_set()

    allow_finish.set()
    record["future"].result(timeout=2)
    assert injected == []
    assert process_registry.completion_queue.empty()
    assert wait_for_async_delegation(handle, timeout_seconds=0) is None


def test_delegate_task_foreground_parent_interrupt_returns_inline_without_late_queue(
    monkeypatch
):
    import tools.delegate_tool as dt

    run_started = threading.Event()
    allow_finish = threading.Event()
    parent = _parent()
    response = []

    def run_child(task_index, goal, **_kwargs):
        run_started.set()
        assert allow_finish.wait(2)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": f"late: {goal}",
        }

    built = _install_fake_delegate_runtime(monkeypatch, run_child)
    monkeypatch.setattr(
        dt,
        "_load_config",
        lambda: {
            "foreground_wait_timeout_seconds": 60,
            "child_run_timeout_seconds": 60,
        },
    )

    thread = threading.Thread(
        target=lambda: response.append(
            json.loads(
                dt.delegate_task(
                    goal="interrupt me",
                    subagent_type="Explore",
                    scheduling="foreground",
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
    assert built[0].interrupt.call_count == 1
    assert process_registry.completion_queue.empty()

    with ad._records_lock:
        future = next(iter(ad._records.values()))["future"]
    allow_finish.set()
    future.result(timeout=2)
    assert process_registry.completion_queue.empty()


def test_completed_foreground_result_wins_over_already_set_parent_interrupt():
    injected = []
    interrupt_called = threading.Event()
    parent_interrupt = threading.Event()
    handle = dispatch_async_delegation_batch(
        goals=["already complete"],
        context=None,
        toolsets=None,
        role="leaf",
        model="test-model",
        session_key="session",
        runner=lambda: {
            "results": [{"status": "completed", "summary": "won race"}],
            "total_duration_seconds": 0.01,
        },
        interrupt_fn=interrupt_called.set,
        max_async_children=3,
        initial_delivery_mode="foreground_waiting",
        inject_fn=lambda event: injected.append(event),
    )
    assert _record(handle)["done_event"].wait(2)
    parent_interrupt.set()

    payload = wait_for_async_delegation(
        handle,
        timeout_seconds=60,
        interrupt_requested=parent_interrupt.is_set,
    )

    assert json.loads(payload)["results"][0]["summary"] == "won race"
    assert not interrupt_called.is_set()
    assert injected == []


def test_unclaimed_foreground_completion_is_not_pruned_before_wait(monkeypatch):
    monkeypatch.setattr(ad, "_MAX_RETAINED_COMPLETED", 2)
    foreground_injected = []
    foreground = _dispatch(
        lambda: {
            "results": [{"status": "completed", "summary": "must survive"}],
            "total_duration_seconds": 0.01,
        },
        foreground_injected,
    )
    assert _record(foreground)["done_event"].wait(timeout=2)

    background_futures = []
    for i in range(3):
        handle = dispatch_async_delegation_batch(
            goals=[f"background {i}"],
            context=None,
            toolsets=None,
            role="leaf",
            model="test-model",
            session_key="session",
            runner=lambda: {"results": [], "total_duration_seconds": 0.0},
            max_async_children=5,
            inject_fn=lambda _event: None,
        )
        background_futures.append(_record(handle)["future"])
    for future in background_futures:
        future.result(timeout=2)

    payload = wait_for_async_delegation(foreground, timeout_seconds=0)
    assert payload is not None
    assert json.loads(payload)["results"][0]["summary"] == "must survive"
    assert foreground_injected == []


def test_list_async_delegations_excludes_runtime_sync_fields():
    injected = []
    gate = threading.Event()
    handle = _dispatch(
        lambda: (gate.wait(2), {"results": [], "total_duration_seconds": 0.0})[1],
        injected,
    )

    listed = next(
        item for item in ad.list_async_delegations()
        if item["delegation_id"] == handle["delegation_id"]
    )
    for internal in ("future", "done_event", "delivery_lock", "inject_fn", "interrupt_fn"):
        assert internal not in listed
    json.dumps(listed)
    gate.set()


def test_run_agent_model_dispatch_uses_scheduling_resolver_not_forced_background():
    from unittest.mock import patch
    import run_agent

    captured = {}

    def fake_delegate(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", fake_delegate):
        run_agent.AIAgent._dispatch_delegate_task(
            _parent(),
            {"goal": "inspect", "subagent_type": "Explore", "scheduling": "auto"},
        )

    assert captured["scheduling"] == "auto"
    assert captured["_dispatch_origin"] == "model"
    assert "background" not in captured


def test_direct_python_legacy_auto_remains_synchronous(monkeypatch):
    import tools.delegate_tool as dt

    _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, goal, **kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": f"sync: {goal}",
        },
    )

    result = json.loads(dt.delegate_task(goal="legacy", parent_agent=_parent()))

    assert result["results"][0]["summary"] == "sync: legacy"
    assert "delegation_id" not in result
    assert process_registry.completion_queue.empty()


def test_explicit_foreground_quick_completion_returns_inline_with_profile_run_cap(monkeypatch):
    import tools.delegate_tool as dt

    seen = []

    def run_child(task_index, goal, **kwargs):
        seen.append(kwargs.get("child_timeout_override"))
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "quick",
        }

    _install_fake_delegate_runtime(monkeypatch, run_child)
    result = json.loads(
        dt.delegate_task(
            goal="inspect",
            subagent_type="Explore",
            scheduling="foreground",
            parent_agent=_parent(),
        )
    )

    assert result["results"][0]["summary"] == "quick"
    assert seen == [1800]
    assert process_registry.completion_queue.empty()


def test_foreground_wait_timeout_backgrounds_same_delegation_then_delivers_once(monkeypatch):
    import tools.delegate_tool as dt

    gate = threading.Event()

    def run_child(task_index, goal, **kwargs):
        gate.wait(timeout=2)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "late result",
        }

    _install_fake_delegate_runtime(monkeypatch, run_child)
    monkeypatch.setattr(
        dt,
        "_load_config",
        lambda: {
            "foreground_wait_timeout_seconds": 0,
            "child_run_timeout_seconds": 321,
        },
    )

    timed_out = json.loads(
        dt.delegate_task(
            goal="slow inspect",
            subagent_type="Explore",
            scheduling="foreground",
            parent_agent=_parent(),
        )
    )
    delegation_id = timed_out["delegation_id"]

    assert timed_out["status"] == "backgrounded_after_foreground_timeout"
    assert timed_out["mode"] == "background"
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()

    with ad._records_lock:
        future = ad._records[delegation_id]["future"]
    gate.set()
    future.result(timeout=2)
    event = process_registry.completion_queue.get(timeout=2)

    assert event["delegation_id"] == delegation_id
    assert event["results"][0]["summary"] == "late result"
    assert process_registry.completion_queue.empty()


def test_nested_explicit_background_fails_before_child_construction(monkeypatch):
    import tools.delegate_tool as dt

    built = _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, goal, **kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": "unexpected",
        },
    )
    result = json.loads(
        dt.delegate_task(
            goal="worker",
            scheduling="background",
            parent_agent=_parent(depth=1),
        )
    )

    assert "Nested/orchestrator delegation must run foreground" in result["error"]
    assert built == []
    assert ad.active_count() == 0


def test_nested_per_task_background_fails_before_depth_or_child_checks(monkeypatch):
    import tools.delegate_tool as dt

    built = _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, goal, **kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": "unexpected",
        },
    )
    result = json.loads(
        dt.delegate_task(
            tasks=[{"goal": "worker", "scheduling": "background"}],
            parent_agent=_parent(depth=1),
            _dispatch_origin="model",
        )
    )

    assert result["error"] == "Nested/orchestrator delegation must run foreground"
    assert built == []


def test_nested_auto_runs_synchronously_without_async_delivery(monkeypatch):
    import tools.delegate_tool as dt

    _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, goal, **kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": "nested inline",
        },
    )
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 2)

    result = json.loads(
        dt.delegate_task(
            goal="worker",
            scheduling="auto",
            parent_agent=_parent(depth=1),
            _dispatch_origin="model",
        )
    )

    assert result["results"][0]["summary"] == "nested inline"
    assert "delegation_id" not in result
    assert process_registry.completion_queue.empty()
    assert ad.active_count() == 0


def test_nested_foreground_uses_profile_child_run_cap(monkeypatch):
    import tools.delegate_tool as dt

    seen = []

    def run_child(task_index, goal, **kwargs):
        seen.append(kwargs.get("child_timeout_override"))
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "nested capped",
        }

    _install_fake_delegate_runtime(monkeypatch, run_child)
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 2)

    result = json.loads(
        dt.delegate_task(
            goal="inspect",
            subagent_type="Explore",
            scheduling="auto",
            parent_agent=_parent(depth=1),
            _dispatch_origin="model",
        )
    )

    assert result["results"][0]["summary"] == "nested capped"
    assert seen == [1800]
    assert "delegation_id" not in result


def test_legacy_model_auto_dispatches_background(monkeypatch):
    import tools.delegate_tool as dt

    gate = threading.Event()

    def run_child(task_index, goal, **kwargs):
        assert kwargs["child_timeout_override"] is None
        gate.wait(timeout=2)
        return {"task_index": task_index, "status": "completed", "summary": "model bg"}

    _install_fake_delegate_runtime(monkeypatch, run_child)
    dispatched = json.loads(
        dt.delegate_task(
            goal="legacy model call",
            parent_agent=_parent(),
            _dispatch_origin="model",
        )
    )

    assert dispatched["status"] == "dispatched"
    assert dispatched["mode"] == "background"
    gate.set()
    event = process_registry.completion_queue.get(timeout=2)
    assert event["delegation_id"] == dispatched["delegation_id"]
    assert event["results"][0]["summary"] == "model bg"


def test_model_auto_background_falls_back_inline_when_async_delivery_is_unsupported(
    monkeypatch,
):
    import gateway.session_context as session_context
    import tools.delegate_tool as dt

    _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, goal, **kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": "inline safe",
        },
    )
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: False)

    result = json.loads(
        dt.delegate_task(
            goal="legacy model call",
            parent_agent=_parent(),
            _dispatch_origin="model",
        )
    )

    assert result["results"][0]["summary"] == "inline safe"
    assert "stateless HTTP API" in result["note"]
    assert "delegation_id" not in result
    assert ad.active_count() == 0


def test_foreground_on_stateless_endpoint_returns_completed_result_not_handle(monkeypatch):
    import gateway.session_context as session_context
    import tools.delegate_tool as dt

    seen = []

    def run_child(task_index, goal, **kwargs):
        seen.append(kwargs["child_timeout_override"])
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "foreground inline safe",
        }

    _install_fake_delegate_runtime(monkeypatch, run_child)
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: False)
    monkeypatch.setattr(
        dt,
        "_load_config",
        lambda: {"foreground_wait_timeout_seconds": 0},
    )

    result = json.loads(
        dt.delegate_task(
            goal="inspect",
            subagent_type="Explore",
            scheduling="foreground",
            parent_agent=_parent(),
            _dispatch_origin="model",
        )
    )

    assert result["results"][0]["summary"] == "foreground inline safe"
    assert "delegation_id" not in result
    assert seen == [1800]
    assert ad.active_count() == 0


def test_async_pool_rejection_runs_already_built_foreground_child_inline(monkeypatch):
    import tools.delegate_tool as dt

    attached_during_inline_run = []

    def run_child(task_index, goal, **kwargs):
        attached_during_inline_run.append(
            kwargs["child"] in kwargs["parent_agent"]._active_children
        )
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "capacity fallback",
        }

    built = _install_fake_delegate_runtime(monkeypatch, run_child)
    monkeypatch.setattr(
        ad,
        "dispatch_async_delegation_batch",
        lambda **kwargs: {"status": "rejected", "error": "capacity reached"},
    )

    parent = _parent()
    result = json.loads(
        dt.delegate_task(
            goal="inspect",
            subagent_type="Explore",
            scheduling="foreground",
            parent_agent=parent,
        )
    )

    assert len(built) == 1
    assert attached_during_inline_run == [True]
    assert result["results"][0]["summary"] == "capacity fallback"
    assert "pool was at capacity" in result["note"]


def test_pure_background_keeps_historical_no_blanket_run_cap(monkeypatch):
    import tools.delegate_tool as dt

    gate = threading.Event()
    seen = []

    def run_child(task_index, goal, **kwargs):
        seen.append(("child_timeout_override" in kwargs, kwargs.get("child_timeout_override")))
        gate.wait(timeout=2)
        return {"task_index": task_index, "status": "completed", "summary": "done"}

    _install_fake_delegate_runtime(monkeypatch, run_child)
    dispatched = json.loads(
        dt.delegate_task(goal="legacy bg", background=True, parent_agent=_parent())
    )
    assert dispatched["status"] == "dispatched"
    gate.set()
    event = process_registry.completion_queue.get(timeout=2)

    assert event["results"][0]["summary"] == "done"
    assert seen == [(True, None)]


def test_mixed_explicit_foreground_batch_uses_each_profile_run_cap(monkeypatch):
    import tools.delegate_tool as dt

    seen = {}

    def run_child(task_index, goal, **kwargs):
        seen[goal] = kwargs.get("child_timeout_override")
        return {"task_index": task_index, "status": "completed", "summary": goal}

    _install_fake_delegate_runtime(monkeypatch, run_child)
    result = json.loads(
        dt.delegate_task(
            tasks=[
                {
                    "goal": "explore",
                    "subagent_type": "Explore",
                    "scheduling": "foreground",
                },
                {
                    "goal": "plan",
                    "subagent_type": "Plan",
                    "scheduling": "foreground",
                },
            ],
            scheduling="foreground",
            parent_agent=_parent(),
        )
    )

    assert [item["summary"] for item in result["results"]] == ["explore", "plan"]
    assert seen == {"explore": 1800, "plan": 3600}
    assert process_registry.completion_queue.empty()


def test_timeout_fields_are_not_model_facing():
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA

    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    task_props = props["tasks"]["items"]["properties"]
    for field in (
        "foreground_wait_timeout_seconds",
        "child_run_timeout_seconds",
        "max_foreground_wait_timeout_seconds",
    ):
        assert field not in props
        assert field not in task_props


def test_model_schema_truthfully_describes_scheduling_and_batch_delivery():
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA, _build_top_level_description

    description = _build_top_level_description()
    assert "BOTH MODES RUN IN THE BACKGROUND" not in description
    assert "scheduling='foreground'" in description
    assert "scheduling='background'" in description
    assert "one batch handle" in description
    assert "one consolidated completion" in description

    background_description = DELEGATE_TASK_SCHEMA["parameters"]["properties"][
        "background"
    ]["description"]
    assert "Use 'scheduling' instead" in background_description
    assert "always run" not in background_description
