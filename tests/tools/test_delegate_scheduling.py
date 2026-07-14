"""Task 4 scheduling and foreground/background delivery race tests."""

import json
import threading
import time

import pytest

from tools import async_delegation as ad
from tools import delegation_capacity as dc
from tools.process_registry import process_registry
from tools.async_delegation import (
    dispatch_async_delegation_batch,
    wait_for_async_delegation,
)
from tools.delegate_tool import _resolve_run_in_background


@pytest.fixture(autouse=True)
def _clean_async_registry():
    ad._reset_for_tests()
    dc._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    ad._reset_for_tests()
    dc._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _dispatch(runner, injected, *, mode="foreground_waiting"):
    return dispatch_async_delegation_batch(
        goals=["test goal"],
        context="context",
        toolsets=None,
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
        child._subagent_profile = kwargs.get("profile")
        child._subagent_id = f"fake-{len(built)}"
        parent = kwargs.get("parent_agent")
        if parent is not None:
            parent._active_children.append(child)
        built.append(child)
        return child

    def run_child_with_capacity(*args, **kwargs):
        on_runner_finished = kwargs.get("on_runner_finished")
        try:
            return run_child(*args, **kwargs)
        finally:
            if on_runner_finished is not None:
                on_runner_finished()

    monkeypatch.setattr(dt, "_build_child_agent", build_child)
    monkeypatch.setattr(dt, "_run_single_child", run_child_with_capacity)
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


def test_top_level_omission_defaults_background():
    assert _resolve_run_in_background(None, is_subagent=False) is True
    assert (
        _resolve_run_in_background(
            None, is_subagent=False, subagent_type="Reviewer"
        )
        is False
    )


def test_top_level_explicit_background_boolean_is_respected():
    assert _resolve_run_in_background(True, is_subagent=False) is True
    assert _resolve_run_in_background(False, is_subagent=False) is False


def test_nested_omission_and_false_are_foreground_but_true_fails():
    assert _resolve_run_in_background(None, is_subagent=True) is False
    assert _resolve_run_in_background(False, is_subagent=True) is False
    with pytest.raises(ValueError, match="Nested delegation cannot run in the background"):
        _resolve_run_in_background(True, is_subagent=True)


def test_non_boolean_background_value_fails_closed():
    with pytest.raises(ValueError, match="must be a boolean"):
        _resolve_run_in_background("background", is_subagent=False)


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

    def run_child(task_index, description, **_kwargs):
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
                    description="interrupt me", prompt="interrupt me",
                    subagent_type="Explore",
                    run_in_background=False,
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


def test_run_agent_model_dispatch_forwards_static_background_boolean():
    from unittest.mock import patch
    import run_agent

    captured = {}

    def fake_delegate(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", fake_delegate):
        run_agent.AIAgent._dispatch_delegate_task(
            _parent(),
            {
                "description": "inspect code",
                "prompt": "inspect the implementation",
                "subagent_type": "Reviewer",
                "review_root": "/tmp/review-repo",
                "run_in_background": False,
            },
        )

    assert captured["description"] == "inspect code"
    assert captured["prompt"] == "inspect the implementation"
    assert captured["review_root"] == "/tmp/review-repo"
    assert captured["run_in_background"] is False
    assert "scheduling" not in captured
    assert "_dispatch_origin" not in captured


def test_runner_capacity_rejects_second_batch_without_partial_start(monkeypatch):
    import tools.delegate_tool as dt

    active = 0
    peak = 0
    lock = threading.Lock()
    first_batch_started = threading.Event()
    release = threading.Event()

    def run_child(task_index, description, **_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 3:
                first_batch_started.set()
        try:
            assert release.wait(3)
            return {
                "task_index": task_index,
                "status": "completed",
                "summary": description,
            }
        finally:
            with lock:
                active -= 1

    built = _install_fake_delegate_runtime(monkeypatch, run_child)
    monkeypatch.setattr(
        dt,
        "_load_config",
        lambda: {
            "max_concurrent_children": 3,
            "foreground_wait_timeout_seconds": 60,
            "child_run_timeout_seconds": 60,
        },
    )
    tasks = [
        {"description": f"task-{index}", "prompt": f"task-{index}"}
        for index in range(3)
    ]

    first = json.loads(
        dt.delegate_task(tasks=tasks, run_in_background=True, parent_agent=_parent())
    )
    assert first["status"] == "dispatched"
    assert first_batch_started.wait(1)

    second = json.loads(
        dt.delegate_task(tasks=tasks, run_in_background=True, parent_agent=_parent())
    )
    try:
        assert second["status"] == "rejected"
        assert "capacity reached" in second["error"].lower()
        assert len(built) == 3
        assert peak <= 3
    finally:
        release.set()
        with ad._records_lock:
            futures = [
                record.get("future")
                for record in ad._records.values()
                if record.get("future") is not None
            ]
        for future in futures:
            future.result(timeout=3)


def test_runner_capacity_combines_global_and_root_session_limits(monkeypatch):
    import tools.delegate_tool as dt

    active = 0
    lock = threading.Lock()
    first_session_full = threading.Event()
    global_full = threading.Event()
    release = threading.Event()

    def run_child(task_index, description, **_kwargs):
        nonlocal active
        with lock:
            active += 1
            if active >= 2:
                first_session_full.set()
            if active >= 4:
                global_full.set()
        try:
            assert release.wait(3)
            return {
                "task_index": task_index,
                "status": "completed",
                "summary": description,
            }
        finally:
            with lock:
                active -= 1

    built = _install_fake_delegate_runtime(monkeypatch, run_child)
    monkeypatch.setattr(
        dt,
        "_load_config",
        lambda: {
            "max_concurrent_children": 2,
            "max_global_concurrent_children": 4,
            "foreground_wait_timeout_seconds": 60,
            "child_run_timeout_seconds": 60,
        },
    )
    two_tasks = [
        {"description": f"task-{index}", "prompt": f"task-{index}"}
        for index in range(2)
    ]
    parent_a = _parent()
    parent_a.session_id = "root-a"
    parent_b = _parent()
    parent_b.session_id = "root-b"
    parent_c = _parent()
    parent_c.session_id = "root-c"

    first = json.loads(
        dt.delegate_task(
            tasks=two_tasks,
            run_in_background=True,
            parent_agent=parent_a,
        )
    )
    assert first["status"] == "dispatched"
    assert first_session_full.wait(1)

    nested_parent_a = _parent(depth=1)
    nested_parent_a.session_id = "child-session-a"
    nested_parent_a._delegate_root_session_id = "root-a"
    same_session = json.loads(
        dt.delegate_task(
            description="overflow-a",
            prompt="overflow-a",
            run_in_background=False,
            parent_agent=nested_parent_a,
        )
    )
    assert same_session["status"] == "rejected"
    assert "per-session" in same_session["error"]
    assert len(built) == 2

    second = json.loads(
        dt.delegate_task(
            tasks=two_tasks,
            run_in_background=True,
            parent_agent=parent_b,
        )
    )
    assert second["status"] == "dispatched"
    assert global_full.wait(1)

    global_overflow = json.loads(
        dt.delegate_task(
            description="overflow-global",
            prompt="overflow-global",
            run_in_background=True,
            parent_agent=parent_c,
        )
    )
    try:
        assert global_overflow["status"] == "rejected"
        assert "global" in global_overflow["error"]
        assert len(built) == 4
        assert dc.active_runner_slots() == 4
        assert dc.active_runner_slots(session_id="root-a") == 2
        assert dc.active_runner_slots(session_id="root-b") == 2
    finally:
        release.set()
        with ad._records_lock:
            futures = [
                record.get("future")
                for record in ad._records.values()
                if record.get("future") is not None
            ]
        for future in futures:
            assert future is not None
            future.result(timeout=3)

    assert dc.active_runner_slots() == 0


def test_initial_timeout_keeps_capacity_until_underlying_worker_exits(monkeypatch):
    import tools.delegate_tool as dt

    release_worker = threading.Event()
    worker_exited = threading.Event()

    _install_fake_delegate_runtime(
        monkeypatch,
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("wrapped fake must be replaced")
        ),
    )

    def timeout_child(task_index, on_runner_finished, **_kwargs):
        def underlying_worker():
            try:
                assert release_worker.wait(3)
            finally:
                worker_exited.set()
                on_runner_finished()

        threading.Thread(target=underlying_worker, daemon=True).start()
        return {
            "task_index": task_index,
            "status": "timeout",
            "summary": None,
            "error": "timed out while worker remains live",
            "exit_reason": "timeout",
            "api_calls": 1,
            "duration_seconds": 0.05,
        }

    monkeypatch.setattr(dt, "_run_single_child", timeout_child)
    monkeypatch.setattr(
        dt,
        "_load_config",
        lambda: {
            "max_concurrent_children": 1,
            "max_global_concurrent_children": 1,
            "foreground_wait_timeout_seconds": 2,
            "child_run_timeout_seconds": 1,
        },
    )
    parent = _parent()

    try:
        timed_out = json.loads(
            dt.delegate_task(
                description="timeout",
                prompt="timeout",
                run_in_background=False,
                parent_agent=parent,
            )
        )
        assert timed_out["results"][0]["status"] == "timeout"
        assert not worker_exited.is_set()
        assert dc.active_runner_slots() == 1
        assert dc.active_runner_slots(session_id="parent-session") == 1

        rejected = json.loads(
            dt.delegate_task(
                description="must not bypass cap",
                prompt="must not bypass cap",
                run_in_background=False,
                parent_agent=parent,
            )
        )
        assert rejected["status"] == "rejected"
        assert "capacity reached" in rejected["error"]
    finally:
        release_worker.set()
        assert worker_exited.wait(2)

    deadline = time.monotonic() + 1
    while dc.active_runner_slots() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert dc.active_runner_slots() == 0


def test_prepared_children_are_closed_when_async_dispatch_rejects(monkeypatch):
    import tools.delegate_tool as dt

    built = _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, description, **_kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": description,
        },
    )
    parent = _parent()
    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        lambda **_kwargs: {"status": "rejected", "error": "pool full"},
    )

    result = json.loads(
        dt.delegate_task(
            description="prepared",
            prompt="prepared",
            run_in_background=True,
            parent_agent=parent,
        )
    )

    assert result["status"] == "rejected"
    assert len(built) == 1
    built[0].close.assert_called_once_with()
    assert parent._active_children == []


def test_prepared_children_are_closed_when_later_child_build_fails(monkeypatch):
    import tools.delegate_tool as dt
    from unittest.mock import MagicMock

    _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, description, **_kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": description,
        },
    )
    parent = _parent()
    first = MagicMock()
    first._subagent_id = "first"
    build_count = 0

    def _build(**_kwargs):
        nonlocal build_count
        build_count += 1
        if build_count == 1:
            parent._active_children.append(first)
            return first
        raise RuntimeError("second build failed")

    monkeypatch.setattr(dt, "_build_child_agent", _build)

    with pytest.raises(RuntimeError, match="second build failed"):
        dt.delegate_task(
            tasks=[
                {"description": "first", "prompt": "first"},
                {"description": "second", "prompt": "second"},
            ],
            run_in_background=True,
            parent_agent=parent,
        )

    first.close.assert_called_once_with()
    assert parent._active_children == []
    assert dc.active_runner_slots() == 0


def test_runner_reservation_released_when_async_preparation_raises(monkeypatch):
    import tools.delegate_tool as dt

    built = _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, description, **_kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": description,
        },
    )
    monkeypatch.setattr(
        "tools.approval.get_current_session_key",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("session key boom")),
    )

    result = json.loads(
        dt.delegate_task(
            description="first",
            prompt="first",
            run_in_background=True,
            parent_agent=_parent(),
        )
    )

    assert result["status"] == "rejected"
    assert "session key boom" in result["error"]
    assert len(built) == 1
    built[0].close.assert_called_once_with()
    assert dc.active_runner_slots() == 0


def test_runner_reservation_released_when_async_registry_raises(monkeypatch):
    import tools.delegate_tool as dt

    _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, description, **_kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": description,
        },
    )
    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("registry boom")),
    )

    result = json.loads(
        dt.delegate_task(
            description="first",
            prompt="first",
            run_in_background=True,
            parent_agent=_parent(),
        )
    )

    assert result["status"] == "rejected"
    assert "registry boom" in result["error"]
    assert dc.active_runner_slots() == 0


def test_background_dispatch_applies_profile_run_cap(monkeypatch):
    import tools.delegate_tool as dt

    seen = []

    def run_child(task_index, description, **kwargs):
        seen.append(kwargs.get("child_timeout_override"))
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "done",
        }

    _install_fake_delegate_runtime(monkeypatch, run_child)

    def fake_dispatch(*, runner, **_kwargs):
        runner()
        return {"status": "dispatched", "delegation_id": "d-background-cap"}

    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        fake_dispatch,
    )

    result = json.loads(
        dt.delegate_task(
            description="inspect",
            prompt="inspect",
            subagent_type="Explore",
            run_in_background=True,
            parent_agent=_parent(),
        )
    )

    assert result["status"] == "dispatched"
    assert seen == [1800]


def test_explicit_foreground_quick_completion_returns_inline_with_profile_run_cap(monkeypatch):
    import tools.delegate_tool as dt

    seen = []

    def run_child(task_index, description, **kwargs):
        seen.append(kwargs.get("child_timeout_override"))
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "quick",
        }

    _install_fake_delegate_runtime(monkeypatch, run_child)
    result = json.loads(
        dt.delegate_task(
            description="inspect", prompt="inspect",
            subagent_type="Explore",
            run_in_background=False,
            parent_agent=_parent(),
        )
    )

    assert result["results"][0]["summary"] == "quick"
    assert seen == [1800]
    assert process_registry.completion_queue.empty()


def test_foreground_wait_timeout_backgrounds_same_delegation_then_delivers_once(monkeypatch):
    import tools.delegate_tool as dt

    gate = threading.Event()

    def run_child(task_index, description, **kwargs):
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
            "foreground_wait_timeout_seconds": 1,
            "child_run_timeout_seconds": 321,
        },
    )

    timed_out = json.loads(
        dt.delegate_task(
            description="slow inspect", prompt="slow inspect",
            subagent_type="Explore",
            run_in_background=False,
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
        lambda task_index, description, **kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": "unexpected",
        },
    )
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 2)
    result = json.loads(
        dt.delegate_task(
            description="worker",
            prompt="run worker",
            run_in_background=True,
            parent_agent=_parent(depth=1),
        )
    )

    assert result["error"] == "Nested delegation cannot run in the background."
    assert built == []
    assert ad.active_count() == 0


def test_nested_batch_rejects_removed_per_task_scheduling(monkeypatch):
    import tools.delegate_tool as dt

    built = _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, description, **kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": "unexpected",
        },
    )
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 2)
    result = json.loads(
        dt.delegate_task(
            tasks=[
                {
                    "description": "worker",
                    "prompt": "run worker",
                    "scheduling": "background",
                }
            ],
            parent_agent=_parent(depth=1),
        )
    )

    assert "unsupported fields: scheduling" in result["error"]
    assert built == []


def test_nested_auto_runs_synchronously_without_async_delivery(monkeypatch):
    import tools.delegate_tool as dt

    _install_fake_delegate_runtime(
        monkeypatch,
        lambda task_index, description, **kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": "nested inline",
        },
    )
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 2)

    result = json.loads(
        dt.delegate_task(
            description="worker", prompt="worker",
            run_in_background=None,
            parent_agent=_parent(depth=1),

        )
    )

    assert result["results"][0]["summary"] == "nested inline"
    assert "delegation_id" not in result
    assert process_registry.completion_queue.empty()
    assert ad.active_count() == 0


def test_nested_foreground_uses_profile_child_run_cap(monkeypatch):
    import tools.delegate_tool as dt

    seen = []

    def run_child(task_index, description, **kwargs):
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
            description="inspect", prompt="inspect",
            subagent_type="Explore",
            run_in_background=None,
            parent_agent=_parent(depth=1),

        )
    )

    assert result["results"][0]["summary"] == "nested capped"
    assert seen == [1800]
    assert "delegation_id" not in result


def test_legacy_model_auto_dispatches_background(monkeypatch):
    import tools.delegate_tool as dt

    gate = threading.Event()

    def run_child(task_index, description, **kwargs):
        assert kwargs["child_timeout_override"] == 7200
        gate.wait(timeout=2)
        return {"task_index": task_index, "status": "completed", "summary": "model bg"}

    _install_fake_delegate_runtime(monkeypatch, run_child)
    dispatched = json.loads(
        dt.delegate_task(
            description="legacy model call", prompt="legacy model call",
            parent_agent=_parent(),

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
        lambda task_index, description, **kwargs: {
            "task_index": task_index,
            "status": "completed",
            "summary": "inline safe",
        },
    )
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: False)

    result = json.loads(
        dt.delegate_task(
            description="legacy model call", prompt="legacy model call",
            parent_agent=_parent(),

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

    def run_child(task_index, description, **kwargs):
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
        lambda: {"foreground_wait_timeout_seconds": 1},
    )

    result = json.loads(
        dt.delegate_task(
            description="inspect", prompt="inspect",
            subagent_type="Explore",
            run_in_background=False,
            parent_agent=_parent(),

        )
    )

    assert result["results"][0]["summary"] == "foreground inline safe"
    assert "delegation_id" not in result
    assert seen == [1800]
    assert ad.active_count() == 0


def test_async_pool_rejection_does_not_run_extra_child_inline(monkeypatch):
    import tools.delegate_tool as dt

    run_count = 0

    def run_child(task_index, description, **kwargs):
        nonlocal run_count
        run_count += 1
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "should not run",
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
            description="inspect", prompt="inspect",
            subagent_type="Explore",
            run_in_background=False,
            parent_agent=parent,
        )
    )

    assert len(built) == 1
    assert run_count == 0
    assert result["status"] == "rejected"
    assert "capacity reached" in result["error"]
    assert "not started" in result["note"]
    assert parent._active_children == []


def test_pure_background_uses_general_purpose_profile_run_cap(monkeypatch):
    import tools.delegate_tool as dt

    gate = threading.Event()
    seen = []

    def run_child(task_index, description, **kwargs):
        seen.append(("child_timeout_override" in kwargs, kwargs.get("child_timeout_override")))
        gate.wait(timeout=2)
        return {"task_index": task_index, "status": "completed", "summary": "done"}

    _install_fake_delegate_runtime(monkeypatch, run_child)
    dispatched = json.loads(
        dt.delegate_task(description="legacy bg", prompt="legacy bg", run_in_background=True, parent_agent=_parent())
    )
    assert dispatched["status"] == "dispatched"
    gate.set()
    event = process_registry.completion_queue.get(timeout=2)

    assert event["results"][0]["summary"] == "done"
    assert seen == [(True, 7200)]


def test_mixed_explicit_foreground_batch_uses_each_profile_run_cap(monkeypatch):
    import tools.delegate_tool as dt

    seen = {}

    def run_child(task_index, description, **kwargs):
        seen[description] = kwargs.get("child_timeout_override")
        return {"task_index": task_index, "status": "completed", "summary": description}

    _install_fake_delegate_runtime(monkeypatch, run_child)
    result = json.loads(
        dt.delegate_task(
            tasks=[
                {
                    "description": "explore",
                    "prompt": "explore the implementation",
                    "subagent_type": "Explore",
                },
                {
                    "description": "plan",
                    "prompt": "plan the implementation",
                    "subagent_type": "Plan",
                },
            ],
            run_in_background=False,
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


def test_model_schema_truthfully_describes_background_and_batch_delivery():
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA

    description = DELEGATE_TASK_SCHEMA["description"]
    assert "run_in_background=false" in description
    assert "one batch handle" in description
    assert "one consolidated completion" in description

    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert "background" not in props
    assert "scheduling" not in props
    assert props["run_in_background"] == {"type": "boolean"}
