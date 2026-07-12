"""Dual global/per-root-session delegation capacity tests."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time
from typing import cast

import pytest

from tools import delegation_capacity as dc


@pytest.fixture(autouse=True)
def _clean_capacity_state():
    dc._reset_for_tests()
    yield
    dc._reset_for_tests()


def test_runner_capacity_enforces_session_and_global_limits_atomically():
    session_a = dc.try_reserve_runner_slots(
        5,
        global_limit=20,
        session_id="root-a",
        session_limit=5,
    )
    assert session_a is not None
    assert dc.active_runner_slots() == 5
    assert dc.active_runner_slots(session_id="root-a") == 5

    assert dc.try_reserve_runner_slots(
        1,
        global_limit=20,
        session_id="root-a",
        session_limit=5,
    ) is None

    session_b = dc.try_reserve_runner_slots(
        15,
        global_limit=20,
        session_id="root-b",
        session_limit=15,
    )
    assert session_b is not None
    assert dc.active_runner_slots() == 20
    assert dc.active_runner_slots(session_id="root-b") == 15

    assert dc.try_reserve_runner_slots(
        1,
        global_limit=20,
        session_id="root-c",
        session_limit=5,
    ) is None

    session_a.release()
    assert dc.active_runner_slots() == 15
    assert dc.active_runner_slots(session_id="root-a") == 0

    session_c = dc.try_reserve_runner_slots(
        1,
        global_limit=20,
        session_id="root-c",
        session_limit=5,
    )
    assert session_c is not None
    assert dc.active_runner_slots() == 16
    assert dc.active_runner_slots(session_id="root-c") == 1

    session_a.release()  # idempotent
    session_b.release()
    session_c.release()
    assert dc.active_runner_slots() == 0


def test_runner_capacity_rejects_empty_session_identity():
    assert dc.try_reserve_runner_slots(
        1,
        global_limit=20,
        session_id="",
        session_limit=5,
    ) is None


def test_runner_capacity_rejects_non_integer_limits():
    with pytest.raises(ValueError, match="positive integers"):
        dc.try_reserve_runner_slots(
            True,
            global_limit=20,
            session_id="root-a",
            session_limit=5,
        )
    with pytest.raises(ValueError, match="positive integers"):
        dc.try_reserve_runner_slots(
            1,
            global_limit=True,
            session_id="root-a",
            session_limit=5,
        )
    with pytest.raises(ValueError, match="positive integers"):
        dc.try_reserve_runner_slots(
            1,
            global_limit=20,
            session_id="root-a",
            session_limit=False,
        )


def test_enqueue_then_raise_submit_never_starts_child_runner():
    from tools.delegate_tool import _submit_with_context_commit_gate

    runner_called = threading.Event()
    queued_wrapper_exited = threading.Event()

    class EnqueueThenRaiseExecutor:
        def submit(self, fn, *args):
            def queued():
                try:
                    fn(*args)
                finally:
                    queued_wrapper_exited.set()

            threading.Thread(target=queued, daemon=True).start()
            raise RuntimeError("submit raised after enqueue")

    with pytest.raises(RuntimeError, match="after enqueue"):
        _submit_with_context_commit_gate(
            cast(ThreadPoolExecutor, EnqueueThenRaiseExecutor()),
            runner_called.set,
        )

    assert queued_wrapper_exited.wait(1)
    assert not runner_called.is_set()


def test_stateless_parent_gets_stable_non_reusable_capacity_owner():
    from tools.delegate_tool import _delegation_root_session_id

    parent_a = SimpleNamespace(session_id=None)
    parent_b = SimpleNamespace(session_id=None)

    owner_a = _delegation_root_session_id(parent_a)
    assert owner_a.startswith("ephemeral-agent:")
    assert _delegation_root_session_id(parent_a) == owner_a
    assert parent_a._delegate_root_session_id == owner_a

    owner_b = _delegation_root_session_id(parent_b)
    assert owner_b.startswith("ephemeral-agent:")
    assert owner_b != owner_a


class _NoDynamicAttributes:
    __slots__ = ()


def test_stateless_parent_that_cannot_retain_owner_fails_closed():
    from tools.delegate_tool import _delegation_root_session_id

    assert _delegation_root_session_id(_NoDynamicAttributes()) == ""


def test_timeout_helper_enqueue_then_raise_releases_without_starting_worker(
    monkeypatch,
):
    from tools.delegate_tool import _run_child_conversation_with_timeout

    reservation = dc.try_reserve_runner_slots(
        1,
        global_limit=1,
        session_id="root-timeout-submit",
        session_limit=1,
    )
    assert reservation is not None
    runner_called = threading.Event()
    queued_wrapper_exited = threading.Event()

    class EnqueueThenRaiseExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, fn, *args):
            def queued():
                try:
                    fn(*args)
                finally:
                    queued_wrapper_exited.set()

            threading.Thread(target=queued, daemon=True).start()
            raise RuntimeError("timeout submit raised after enqueue")

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(
        "tools.daemon_pool.DaemonThreadPoolExecutor",
        EnqueueThenRaiseExecutor,
    )

    with pytest.raises(RuntimeError, match="after enqueue"):
        _run_child_conversation_with_timeout(
            child=SimpleNamespace(),
            run_callable=lambda: runner_called.set(),
            timeout_seconds=1,
            task_index=0,
            goal="must abort",
            child_start=time.monotonic(),
            on_worker_finished=reservation.release_callback(0),
            handoff_state={"handed_off": False},
        )

    assert queued_wrapper_exited.wait(1)
    assert not runner_called.is_set()
    assert dc.active_runner_slots() == 0


def test_timeout_helper_import_failure_releases_slot(monkeypatch):
    import builtins
    from tools.delegate_tool import _run_child_conversation_with_timeout

    reservation = dc.try_reserve_runner_slots(
        1,
        global_limit=1,
        session_id="root-import-failure",
        session_limit=1,
    )
    assert reservation is not None
    original_import = builtins.__import__

    def fail_daemon_pool_import(name, *args, **kwargs):
        if name == "tools.daemon_pool":
            raise ImportError("daemon pool unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_daemon_pool_import)
    with pytest.raises(ImportError, match="daemon pool unavailable"):
        _run_child_conversation_with_timeout(
            child=SimpleNamespace(),
            run_callable=lambda: {"completed": True},
            timeout_seconds=1,
            task_index=0,
            goal="import failure",
            child_start=time.monotonic(),
            on_worker_finished=reservation.release_callback(0),
        )
    assert dc.active_runner_slots() == 0


def test_runner_slot_released_when_credential_lease_setup_fails():
    from tools.delegate_tool import _run_single_child

    reservation = dc.try_reserve_runner_slots(
        1,
        global_limit=1,
        session_id="root-lease-failure",
        session_limit=1,
    )
    assert reservation is not None

    class BrokenPool:
        def acquire_lease(self):
            raise RuntimeError("lease setup failed")

    child = SimpleNamespace(
        tool_progress_callback=None,
        _delegate_saved_tool_names=[],
        _credential_pool=BrokenPool(),
    )

    with pytest.raises(RuntimeError, match="lease setup failed"):
        _run_single_child(
            task_index=0,
            description="lease failure",
            child=child,
            parent_agent=SimpleNamespace(),
            prompt="fail before worker start",
            on_runner_finished=reservation.release_callback(0),
        )
    assert dc.active_runner_slots() == 0


def test_timeout_executor_initializer_failure_releases_slot(monkeypatch):
    from tools.delegate_tool import _run_child_conversation_with_timeout

    reservation = dc.try_reserve_runner_slots(
        1,
        global_limit=1,
        session_id="root-initializer-failure",
        session_limit=1,
    )
    assert reservation is not None

    def fail_initializer(*_args, **_kwargs):
        raise RuntimeError("initializer failed")

    monkeypatch.setattr(
        "tools.delegate_tool._set_subagent_approval_cb", fail_initializer
    )
    result, error_entry = _run_child_conversation_with_timeout(
        child=SimpleNamespace(
            interrupt=lambda: None,
            get_activity_summary=lambda: {"api_call_count": 0},
        ),
        run_callable=lambda: {"completed": True},
        timeout_seconds=1,
        task_index=0,
        goal="initializer failure",
        child_start=time.monotonic(),
        on_worker_finished=reservation.release_callback(0),
    )

    assert result is None
    assert error_entry is not None
    assert error_entry["status"] == "error"
    assert dc.active_runner_slots() == 0


def test_timeout_helper_defers_slot_release_until_worker_exit():
    from tools.delegate_tool import _run_child_conversation_with_timeout

    reservation = dc.try_reserve_runner_slots(
        1,
        global_limit=1,
        session_id="root-timeout",
        session_limit=1,
    )
    assert reservation is not None
    worker_started = threading.Event()
    allow_worker_exit = threading.Event()
    worker_exited = threading.Event()

    def run_callable():
        worker_started.set()
        try:
            assert allow_worker_exit.wait(3)
            return {"completed": True, "final_response": "late"}
        finally:
            worker_exited.set()

    child = SimpleNamespace(
        interrupt=lambda: None,
        get_activity_summary=lambda: {"api_call_count": 1},
    )
    result, timeout_entry = _run_child_conversation_with_timeout(
        child=child,
        run_callable=run_callable,
        timeout_seconds=0.05,
        task_index=0,
        goal="ignore interrupt",
        child_start=time.monotonic(),
        on_worker_finished=reservation.release,
    )

    assert worker_started.is_set()
    assert result is None
    assert timeout_entry is not None
    assert timeout_entry["status"] == "timeout"
    assert dc.active_runner_slots() == 1
    assert dc.active_runner_slots(session_id="root-timeout") == 1

    allow_worker_exit.set()
    assert worker_exited.wait(1)
    deadline = time.monotonic() + 1
    while dc.active_runner_slots() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert dc.active_runner_slots() == 0


def test_default_delegation_limits_are_global_20_session_5_depth_2():
    from hermes_cli.config import DEFAULT_CONFIG

    delegation = DEFAULT_CONFIG["delegation"]
    assert delegation["max_global_concurrent_children"] == 20
    assert delegation["max_concurrent_children"] == 5
    assert delegation["max_spawn_depth"] == 2
