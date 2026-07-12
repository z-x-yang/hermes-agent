#!/usr/bin/env python3
"""
Async (background) delegation registry.

Backs ``delegate_task(run_in_background=true)``: the parent agent dispatches a
subagent that runs on a module-level daemon executor and returns a handle
immediately, so the user and the model can keep working while the child runs.

When the child finishes, a completion event is pushed onto the SHARED
``process_registry.completion_queue`` with ``type="async_delegation"``. The
CLI (``cli.py`` process_loop) and gateway (``_run_process_watcher`` /
``completion_queue`` drain) already poll that queue while the agent is idle
and forge a fresh user/internal turn from each event. We deliberately reuse
that rail rather than reaching into a running agent loop:

  - completions surface as a NEW turn when the agent is idle, never spliced
    between a tool result and an assistant message. That keeps strict
    message-role alternation legal and the prompt cache intact (hard
    invariant: never mutate past context).
  - we inherit the queue's de-dup, crash-recovery checkpoint, and the
    existing CLI + gateway drain wiring for free — no new drain loops in the
    two largest files in the repo.

The completion payload carries a RICH, self-contained task-source block (the
original goal, the context the parent supplied, toolsets, model, dispatch
time, status, and the full result summary). When the result re-enters the
conversation the parent may be deep in unrelated context and won't remember
why the subagent existed; the block lets it either use the result or
re-dispatch if the world has moved on.

This module owns ONLY the async lifecycle. The actual child build + run is
delegated back to ``delegate_tool._run_single_child`` via an injected
runner, so all the credential leasing, heartbeat, timeout, and result-shaping
logic stays in one place.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Optional

from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.thread_context import propagate_context_to_thread

logger = logging.getLogger(__name__)

DeliveryMode = Literal[
    "foreground_waiting",
    "foreground_claimed",
    "foreground_interrupted",
    "background",
    "delivered",
]

# Back-compat alias — the daemon executor now lives in tools.daemon_pool so
# other subsystems (tool_executor, memory_manager, delegate_tool, skills_hub)
# can share it. Existing imports of ``_DaemonThreadPoolExecutor`` keep working.
_DaemonThreadPoolExecutor = DaemonThreadPoolExecutor


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
# A persistent daemon executor (NOT a `with ThreadPoolExecutor()` block, which
# would join on exit and defeat the whole point of async). Workers are daemon
# threads so a hard process exit doesn't hang on an in-flight child.
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_executor_max_workers: int = 0


def _submit_with_commit_gate(executor, fn):
    """Submit *fn* without exposing an enqueue-before-return handoff race.

    Some executor implementations can enqueue work and then raise while growing
    their worker pool. The gated wrapper cannot call *fn* until submission is
    committed. If submission raises first, queued work wakes only to exit. Once
    committed, a BaseException in this tiny handoff window is suppressed so the
    caller cannot mistake a live worker for a rejected dispatch.
    """

    ready = threading.Event()
    state = {"committed": False}
    future = None

    def _gated():
        ready.wait()
        if not state["committed"]:
            return None
        return fn()

    try:
        future = executor.submit(propagate_context_to_thread(_gated))
        state["committed"] = True
        ready.set()
        return future
    except BaseException:
        if state["committed"] and future is not None:
            ready.set()
            return future
        ready.set()
        raise

_records_lock = threading.Lock()
# delegation_id -> record dict. Kept for the lifetime of the run plus a short
# tail after completion so `list_async_delegations()` can show recent results.
_records: Dict[str, Dict[str, Any]] = {}

_DEFAULT_MAX_ASYNC_CHILDREN = 3
# How many completed records to retain for status queries before pruning.
_MAX_RETAINED_COMPLETED = 50


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    """Lazily create (or grow) the shared daemon executor.

    We never shrink — ThreadPoolExecutor can't resize — but if the configured
    cap grows between calls we rebuild a larger pool. Existing in-flight
    futures keep running on the old pool until it's garbage collected.
    """
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is None or max_workers > _executor_max_workers:
            # Daemon threads: thread_name_prefix aids debugging in stack dumps.
            _executor = _DaemonThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="async-delegate",
            )
            _executor_max_workers = max_workers
        return _executor


def active_count() -> int:
    """Number of async delegations currently running."""
    with _records_lock:
        return sum(1 for r in _records.values() if r.get("status") == "running")


def _new_delegation_id() -> str:
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _prune_completed_locked() -> None:
    """Drop the oldest completed records beyond the retention cap.

    Caller must hold ``_records_lock``.
    """
    completed = [
        (rid, r)
        for rid, r in _records.items()
        if r.get("status") != "running"
        and r.get("delivery_mode") != "foreground_waiting"
    ]
    if len(completed) <= _MAX_RETAINED_COMPLETED:
        return
    # Oldest-first by completion time (fall back to dispatch time).
    completed.sort(key=lambda kv: kv[1].get("completed_at") or kv[1].get("dispatched_at") or 0)
    for rid, _ in completed[: len(completed) - _MAX_RETAINED_COMPLETED]:
        _records.pop(rid, None)


def dispatch_async_delegation(
    *,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    session_key: str,
    runner: Callable[[], Dict[str, Any]],
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
) -> Dict[str, Any]:
    """Spawn ``runner`` on the daemon executor and return a handle immediately.

    Parameters
    ----------
    goal, context, toolsets, model
        The dispatch-time task spec, captured verbatim for the rich
        completion block.
    session_key
        The gateway session_key (from ``tools.approval.get_current_session_key``)
        captured on the parent thread BEFORE dispatch, because the daemon
        worker thread won't carry the contextvar. Used to route the
        completion back to the originating session.
    runner
        Zero-arg callable that builds + runs the child and returns the same
        result dict ``_run_single_child`` produces. Runs on the worker thread.
    interrupt_fn
        Optional callable to signal the child to stop (used on shutdown /
        explicit cancel).
    max_async_children
        Concurrency cap. When at capacity the dispatch is REJECTED rather than
        queued or converted into inline work, so a runaway model cannot exceed
        the global child cap.

    Returns
    -------
    dict
        ``{"status": "dispatched", "delegation_id": ...}`` on success, or
        ``{"status": "rejected", "error": ...}`` when at capacity.
    """
    delegation_id = _new_delegation_id()
    dispatched_at = time.time()
    record: Dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": goal,
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "model": model,
        "session_key": session_key,
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
    }
    try:
        executor = _get_executor(max_async_children)
    except Exception as exc:
        return {
            "status": "rejected",
            "error": f"Failed to prepare async delegation executor: {exc}",
        }
    # Capacity check and record insert under ONE lock hold — checking
    # active_count() separately would let two concurrent dispatches (e.g.
    # from different gateway sessions) both pass the check and exceed the cap.
    with _records_lock:
        running = sum(
            1 for r in _records.values() if r.get("status") == "running"
        )
        if running >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"running). Wait for one to finish (its result will re-enter "
                    f"the chat), or run this task synchronously "
                    f"(background=false). Raise delegation.max_global_concurrent_children in "
                    f"config.yaml to allow more concurrent background subagents."
                ),
            }
        _records[delegation_id] = record

    def _worker() -> None:
        result: Dict[str, Any] = {}
        status = "error"
        try:
            result = runner() or {}
            status = result.get("status") or "completed"
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception("Async delegation %s crashed", delegation_id)
            result = {
                "status": "error",
                "summary": None,
                "error": f"{type(exc).__name__}: {exc}",
                "api_calls": 0,
                "duration_seconds": round(time.time() - dispatched_at, 2),
            }
            status = "error"
        finally:
            _finalize(delegation_id, result, status)

    try:
        _submit_with_commit_gate(executor, _worker)
    except Exception as exc:  # pragma: no cover — pool submit failure is rare
        with _records_lock:
            _records.pop(delegation_id, None)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation: {exc}",
        }
    except BaseException:
        with _records_lock:
            _records.pop(delegation_id, None)
        raise

    try:
        logger.info(
            "Dispatched async delegation %s (session_key=%s): %s",
            delegation_id, session_key or "<cli>", (goal or "")[:80],
        )
    except BaseException:
        pass
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize(delegation_id: str, result: Dict[str, Any], status: str) -> None:
    """Mark a record complete and push the completion event onto the queue."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None:
            return
        record["status"] = status
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None  # drop the closure; child is done
        # Snapshot fields needed for the event while holding the lock.
        event_record = dict(record)
        _prune_completed_locked()

    _push_completion_event(event_record, result, status)


def _push_completion_event(
    record: Dict[str, Any], result: Dict[str, Any], status: str
) -> None:
    """Push a type='async_delegation' event onto the shared completion queue.

    Best-effort: a failure here must not crash the worker, but it WOULD mean a
    silently-lost result, so we log loudly.
    """
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s finished but process_registry import failed; "
            "result lost: %s",
            record.get("delegation_id"), exc,
        )
        return

    summary = result.get("summary")
    error = result.get("error")
    dispatched_at = record.get("dispatched_at") or time.time()
    completed_at = record.get("completed_at") or time.time()

    evt = {
        "type": "async_delegation",
        "delegation_id": record.get("delegation_id"),
        # session_key routes the completion back to the originating gateway
        # session; empty string => CLI (single-session) path.
        "session_key": record.get("session_key", ""),
        "goal": record.get("goal", ""),
        "context": record.get("context"),
        "toolsets": record.get("toolsets"),
        "model": result.get("model") or record.get("model"),
        "status": status,
        "summary": summary,
        "error": error,
        "api_calls": result.get("api_calls", 0),
        "duration_seconds": result.get(
            "duration_seconds", round(completed_at - dispatched_at, 2)
        ),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
        "exit_reason": result.get("exit_reason"),
    }
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s: failed to enqueue completion event; "
            "result lost: %s",
            record.get("delegation_id"), exc,
        )


def dispatch_async_delegation_batch(
    *,
    goals: List[str],
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    session_key: str,
    runner: Callable[[], Dict[str, Any]],
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    initial_delivery_mode: DeliveryMode = "background",
    inject_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Dispatch a whole fan-out batch as one async registry unit.

    ``runner`` executes the entire batch, joining every child and returning the
    same combined ``{"results": [...], "total_duration_seconds": N}`` payload
    as the synchronous path. The unit begins in either background-delivery or
    foreground-waiting mode; a timed-out foreground waiter atomically hands the
    same future to background delivery.

    The batch occupies one async slot (in-batch parallelism is bounded by
    ``max_concurrent_children``). Background completion preserves the existing
    single consolidated ``process_registry.completion_queue`` event schema.

    Returns ``{"status": "dispatched", "delegation_id": ...}`` on success or
    ``{"status": "rejected", "error": ...}`` when the async pool is at
    capacity.
    """
    delegation_id = _new_delegation_id()
    dispatched_at = time.time()
    n = len(goals)
    # A combined goal label for status listings / the completion header.
    combined_goal = (
        goals[0] if n == 1 else f"{n} parallel subagents: " + "; ".join(g[:40] for g in goals)
    )
    record: Dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": combined_goal,
        "goals": list(goals),
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "model": model,
        "session_key": session_key,
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "is_batch": True,
        "delivery_mode": initial_delivery_mode,
        "done_event": threading.Event(),
        "result_payload": None,
        "combined_result": None,
        "future": None,
        "inject_fn": inject_fn,
    }
    try:
        executor = _get_executor(max_async_children)
    except Exception as exc:
        return {
            "status": "rejected",
            "error": f"Failed to prepare async delegation batch executor: {exc}",
        }
    with _records_lock:
        running = sum(
            1 for r in _records.values() if r.get("status") == "running"
        )
        if running >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"running). Wait for one to finish (its result will re-enter "
                    f"the chat), or raise delegation.max_global_concurrent_children in "
                    f"config.yaml to allow more concurrent background units."
                ),
            }
        _records[delegation_id] = record

    def _worker() -> None:
        combined: Dict[str, Any] = {}
        status = "error"
        try:
            combined = runner() or {}
            # Batch status: completed unless every child errored/was interrupted.
            child_results = combined.get("results") or []
            if child_results and all(
                (r.get("status") not in ("completed", "success"))
                for r in child_results
            ):
                status = "error"
            else:
                status = "completed"
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception("Async delegation batch %s crashed", delegation_id)
            combined = {
                "results": [],
                "error": f"{type(exc).__name__}: {exc}",
                "total_duration_seconds": round(time.time() - dispatched_at, 2),
            }
            status = "error"
        finally:
            _finalize_batch(delegation_id, combined, status)

    try:
        future = _submit_with_commit_gate(executor, _worker)
    except Exception as exc:  # pragma: no cover
        with _records_lock:
            _records.pop(delegation_id, None)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation batch: {exc}",
        }
    except BaseException:
        with _records_lock:
            _records.pop(delegation_id, None)
        raise

    # Handoff is committed. From here onward never propagate a bookkeeping or
    # logging failure as a rejected dispatch while the worker may be live.
    try:
        with _records_lock:
            current = _records.get(delegation_id)
            if current is not None:
                current["future"] = future
    except BaseException:
        pass

    try:
        logger.info(
            "Dispatched async delegation batch %s (%d task(s), session_key=%s)",
            delegation_id, n, session_key or "<cli>",
        )
    except BaseException:
        pass
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize_batch(
    delegation_id: str, combined: Dict[str, Any], status: str
) -> None:
    """Record completion, then deliver it to exactly one consumer."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None:
            return
        record["status"] = status
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None
        record["combined_result"] = combined
        record["result_payload"] = json.dumps(combined, ensure_ascii=False)
        event = _build_batch_completion_event(dict(record), combined, status)
        record["done_event"].set()
        should_inject = record.get("delivery_mode") == "background"
        if should_inject:
            record["delivery_mode"] = "delivered"
        inject_fn = record.get("inject_fn")
        _prune_completed_locked()

    if not should_inject:
        return
    if inject_fn is not None:
        inject_fn(event)
        return
    _push_batch_completion_event(event)


def _build_batch_completion_event(
    event_record: Dict[str, Any], combined: Dict[str, Any], status: str
) -> Dict[str, Any]:
    """Build the existing consolidated completion event schema."""
    dispatched_at = event_record.get("dispatched_at") or time.time()
    completed_at = event_record.get("completed_at") or time.time()
    return {
        "type": "async_delegation",
        "delegation_id": event_record.get("delegation_id"),
        "session_key": event_record.get("session_key", ""),
        "goal": event_record.get("goal", ""),
        "goals": event_record.get("goals"),
        "context": event_record.get("context"),
        "toolsets": event_record.get("toolsets"),
        "model": event_record.get("model"),
        "status": status,
        "is_batch": True,
        # The full per-task results list — the formatter renders a
        # consolidated multi-task block from this.
        "results": combined.get("results") or [],
        "error": combined.get("error"),
        "total_duration_seconds": combined.get("total_duration_seconds"),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
    }


def _push_batch_completion_event(event: Dict[str, Any]) -> None:
    """Push one already-built batch event onto the production queue."""
    delegation_id = event.get("delegation_id")
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s finished but process_registry import "
            "failed; result lost: %s",
            delegation_id, exc,
        )
        return

    try:
        process_registry.completion_queue.put(event)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s: failed to enqueue completion event; "
            "result lost: %s",
            delegation_id, exc,
        )


def wait_for_async_delegation(
    record_or_id: Any,
    timeout_seconds: float,
    *,
    interrupt_requested: Optional[Callable[[], bool]] = None,
) -> Optional[str]:
    """Claim foreground delivery, interrupt it, or hand the same work to background."""
    if isinstance(record_or_id, dict):
        delegation_id = record_or_id.get("delegation_id")
    else:
        delegation_id = record_or_id
    if not delegation_id:
        return None

    with _records_lock:
        record = _records.get(str(delegation_id))
        if record is None:
            return None
        done_event = record.get("done_event")
    if done_event is None:
        return None

    timeout = max(0.0, float(timeout_seconds))
    deadline = time.monotonic() + timeout
    while True:
        with _records_lock:
            record = _records.get(str(delegation_id))
            if record is None or record.get("delivery_mode") != "foreground_waiting":
                return None
            payload = record.get("result_payload")
            if payload is not None:
                record["delivery_mode"] = "foreground_claimed"
                return payload

        parent_interrupted = False
        if interrupt_requested is not None:
            try:
                parent_interrupted = bool(interrupt_requested())
            except Exception:
                logger.debug(
                    "Foreground delegation %s interrupt predicate failed",
                    delegation_id,
                    exc_info=True,
                )
        if parent_interrupted:
            with _records_lock:
                record = _records.get(str(delegation_id))
                if record is None or record.get("delivery_mode") != "foreground_waiting":
                    return None
                payload = record.get("result_payload")
                if payload is not None:
                    record["delivery_mode"] = "foreground_claimed"
                    return payload
                record["delivery_mode"] = "foreground_interrupted"
                interrupt_fn = record.get("interrupt_fn")
            if callable(interrupt_fn):
                try:
                    interrupt_fn()
                except Exception:
                    logger.debug(
                        "Foreground delegation %s interrupt failed",
                        delegation_id,
                        exc_info=True,
                    )
            return json.dumps(
                {
                    "status": "interrupted",
                    "mode": "foreground",
                    "delegation_id": str(delegation_id),
                    "error": "Foreground delegation interrupted by parent.",
                    "note": (
                        "Late completion delivery is suppressed for this delegation."
                    ),
                },
                ensure_ascii=False,
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            with _records_lock:
                record = _records.get(str(delegation_id))
                if record is None or record.get("delivery_mode") != "foreground_waiting":
                    return None
                payload = record.get("result_payload")
                if payload is not None:
                    record["delivery_mode"] = "foreground_claimed"
                    return payload
                record["delivery_mode"] = "background"
                return None
        done_event.wait(timeout=min(0.05, remaining))


def list_async_delegations() -> List[Dict[str, Any]]:
    """Snapshot of async delegations (running + recently completed).

    Safe to call from any thread. Excludes runtime-only synchronization fields.
    """
    with _records_lock:
        return [
            {
                k: v
                for k, v in r.items()
                if k
                not in {
                    "interrupt_fn",
                    "future",
                    "done_event",
                    "delivery_lock",
                    "inject_fn",
                }
            }
            for r in _records.values()
        ]


def interrupt_all(reason: str = "shutdown") -> int:
    """Signal every running async delegation to stop. Returns how many.

    Used on ``/stop`` and gateway shutdown so a dangling background subagent
    can't keep burning tokens with no one listening. The child still emits a
    completion event (status='interrupted') via the normal finalize path.
    """
    count = 0
    with _records_lock:
        targets = [
            r for r in _records.values() if r.get("status") == "running"
        ]
    for r in targets:
        fn = r.get("interrupt_fn")
        if callable(fn):
            try:
                fn()
                count += 1
            except Exception as exc:
                logger.debug(
                    "interrupt_all: %s interrupt failed: %s",
                    r.get("delegation_id"), exc,
                )
    if count:
        logger.info("Interrupted %d async delegation(s) (%s)", count, reason)
    return count


def _reset_for_tests() -> None:
    """Test-only: clear all state and tear down the executor."""
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = None
        _executor_max_workers = 0
    with _records_lock:
        _records.clear()
