from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import threading
import time
from typing import Any


@dataclass(frozen=True)
class RetainedSubagentSession:
    """Short-lived in-process metadata for a resumable child agent session.

    This intentionally stores only non-secret routing/capability metadata plus
    the child transcript. Credentials are re-resolved from trusted parent/config
    state at continuation time.
    """

    agent_id: str
    parent_session_id: str
    subagent_type: str
    role: str
    workspace_path: str
    model: str
    provider: str
    conversation_history: list[dict[str, Any]]
    created_at: float
    expires_at: float


_lock = threading.RLock()
_records: dict[str, RetainedSubagentSession] = {}
_in_flight: set[str] = set()


def _copy_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy([item for item in list(history or []) if isinstance(item, dict)])


def _copy_record(record: RetainedSubagentSession) -> RetainedSubagentSession:
    return replace(record, conversation_history=_copy_history(record.conversation_history))


def _prune(now: float, max_records: int) -> None:
    expired = [
        key
        for key, value in _records.items()
        if value.expires_at <= now and key not in _in_flight
    ]
    for key in expired:
        _records.pop(key, None)
    while len(_records) >= max(1, int(max_records or 1)):
        removable = [
            record for key, record in _records.items() if key not in _in_flight
        ]
        if not removable:
            break
        oldest = min(removable, key=lambda item: item.created_at)
        _records.pop(oldest.agent_id, None)


def retain_subagent_session(
    record: RetainedSubagentSession,
    *,
    max_records: int = 64,
) -> None:
    with _lock:
        capacity = max(1, int(max_records or 1))
        _prune(time.time(), capacity)
        if len(_records) >= capacity:
            raise RuntimeError(
                f"Retained subagent session capacity reached ({capacity} records); "
                "all retained sessions are in flight."
            )
        _records[record.agent_id] = _copy_record(record)


def get_retained_subagent_session(agent_id: str) -> RetainedSubagentSession:
    with _lock:
        now = time.time()
        record = _records.get(agent_id)
        if record is None:
            raise KeyError(f"Unknown retained subagent session: {agent_id}")
        if record.expires_at <= now and agent_id not in _in_flight:
            _records.pop(agent_id, None)
            raise KeyError(f"Retained subagent session expired: {agent_id}")
        return _copy_record(record)


def claim_retained_subagent_session(agent_id: str) -> RetainedSubagentSession:
    """Atomically claim one retained transcript for continuation."""
    with _lock:
        record = get_retained_subagent_session(agent_id)
        if agent_id in _in_flight:
            raise RuntimeError(
                f"Retained subagent continuation already in progress: {agent_id}"
            )
        _in_flight.add(agent_id)
        return record


def release_retained_subagent_session(agent_id: str) -> None:
    with _lock:
        _in_flight.discard(agent_id)


def update_retained_history(agent_id: str, history: list[dict[str, Any]]) -> None:
    with _lock:
        record = get_retained_subagent_session(agent_id)
        _records[agent_id] = replace(record, conversation_history=_copy_history(history))


def clear_retained_subagent_sessions() -> None:
    with _lock:
        _records.clear()
        _in_flight.clear()
