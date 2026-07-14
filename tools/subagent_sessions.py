from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import json
import threading
import time
from typing import Any, Optional


class RetainedClaimCancelled(RuntimeError):
    """An interrupted claim attempted to commit late retained state."""


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
    workspace_path: str
    model: str
    provider: str
    conversation_history: list[dict[str, Any]]
    created_at: float
    expires_at: float
    profile_id: str
    canonical_profile_home: str
    original_policy_identities: frozenset[str]
    transport_identity: str = ""
    effective_allowed_tool_names: frozenset[str] = frozenset()
    claim_generation: int = 0
    updated_at: float = 0.0
    status: str = "completed"
    tool_trace_metadata: tuple[tuple[str, int, int, str], ...] = ()
    files_written: tuple[str, ...] = ()


_lock = threading.RLock()
_records: dict[str, RetainedSubagentSession] = {}
_in_flight: set[str] = set()
_active_claim_generations: dict[str, int] = {}
_claim_generation_counters: dict[str, int] = {}
_cancelled_claims: set[tuple[str, int]] = set()
_invalidated: dict[str, str] = {}
_DEFAULT_MAX_RETAINED_SUBAGENT_BYTES = 16777216


def _copy_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy([item for item in list(history or []) if isinstance(item, dict)])


def _copy_record(record: RetainedSubagentSession) -> RetainedSubagentSession:
    return replace(
        record,
        conversation_history=_copy_history(record.conversation_history),
        effective_allowed_tool_names=frozenset(record.effective_allowed_tool_names),
        original_policy_identities=frozenset(record.original_policy_identities),
        tool_trace_metadata=tuple(record.tool_trace_metadata),
        files_written=tuple(record.files_written),
    )


def _serialized_history_bytes(history: list[dict[str, Any]]) -> int:
    try:
        payload = json.dumps(
            _copy_history(history),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Retained subagent transcript is not JSON serializable: {exc}"
        ) from exc
    return len(payload.encode("utf-8"))


def _retained_bytes_locked(*, exclude_agent_id: Optional[str] = None) -> int:
    return sum(
        _serialized_history_bytes(record.conversation_history)
        for agent_id, record in _records.items()
        if agent_id != exclude_agent_id
    )


def retained_subagent_transcript_bytes() -> int:
    """Return current serialized transcript bytes for process-local retention."""
    with _lock:
        return _retained_bytes_locked()


def _prune_expired(now: float) -> None:
    expired = [
        key
        for key, value in _records.items()
        if value.expires_at <= now and key not in _in_flight
    ]
    for key in expired:
        _records.pop(key, None)


def _oldest_removable(*, exclude_agent_id: Optional[str] = None):
    removable = [
        record
        for key, record in _records.items()
        if key not in _in_flight and key != exclude_agent_id
    ]
    return min(removable, key=lambda item: item.created_at) if removable else None


def _budget_drop_reason(agent_id: str) -> str:
    return (
        "Successful continuation exceeded the retained transcript byte budget; "
        f"the result is preserved, but {agent_id} is no longer resumable."
    )


def retain_subagent_session(
    record: RetainedSubagentSession,
    *,
    max_records: int = 64,
    max_total_bytes: int = _DEFAULT_MAX_RETAINED_SUBAGENT_BYTES,
) -> None:
    if not record.effective_allowed_tool_names:
        raise RuntimeError(
            "Retained subagent session has no original effective tool ceiling; "
            "refusing retention."
        )
    if not record.profile_id or not record.canonical_profile_home:
        raise RuntimeError(
            "Retained subagent session has no canonical profile binding; refusing retention."
        )
    if not record.original_policy_identities:
        raise RuntimeError(
            "Retained subagent session has no original exact authority; refusing retention."
        )

    copied = _copy_record(record)
    record_bytes = _serialized_history_bytes(copied.conversation_history)
    byte_budget = max(1, int(max_total_bytes or 1))
    if record_bytes > byte_budget:
        raise RuntimeError(
            "Retained subagent transcript exceeds retained transcript byte budget "
            f"({record_bytes} > {byte_budget} bytes)."
        )

    with _lock:
        capacity = max(1, int(max_records or 1))
        _prune_expired(time.time())
        replacing = record.agent_id in _records
        if replacing and record.agent_id in _in_flight:
            raise RuntimeError(
                f"Retained subagent continuation already in progress: {record.agent_id}"
            )

        while True:
            existing_count = len(_records) - (1 if replacing else 0)
            existing_bytes = _retained_bytes_locked(exclude_agent_id=record.agent_id)
            over_count = existing_count + 1 > capacity
            over_bytes = existing_bytes + record_bytes > byte_budget
            if not over_count and not over_bytes:
                break
            oldest = _oldest_removable(exclude_agent_id=record.agent_id)
            if oldest is None:
                if over_count:
                    raise RuntimeError(
                        f"Retained subagent session capacity reached ({capacity} records); "
                        "all retained sessions are in flight."
                    )
                raise RuntimeError(
                    "Retained subagent transcript byte budget reached; "
                    "all retained sessions are in flight."
                )
            _records.pop(oldest.agent_id, None)
            replacing = record.agent_id in _records
        _records[record.agent_id] = copied


def get_retained_subagent_session(agent_id: str) -> RetainedSubagentSession:
    with _lock:
        now = time.time()
        invalidation_reason = _invalidated.get(agent_id)
        if invalidation_reason is not None:
            raise RuntimeError(invalidation_reason)
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
        generation = _claim_generation_counters.get(agent_id, 0) + 1
        _claim_generation_counters[agent_id] = generation
        _active_claim_generations[agent_id] = generation
        return replace(record, claim_generation=generation)


def cancel_retained_subagent_claim(agent_id: str, claim_generation: int) -> bool:
    """Atomically cancel one exact in-flight claim generation."""
    with _lock:
        if _active_claim_generations.get(agent_id) != claim_generation:
            return False
        _cancelled_claims.add((agent_id, claim_generation))
        return True


def release_retained_subagent_session(agent_id: str) -> None:
    with _lock:
        generation = _active_claim_generations.pop(agent_id, None)
        if generation is not None:
            _cancelled_claims.discard((agent_id, generation))
        _in_flight.discard(agent_id)


def invalidate_retained_subagent_session(agent_id: str, reason: str) -> None:
    """Permanently poison one process-local handle with a stable failure reason."""
    with _lock:
        _records.pop(agent_id, None)
        _invalidated[agent_id] = str(reason)


def update_retained_history(
    agent_id: str,
    history: list[dict[str, Any]],
    *,
    max_total_bytes: int = _DEFAULT_MAX_RETAINED_SUBAGENT_BYTES,
    claim_generation: Optional[int] = None,
) -> Optional[str]:
    """Update a transcript, or invalidate it if the byte budget cannot fit it.

    Returns the stable invalidation reason when retention is dropped; the caller
    can still return the successful continuation result.
    """
    copied_history = _copy_history(history)
    history_bytes = _serialized_history_bytes(copied_history)
    byte_budget = max(1, int(max_total_bytes or 1))
    with _lock:
        if claim_generation is not None:
            active_generation = _active_claim_generations.get(agent_id)
            if (
                active_generation != claim_generation
                or (agent_id, claim_generation) in _cancelled_claims
            ):
                raise RetainedClaimCancelled(
                    "Interrupted retained subagent claim cannot commit late history: "
                    f"{agent_id} generation {claim_generation}"
                )
        record = get_retained_subagent_session(agent_id)
        reason = _budget_drop_reason(agent_id)
        if history_bytes > byte_budget:
            _records.pop(agent_id, None)
            _invalidated[agent_id] = reason
            return reason

        _prune_expired(time.time())
        while _retained_bytes_locked(exclude_agent_id=agent_id) + history_bytes > byte_budget:
            oldest = _oldest_removable(exclude_agent_id=agent_id)
            if oldest is None:
                _records.pop(agent_id, None)
                _invalidated[agent_id] = reason
                return reason
            _records.pop(oldest.agent_id, None)

        _records[agent_id] = replace(
            record,
            conversation_history=copied_history,
            updated_at=time.time(),
            status="completed",
        )
        return None


def clear_retained_subagent_sessions() -> None:
    with _lock:
        _records.clear()
        _in_flight.clear()
        _active_claim_generations.clear()
        _claim_generation_counters.clear()
        _cancelled_claims.clear()
        _invalidated.clear()
