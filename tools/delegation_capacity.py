"""Process-wide and per-root-session live child-runner capacity accounting.

Every path that starts a child runner (foreground, background, batch,
continuation, nested delegation) must reserve through this module. Reservations
are all-or-nothing and count live child runners rather than delivery handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading


_lock = threading.RLock()
_active_runner_slots = 0
_active_runner_slots_by_session: dict[str, int] = {}


@dataclass
class RunnerReservation:
    count: int
    session_id: str
    _released_slots: set[int] = field(default_factory=set, init=False, repr=False)

    def _release_slot(self, slot_index: int) -> None:
        if not isinstance(slot_index, int) or isinstance(slot_index, bool):
            raise ValueError("runner reservation slot index must be an integer")
        if slot_index < 0 or slot_index >= self.count:
            raise ValueError("runner reservation slot index is out of range")
        with _lock:
            if slot_index in self._released_slots:
                return
            global _active_runner_slots
            _active_runner_slots = max(0, _active_runner_slots - 1)
            remaining = max(
                0,
                _active_runner_slots_by_session.get(self.session_id, 0) - 1,
            )
            if remaining:
                _active_runner_slots_by_session[self.session_id] = remaining
            else:
                _active_runner_slots_by_session.pop(self.session_id, None)
            self._released_slots.add(slot_index)

    def release_callback(self, slot_index: int):
        """Return an idempotent callback that releases one reserved runner."""

        # Validate eagerly so wiring bugs fail before any runner starts.
        if not isinstance(slot_index, int) or isinstance(slot_index, bool):
            raise ValueError("runner reservation slot index must be an integer")
        if slot_index < 0 or slot_index >= self.count:
            raise ValueError("runner reservation slot index is out of range")

        def _release() -> None:
            self._release_slot(slot_index)

        return _release

    def release(self) -> None:
        """Idempotently release every slot not already released individually."""

        for slot_index in range(self.count):
            self._release_slot(slot_index)


def try_reserve_runner_slots(
    count: int,
    *,
    global_limit: int,
    session_id: str,
    session_limit: int,
) -> RunnerReservation | None:
    """Atomically reserve *count* slots under both capacity ceilings.

    Returns ``None`` without mutating either counter when the process-global or
    root-session ceiling would be exceeded, or when no stable session identity
    is available.
    """

    numeric_limits = (count, global_limit, session_limit)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in numeric_limits
    ):
        raise ValueError(
            "runner reservation count and limits must be positive integers"
        )
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        return None

    global _active_runner_slots
    with _lock:
        session_active = _active_runner_slots_by_session.get(
            normalized_session_id, 0
        )
        if _active_runner_slots + count > global_limit:
            return None
        if session_active + count > session_limit:
            return None
        _active_runner_slots += count
        _active_runner_slots_by_session[normalized_session_id] = (
            session_active + count
        )
        return RunnerReservation(
            count=count,
            session_id=normalized_session_id,
        )


def active_runner_slots(*, session_id: str | None = None) -> int:
    with _lock:
        if session_id is None:
            return _active_runner_slots
        return _active_runner_slots_by_session.get(str(session_id), 0)


def _reset_for_tests() -> None:
    global _active_runner_slots
    with _lock:
        _active_runner_slots = 0
        _active_runner_slots_by_session.clear()
