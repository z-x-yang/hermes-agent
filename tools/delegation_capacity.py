"""Process-wide reservation of live subagent runner slots."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading


_lock = threading.Lock()
_active_runner_slots = 0


@dataclass
class RunnerReservation:
    count: int
    _released: bool = False
    _release_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def release(self) -> None:
        global _active_runner_slots
        with self._release_lock:
            if self._released:
                return
            with _lock:
                _active_runner_slots -= self.count
                if _active_runner_slots < 0:  # defensive invariant
                    _active_runner_slots = 0
            self._released = True


def try_reserve_runner_slots(count: int, *, limit: int) -> RunnerReservation | None:
    """Atomically reserve a whole child batch, or reject it without partial start."""
    if isinstance(count, bool) or isinstance(limit, bool) or count <= 0 or limit <= 0:
        raise ValueError("runner reservation count and limit must be positive integers")
    global _active_runner_slots
    with _lock:
        if _active_runner_slots + count > limit:
            return None
        _active_runner_slots += count
    return RunnerReservation(count=count)


def active_runner_slots() -> int:
    with _lock:
        return _active_runner_slots


def _reset_for_tests() -> None:
    global _active_runner_slots
    with _lock:
        _active_runner_slots = 0
