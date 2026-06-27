"""Unit tests for the persistent origin-overload self-heal decision.

The loop wiring is exercised end-to-end in
tests/run_agent/test_persistent_502_self_heal_e2e.py; this file pins the pure
decision function. See docs/design/2026-06-22-persistent-502-self-heal.md.
"""

from agent.conversation_loop import _should_self_heal_persistent_overload
from agent.error_classifier import FailoverReason


def _call(**override):
    base = dict(
        reason=FailoverReason.server_error,
        consecutive_overload_errors=5,
        threshold=5,
        compression_enabled=True,
        compression_attempts=0,
        max_compression_attempts=3,
    )
    base.update(override)
    return _should_self_heal_persistent_overload(**base)


def test_triggers_at_threshold():
    assert _call() is True


def test_overloaded_reason_also_triggers():
    # 503/529 classify as FailoverReason.overloaded — same self-heal.
    assert _call(reason=FailoverReason.overloaded) is True


def test_below_threshold_does_not_trigger():
    assert _call(consecutive_overload_errors=4) is False


def test_non_overload_reason_does_not_trigger():
    assert _call(reason=FailoverReason.rate_limit) is False
    assert _call(reason=FailoverReason.format_error) is False
    assert _call(reason=FailoverReason.billing) is False


def test_compression_disabled_does_not_trigger():
    assert _call(compression_enabled=False) is False


def test_compression_budget_exhausted_does_not_trigger():
    assert _call(compression_attempts=3) is False
    assert _call(compression_attempts=4) is False


def test_just_under_budget_triggers():
    assert _call(compression_attempts=2) is True
