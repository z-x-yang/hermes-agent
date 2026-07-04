"""Per-job cron reasoning effort resolution."""

from __future__ import annotations


def test_cron_reasoning_effort_job_override_wins_over_global():
    from cron.scheduler import _resolve_cron_reasoning_config

    cfg = {"agent": {"reasoning_effort": "low"}}

    assert _resolve_cron_reasoning_config({"reasoning_effort": "high"}, cfg) == {
        "enabled": True,
        "effort": "high",
    }


def test_cron_reasoning_effort_missing_job_field_uses_global():
    from cron.scheduler import _resolve_cron_reasoning_config

    cfg = {"agent": {"reasoning_effort": "medium"}}

    assert _resolve_cron_reasoning_config({}, cfg) == {
        "enabled": True,
        "effort": "medium",
    }


def test_cron_reasoning_effort_none_disables_reasoning():
    from cron.scheduler import _resolve_cron_reasoning_config

    cfg = {"agent": {"reasoning_effort": "high"}}

    assert _resolve_cron_reasoning_config({"reasoning_effort": "none"}, cfg) == {
        "enabled": False,
    }
