"""Cron reasoning effort resolution."""

from __future__ import annotations

import pytest


def test_cron_reasoning_effort_job_override_wins_over_cron_default():
    from cron.scheduler import _resolve_cron_reasoning_config

    cfg = {
        "cron": {"reasoning_effort": "medium"},
        "agent": {"reasoning_effort": "low"},
    }

    assert _resolve_cron_reasoning_config({"reasoning_effort": "high"}, cfg) == {
        "enabled": True,
        "effort": "high",
    }


def test_cron_reasoning_effort_accepts_max_override():
    from cron.scheduler import _resolve_cron_reasoning_config

    assert _resolve_cron_reasoning_config({"reasoning_effort": "max"}, {}) == {
        "enabled": True,
        "effort": "max",
    }


def test_cron_reasoning_effort_cron_default_wins_over_agent_global():
    from cron.scheduler import _resolve_cron_reasoning_config

    cfg = {
        "cron": {"reasoning_effort": "high"},
        "agent": {"reasoning_effort": "low"},
    }

    assert _resolve_cron_reasoning_config({}, cfg) == {
        "enabled": True,
        "effort": "high",
    }


def test_cron_reasoning_effort_missing_cron_default_uses_legacy_agent_global():
    from cron.scheduler import _resolve_cron_reasoning_config

    cfg = {"agent": {"reasoning_effort": "medium"}}

    assert _resolve_cron_reasoning_config({}, cfg) == {
        "enabled": True,
        "effort": "medium",
    }


def test_cron_reasoning_effort_null_cron_default_uses_legacy_agent_global():
    from cron.scheduler import _resolve_cron_reasoning_config

    cfg = {"cron": {"reasoning_effort": None}, "agent": {"reasoning_effort": "low"}}

    assert _resolve_cron_reasoning_config({}, cfg) == {
        "enabled": True,
        "effort": "low",
    }


def test_cron_reasoning_effort_none_disables_reasoning():
    from cron.scheduler import _resolve_cron_reasoning_config

    cfg = {"cron": {"reasoning_effort": "high"}}

    assert _resolve_cron_reasoning_config({"reasoning_effort": "none"}, cfg) == {
        "enabled": False,
    }


def test_cron_reasoning_effort_invalid_cron_default_fails_fast():
    from cron.scheduler import _resolve_cron_reasoning_config

    cfg = {"cron": {"reasoning_effort": "turbo"}}

    with pytest.raises(ValueError, match="Invalid cron reasoning_effort"):
        _resolve_cron_reasoning_config({}, cfg)


def test_default_config_exposes_cron_reasoning_effort():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["cron"]["reasoning_effort"] == ""
