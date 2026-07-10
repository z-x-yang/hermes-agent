import pytest

from tools.subagent_profiles import (
    SUPPORTED_SUBAGENT_TYPES,
    get_subagent_profile,
    resolve_profile_config,
)


def test_only_claude_aligned_builtin_types_are_exposed():
    assert SUPPORTED_SUBAGENT_TYPES == (
        "Explore",
        "Plan",
        "general-purpose",
    )


@pytest.mark.parametrize("name", SUPPORTED_SUBAGENT_TYPES)
def test_builtin_profile_round_trip(name):
    profile = get_subagent_profile(name)
    assert profile.name == name
    assert profile.model == "inherit"
    assert profile.can_external_side_effects is False


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError, match="Unknown subagent_type"):
        get_subagent_profile("review-readonly")


def test_per_agent_config_overrides_global_without_exposing_to_model():
    cfg = {
        "model": "global-model",
        "provider": "openrouter",
        "foreground_wait_timeout_seconds": 1200,
        "child_run_timeout_seconds": 2400,
        "max_foreground_wait_timeout_seconds": 7200,
        "agents": {
            "Explore": {
                "model": "cheap-model",
                "foreground_wait_timeout_seconds": 900,
                "child_run_timeout_seconds": 1800,
            }
        },
    }
    resolved = resolve_profile_config("Explore", cfg)
    assert resolved.model == "cheap-model"
    assert resolved.provider == "openrouter"
    assert resolved.foreground_wait_timeout_seconds == 900
    assert resolved.child_run_timeout_seconds == 1800


def test_global_timeouts_override_profile_defaults():
    resolved = resolve_profile_config(
        "Explore",
        {
            "foreground_wait_timeout_seconds": 1234,
            "child_run_timeout_seconds": 2345,
        },
    )
    assert resolved.foreground_wait_timeout_seconds == 1234
    assert resolved.child_run_timeout_seconds == 2345


def test_foreground_wait_timeout_is_clamped_by_positive_maximum():
    resolved = resolve_profile_config(
        "Plan",
        {
            "foreground_wait_timeout_seconds": 9000,
            "max_foreground_wait_timeout_seconds": 4000,
            "agents": {"Plan": {"foreground_wait_timeout_seconds": 8000}},
        },
    )
    assert resolved.foreground_wait_timeout_seconds == 4000
