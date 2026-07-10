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
