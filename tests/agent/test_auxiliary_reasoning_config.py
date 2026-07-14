from hermes_constants import parse_auxiliary_reasoning_config


def test_canonical_auxiliary_reasoning_effort_overrides_legacy_extra_body():
    parsed = parse_auxiliary_reasoning_config(
        {
            "reasoning_effort": "high",
            "extra_body": {"reasoning": {"enabled": True, "effort": "low"}},
        }
    )

    assert parsed == {"enabled": True, "effort": "high"}
