from hermes_cli.oneshot import _oneshot_clarify_callback


def test_oneshot_accepts_context_and_renders_labels():
    # 3-arg contract: must not TypeError; renders choice LABELS, not raw dicts.
    out = _oneshot_clarify_callback(
        "Which target?",
        [{"label": "staging", "description": "test cluster"},
         {"label": "prod", "description": "production"}],
        "Two clusters exist.",
    )
    assert "staging" in out and "prod" in out
    assert "description" not in out  # raw dict not leaked
    assert "{'label'" not in out


def test_oneshot_open_ended_accepts_context():
    out = _oneshot_clarify_callback("Anything?", None, "some background")
    assert isinstance(out, str) and out
