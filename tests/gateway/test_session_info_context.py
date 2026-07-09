"""Regression tests for the gateway reset/new-session info banner."""

from gateway import run as gateway_run


def test_reset_session_info_uses_internal_context_window(monkeypatch):
    """The user-visible reset banner must not expose the runtime API window."""
    runner = object.__new__(gateway_run.GatewayRunner)
    config = {
        "model": {
            "default": "gpt-5.6-sol",
            "provider": "openai-codex",
            "context_length": 1_000_000,
        },
        "compression": {
            "internal_context_length": 272_000,
        },
    }

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda *_args, **_kwargs: "gpt-5.6-sol",
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-codex",
            "base_url": "",
            "api_key": "",
        },
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 1_000_000,
    )

    rendered = runner._format_session_info()

    assert "◆ Context: 272K tokens (config)" in rendered
    assert "◆ Context: 1.0M tokens" not in rendered


def test_live_gateway_metadata_uses_internal_compressor_window():
    """Footer/status metadata must use the live compressor's internal window."""
    compressor = type(
        "Compressor",
        (),
        {
            "context_length": 1_000_000,
            "compression_context_length": 272_000,
        },
    )()

    resolver = getattr(gateway_run, "_get_user_visible_context_length", None)

    assert callable(resolver)
    assert resolver(compressor) == 272_000
