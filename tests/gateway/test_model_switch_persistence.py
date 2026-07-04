"""Tests that gateway /model switch persists across messages.

The gateway /model command stores session overrides in
``_session_model_overrides``.  These must:

1. Be applied in ``run_sync()`` so the next agent uses the switched model.
2. Not be mistaken for fallback activation (which evicts the cached agent).
3. Survive across multiple messages until /reset clears them.

Tests exercise the real ``_apply_session_model_override()`` and
``_is_intentional_model_switch()`` methods on ``GatewayRunner``.
"""

from datetime import datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner():
    """Create a minimal GatewayRunner with stubbed internals."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._effective_model = None
    runner._effective_provider = None
    runner.session_store = MagicMock()
    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    return runner


# ---------------------------------------------------------------------------
# Tests: _apply_session_model_override
# ---------------------------------------------------------------------------


class TestApplySessionModelOverride:
    """Verify _apply_session_model_override replaces config defaults."""

    def test_override_replaces_all_fields(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4-turbo",
            "provider": "openrouter",
            "api_key": "or-key-123",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
        )

        assert model == "gpt-5.4-turbo"
        assert rt["provider"] == "openrouter"
        assert rt["api_key"] == "or-key-123"
        assert rt["base_url"] == "https://openrouter.ai/api/v1"
        assert rt["api_mode"] == "chat_completions"

    def test_no_override_returns_originals(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        orig_model = "anthropic/claude-sonnet-4"
        orig_rt = {"provider": "anthropic", "api_key": "key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"}

        model, rt = runner._apply_session_model_override(sk, orig_model, dict(orig_rt))

        assert model == orig_model
        assert rt == orig_rt

    def test_none_values_do_not_overwrite(self):
        """Override with None api_key/base_url should preserve config defaults."""
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": None,
            "base_url": None,
            "api_mode": "chat_completions",
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
        )

        assert model == "gpt-5.4"
        assert rt["provider"] == "openai"
        assert rt["api_key"] == "ant-key"  # preserved — None didn't overwrite
        assert rt["base_url"] == "https://api.anthropic.com"  # preserved
        assert rt["api_mode"] == "chat_completions"  # overwritten (not None)

    def test_empty_string_overwrites(self):
        """Empty string is not None — it should overwrite the config value."""
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "local-model",
            "provider": "custom",
            "api_key": "local-key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        _, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
        )

        assert rt["base_url"] == ""  # empty string overwrites

    def test_different_session_key_not_affected(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        other_sk = "other_session"

        runner._session_model_overrides[other_sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "url", "api_mode": "anthropic_messages"},
        )

        assert model == "anthropic/claude-sonnet-4"  # unchanged — wrong session key


# ---------------------------------------------------------------------------
# Tests: _is_intentional_model_switch
# ---------------------------------------------------------------------------


class TestIsIntentionalModelSwitch:
    """Verify fallback detection respects intentional /model overrides."""

    def test_matches_override(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        assert runner._is_intentional_model_switch(sk, "gpt-5.4") is True

    def test_no_override_returns_false(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        assert runner._is_intentional_model_switch(sk, "gpt-5.4") is False

    def test_different_model_returns_false(self):
        """Agent fell back to a different model than the override."""
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        assert runner._is_intentional_model_switch(sk, "gpt-5.4-mini") is False

    def test_wrong_session_key(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides["other_session"] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        assert runner._is_intentional_model_switch(sk, "gpt-5.4") is False


def _make_model_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
    )


def _openai_codex_switch_result():
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="gpt-5.5",
        target_provider="openai-codex",
        provider_changed=True,
        api_key="***",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        provider_label="OpenAI Codex",
        is_global=True,
    )


@pytest.mark.asyncio
async def test_status_uses_explicit_provider_after_model_switch(tmp_path, monkeypatch):
    """`/status` must report the provider selected by `/model --provider`.

    Regression: switching from a custom GPTCodex route to OpenAI Codex with the
    same model name updated only ``sessions.model``. With no live cached agent,
    `/status` then combined the new model with the stale ``billing_provider``
    and showed ``gpt-5.5 (custom)`` immediately after a successful switch.
    """
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.5",
                    "provider": "custom:gptcodex",
                    "base_url": "https://gptcodex.top/v1",
                },
                "providers": {},
            }
        ),
        encoding="utf-8",
    )

    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session("sess-1", source="discord", model="gpt-5.5")
    session_db.update_token_counts(
        "sess-1",
        billing_provider="custom",
        billing_base_url="https://gptcodex.top/v1",
        absolute=True,
    )

    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)
    runner.session_store._generate_session_key = MagicMock(
        side_effect=lambda src: build_session_key(src)
    )
    runner.session_store.get_or_create_session.return_value.session_key = session_key
    runner.session_store.get_or_create_session.return_value.session_id = "sess-1"
    runner._session_db = session_db

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: _openai_codex_switch_result(),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length",
        lambda *_args, **_kwargs: 272_000,
    )
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)

    switch_reply = await runner._handle_model_command(
        _make_model_event("/model gpt-5.5 --provider openai-codex")
    )
    assert switch_reply is not None
    assert "OpenAI Codex" in switch_reply

    row = session_db.get_session("sess-1")
    assert row is not None
    assert row["model"] == "gpt-5.5"
    assert row["billing_provider"] == "openai-codex"
    assert row["billing_base_url"] == "https://chatgpt.com/backend-api/codex"
    stored_config = json.loads(row["model_config"])
    assert stored_config["provider"] == "openai-codex"
    assert stored_config["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert stored_config["api_mode"] == "codex_responses"

    status = await runner._handle_status_command(_make_model_event("/status"))

    assert "gpt-5.5" in status
    assert "openai-codex" in status
    assert "(custom)" not in status
