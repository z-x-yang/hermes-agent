"""Tests for hermes_cli.context_switch_guard."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.context_switch_guard import merge_preflight_compression_warning
from hermes_cli.model_switch import ModelSwitchResult


def _result(*, model: str = "small-model") -> ModelSwitchResult:
    return ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider="openrouter",
        provider_changed=False,
        api_key="k",
        base_url="https://example.com/v1",
        api_mode="chat_completions",
        provider_label="openrouter",
        model_info={"context_length": 32_000},
    )


def _compressor(
    monkeypatch,
    *,
    context_length: int = 200_000,
    compression_context_length: int | None = None,
):
    from agent.context_compressor import ContextCompressor

    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length",
        lambda *a, **k: context_length,
    )
    return ContextCompressor(
        model="big-model",
        threshold_percent=0.5,
        protect_first_n=3,
        protect_last_n=20,
        quiet_mode=True,
        config_context_length=context_length,
        compression_context_length=compression_context_length,
    )


def test_no_warning_when_below_new_threshold(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    cc = _compressor(monkeypatch)
    cc.last_prompt_tokens = 10_000
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="",
        api_key="",
    )
    result = _result()
    merge_preflight_compression_warning(result, agent=agent)
    assert not result.warning_message


def test_switch_guard_no_usage_prefers_provider_visible_estimator():
    from hermes_cli.context_switch_guard import _estimate_tokens

    seen = {}

    def estimate(messages, *, system_prompt="", tools=None):
        seen.update(messages=messages, system_prompt=system_prompt, tools=tools)
        return 1_729

    cc = SimpleNamespace(
        protect_first_n=0,
        protect_last_n=0,
        estimate_provider_request_tokens=estimate,
        last_prompt_tokens=0,
    )
    agent = SimpleNamespace(
        context_compressor=cc,
        _cached_system_prompt="SYSTEM",
        tools=[{"type": "function", "function": {"name": "terminal"}}],
        session_prompt_tokens=0,
    )
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer", "reasoning": "R" * 10_000},
    ]

    assert _estimate_tokens(agent, messages) == 1_729
    assert seen == {
        "messages": messages,
        "system_prompt": "SYSTEM",
        "tools": agent.tools,
    }


def test_warns_when_estimate_exceeds_new_threshold(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: 90_000,
    )
    cc = _compressor(monkeypatch)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="",
        api_key="",
    )
    result = _result()
    merge_preflight_compression_warning(result, agent=agent)
    assert result.warning_message
    assert "preflight compression" in result.warning_message
    assert "shrinks" in result.warning_message


def test_warns_against_internal_window_when_runtime_is_large(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 1_000_000,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: 300_000,
    )
    cc = _compressor(
        monkeypatch,
        context_length=1_000_000,
        compression_context_length=272_000,
    )
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="",
        api_key="",
    )
    result = _result(model="gpt-5.5")
    merge_preflight_compression_warning(result, agent=agent)
    assert result.warning_message
    assert "preflight compression" in result.warning_message
    assert "272,000" in result.warning_message


def test_merge_appends_to_existing_warning(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: 90_000,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    cc = _compressor(monkeypatch)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )
    result = _result()
    result.warning_message = "expensive"
    merge_preflight_compression_warning(result, agent=agent)
    assert "expensive" in result.warning_message
    assert "preflight compression" in result.warning_message
