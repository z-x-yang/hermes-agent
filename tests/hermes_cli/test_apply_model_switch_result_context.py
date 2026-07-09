"""Regression test for the `/model` picker confirmation display.

Bug class: after choosing a model from the interactive `/model` picker,
``HermesCLI._apply_model_switch_result()`` must not print raw
``ModelInfo.context_window`` when the provider-aware resolver has a more
accurate value for the active route. Both display paths go through
``resolve_display_context_length()``.
"""
from __future__ import annotations

from unittest.mock import patch

from hermes_cli.model_switch import ModelSwitchResult


class _FakeModelInfo:
    context_window = 1_050_000
    max_output = 0

    def has_cost_data(self):
        return False

    def format_capabilities(self):
        return ""


class _StubCLI:
    """Minimum attrs ``_apply_model_switch_result`` reads on ``self``."""
    agent = None
    model = ""
    provider = ""
    requested_provider = ""
    api_key = ""
    _explicit_api_key = ""
    base_url = ""
    _explicit_base_url = ""
    api_mode = ""
    _pending_model_switch_note = ""


def _run_display(monkeypatch, result):
    import cli as cli_mod

    captured: list[str] = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda s, *a, **k: captured.append(str(s)))
    # Avoid writing to ~/.hermes/config.yaml during the test.
    monkeypatch.setattr(cli_mod, "save_config_value", lambda *a, **k: None)
    cli_mod.HermesCLI._apply_model_switch_result(_StubCLI(), result, False)
    return captured


def test_picker_path_uses_provider_aware_context_on_codex(monkeypatch):
    """``_apply_model_switch_result`` must prefer the provider-aware resolver
    (currently 1M on Codex) over the raw models.dev value.
    """
    result = ModelSwitchResult(
        success=True,
        new_model="gpt-5.5",
        target_provider="openai-codex",
        provider_changed=True,
        api_key="",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        warning_message="",
        provider_label="ChatGPT Codex",
        resolved_via_alias=False,
        capabilities=None,
        model_info=_FakeModelInfo(),  # models.dev says 1.05M
        is_global=False,
    )
    with patch(
        "agent.model_metadata.get_model_context_length",
        return_value=1_000_000,
    ):
        lines = _run_display(monkeypatch, result)

    ctx_line = next((l for l in lines if "Context:" in l), "")
    assert "1,000,000" in ctx_line, ctx_line
    assert "1,050,000" not in ctx_line, (
        f"picker-path display leaked models.dev's value for Codex: {ctx_line!r}"
    )


def test_picker_path_shows_vendor_value_when_no_provider_cap(monkeypatch):
    """On providers with no enforced cap (e.g. OpenRouter), the picker path
    should surface the real 1.05M context for gpt-5.5 — resolver and models.dev
    agree here.
    """
    result = ModelSwitchResult(
        success=True,
        new_model="openai/gpt-5.5",
        target_provider="openrouter",
        provider_changed=True,
        api_key="",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        warning_message="",
        provider_label="OpenRouter",
        resolved_via_alias=False,
        capabilities=None,
        model_info=_FakeModelInfo(),
        is_global=False,
    )
    with patch(
        "agent.model_metadata.get_model_context_length",
        return_value=1_050_000,
    ):
        lines = _run_display(monkeypatch, result)

    ctx_line = next((l for l in lines if "Context:" in l), "")
    assert "1,050,000" in ctx_line, (
        f"OpenRouter gpt-5.5 should show 1.05M context, got: {ctx_line!r}"
    )


def test_picker_path_falls_back_to_model_info_when_resolver_empty(monkeypatch):
    """If ``get_model_context_length`` returns nothing (rare — truly unknown
    endpoint), the display still surfaces ``ModelInfo.context_window`` so the
    user sees *something* rather than a silent blank.
    """
    result = ModelSwitchResult(
        success=True,
        new_model="some-model",
        target_provider="some-provider",
        provider_changed=True,
        api_key="",
        base_url="",
        api_mode="chat_completions",
        warning_message="",
        provider_label="Some Provider",
        resolved_via_alias=False,
        capabilities=None,
        model_info=_FakeModelInfo(),  # context_window = 1_050_000
        is_global=False,
    )
    with patch(
        "agent.model_metadata.get_model_context_length",
        return_value=None,
    ):
        lines = _run_display(monkeypatch, result)

    ctx_line = next((l for l in lines if "Context:" in l), "")
    assert "1,050,000" in ctx_line, (
        f"resolver-empty path should fall back to ModelInfo, got: {ctx_line!r}"
    )
