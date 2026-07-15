from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


def _fingerprint(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _compressor(
    *,
    provider: str = "generic-provider",
    api_mode: str = "chat_completions",
) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="generic-model",
            provider=provider,
            api_mode=api_mode,
            threshold_percent=0.5,
            protect_first_n=1,
            protect_last_n=2,
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": False},
        )
    compressor.bind_summary_runtime_factory(
        lambda: SimpleNamespace(
            provider=provider,
            model="generic-model",
            api_mode=api_mode,
            base_url="https://provider.example/v1",
            fingerprint_prefix=_fingerprint,
        )
    )
    return compressor


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "tail"},
    ]


def test_append_cached_auto_compression_defers_without_reusable_prefix_checkpoint(monkeypatch):
    compressor = _compressor()
    messages = _messages()
    summary_called = False

    monkeypatch.setattr(compressor, "_find_tail_cut_by_tokens", lambda _messages, _start: 5)

    def _unexpected_summary(*_args, **_kwargs):
        nonlocal summary_called
        summary_called = True
        return "## Current Work\n- should not run"

    monkeypatch.setattr(compressor, "_generate_summary", _unexpected_summary)

    result = compressor.compress(
        messages,
        current_tokens=80_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert result is messages
    assert summary_called is False
    assert compressor._last_compress_deferred is True
    assert compressor._last_compression_audit_record["abort_reason"] == (
        "reusable_prefix_checkpoint_unavailable"
    )


@pytest.mark.parametrize(
    ("provider", "api_mode"),
    [
        ("provider-chat", "chat_completions"),
        ("provider-anthropic", "anthropic_messages"),
    ],
)
def test_append_cached_uses_latest_replayable_endpoint(
    monkeypatch,
    provider: str,
    api_mode: str,
):
    compressor = _compressor(provider=provider, api_mode=api_mode)
    messages = _messages()
    captured = {}
    compressor.record_reusable_prefix_checkpoint(
        source_message_count=4,
        prefix_fingerprint=_fingerprint(messages[:4]),
        provider_messages=messages[:4],
    )

    monkeypatch.setattr(compressor, "_find_tail_cut_by_tokens", lambda _messages, _start: 5)

    def _summary(_turns, *_args, **kwargs):
        captured["compress_end"] = kwargs["compress_end"]
        return "## Current Work\n- aligned"

    monkeypatch.setattr(compressor, "_generate_summary", _summary)

    result = compressor.compress(
        messages,
        current_tokens=80_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert compressor._last_compress_deferred is False
    assert captured["compress_end"] == 4
    assert "a2" in result[-4]["content"]
    assert result[-3:] == messages[5:]
    assert compressor._last_compression_audit_record is not None
    receipt = compressor._last_compression_audit_record[
        "reusable_prefix_checkpoint"
    ]
    assert receipt["selected"] is True
    assert receipt["source_message_count"] == 4


def test_auto_compression_moves_cut_forward_to_latest_successful_endpoint(monkeypatch):
    compressor = _compressor()
    messages = _messages()
    replay_messages = [message.copy() for message in messages[:6]]
    replay_messages[1]["content"] += "\n\nturn-local context"
    captured = {}

    compressor.record_reusable_prefix_checkpoint(
        source_message_count=6,
        prefix_fingerprint=_fingerprint(replay_messages),
        provider_messages=replay_messages,
    )
    monkeypatch.setattr(
        compressor,
        "_find_tail_cut_by_tokens",
        lambda _messages, _start: 4,
    )

    def _summary(_turns, *_args, **kwargs):
        captured["compress_end"] = kwargs["compress_end"]
        return "## Current Work\n- latest endpoint"

    monkeypatch.setattr(compressor, "_generate_summary", _summary)

    result = compressor.compress(
        messages,
        current_tokens=80_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert compressor._last_compress_deferred is False
    assert captured["compress_end"] == 6
    assert "a3" in result[-2]["content"]
    assert result[-1] == messages[-1]
    receipt = compressor._last_compression_audit_record[
        "reusable_prefix_checkpoint"
    ]
    assert receipt["selected"] is True
    assert receipt["desired_source_message_count"] == 4
    assert receipt["source_message_count"] == 6
    assert receipt["tail_messages_reduced_by"] == 2


def test_history_rewrite_invalidates_reusable_prefix_checkpoint(monkeypatch):
    compressor = _compressor()
    messages = _messages()
    compressor.record_reusable_prefix_checkpoint(
        source_message_count=4,
        prefix_fingerprint=_fingerprint(messages[:4]),
    )
    messages[2] = {"role": "assistant", "content": "rewritten cleanup result"}
    summary_called = False

    monkeypatch.setattr(compressor, "_find_tail_cut_by_tokens", lambda _messages, _start: 5)

    def _unexpected_summary(*_args, **_kwargs):
        nonlocal summary_called
        summary_called = True
        return "## Current Work\n- should not run"

    monkeypatch.setattr(compressor, "_generate_summary", _unexpected_summary)

    result = compressor.compress(
        messages,
        current_tokens=80_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert result is messages
    assert summary_called is False
    assert compressor._last_compress_deferred is True
    assert compressor._last_compression_audit_record is not None
    receipt = compressor._last_compression_audit_record[
        "reusable_prefix_checkpoint"
    ]
    assert receipt["selected"] is False
    assert receipt["candidate_count"] == 1


def test_checkpoint_loss_during_summary_preserves_original_transcript(monkeypatch):
    compressor = _compressor()
    messages = _messages()
    compressor.record_reusable_prefix_checkpoint(
        source_message_count=4,
        prefix_fingerprint=_fingerprint(messages[:4]),
    )

    monkeypatch.setattr(compressor, "_find_tail_cut_by_tokens", lambda _messages, _start: 5)

    def _deferred_summary(*_args, **_kwargs):
        compressor._last_compress_deferred = True
        compressor._last_summary_call_audit = {
            "mode": "append_cached",
            "fallback_reason": "reusable_prefix_checkpoint_unavailable",
        }
        return None

    monkeypatch.setattr(compressor, "_generate_summary", _deferred_summary)

    result = compressor.compress(
        messages,
        current_tokens=80_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert result is messages
    assert compressor._last_compress_deferred is True
    assert compressor._last_compression_audit_record is not None
    assert compressor._last_compression_audit_record["result"] == "deferred"
    assert compressor._last_compression_audit_record["abort_reason"] == (
        "reusable_prefix_checkpoint_unavailable"
    )


def test_only_latest_successful_endpoint_snapshot_is_retained():
    compressor = _compressor()
    messages = [
        {"role": "user", "content": f"message-{index}"}
        for index in range(201)
    ]
    for source_end in range(1, 202, 2):
        provider_messages = [message.copy() for message in messages[:source_end]]
        compressor.record_reusable_prefix_checkpoint(
            source_message_count=source_end,
            prefix_fingerprint=_fingerprint(provider_messages),
            provider_messages=provider_messages,
        )

    assert len(compressor._reusable_prefix_checkpoints) == 1
    checkpoint = compressor._latest_reusable_prefix_checkpoint
    assert checkpoint is not None
    assert checkpoint.source_message_count == 201
    assert len(checkpoint.provider_messages) == 201
