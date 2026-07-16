from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.chat_completion_helpers import build_provider_request_snapshot
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
            fingerprint_provider_request=lambda request: _fingerprint(
                request.get("messages") or request.get("input") or []
            ),
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


def _snapshot(
    provider_messages: list[dict],
    *,
    source_message_count: int,
    fingerprint: str | None = None,
    calibration_fingerprint: str = "route",
) -> SimpleNamespace:
    return SimpleNamespace(
        source_message_count=source_message_count,
        cache_fingerprint=fingerprint or _fingerprint(provider_messages),
        semantic_tokens=max(1, source_message_count * 1_000),
        calibration_fingerprint=calibration_fingerprint,
        provider_request={"messages": [message.copy() for message in provider_messages]},
    )


def _record(
    compressor: ContextCompressor,
    provider_messages: list[dict],
    *,
    source_message_count: int,
    provider_input_tokens: int,
    fingerprint: str | None = None,
    calibration_fingerprint: str = "route",
) -> None:
    compressor.record_reusable_prefix_checkpoint(
        request_snapshot=_snapshot(
            provider_messages,
            source_message_count=source_message_count,
            fingerprint=fingerprint,
            calibration_fingerprint=calibration_fingerprint,
        ),
        provider_input_tokens=provider_input_tokens,
    )


def test_final_request_snapshot_drives_same_lineage_actual_plus_delta():
    compressor = _compressor()
    lineage = (
        "generic-provider",
        "generic-model",
        "chat_completions",
        "https://provider.example/v1",
        "credential-one",
    )
    accepted = build_provider_request_snapshot(
        {"model": "generic-model", "messages": [{"role": "user", "content": "short"}]},
        source_message_count=1,
        lineage=lineage,
    )
    current = build_provider_request_snapshot(
        {
            "model": "generic-model",
            "messages": [{"role": "user", "content": "short plus a larger semantic delta"}],
        },
        source_message_count=1,
        lineage=lineage,
    )
    switched_credential = build_provider_request_snapshot(
        current.provider_request,
        source_message_count=1,
        lineage=(*lineage[:-1], "credential-two"),
    )

    compressor.record_reusable_prefix_checkpoint(
        request_snapshot=accepted,
        provider_input_tokens=10_000,
    )

    expected = 10_000 + current.semantic_tokens - accepted.semantic_tokens
    assert compressor.estimate_provider_request_tokens(request_snapshot=current) == expected
    assert (
        compressor.estimate_provider_request_tokens(request_snapshot=switched_credential)
        == switched_credential.semantic_tokens
    )


def test_append_cached_auto_compression_uses_transcript_fallback_without_anchor(monkeypatch):
    compressor = _compressor()
    messages = _messages()
    captured = {}

    monkeypatch.setattr(compressor, "_find_tail_cut_by_tokens", lambda _messages, _start: 5)

    def _summary(_turns, *_args, **kwargs):
        captured["compress_end"] = kwargs["compress_end"]
        return "## Current Work\n- transcript fallback"

    monkeypatch.setattr(compressor, "_generate_summary", _summary)

    result = compressor.compress(
        messages,
        current_tokens=80_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert result is not messages
    assert captured["compress_end"] == 5
    assert compressor._last_compress_deferred is False
    assert compressor._last_compression_audit_record["result"] == "success"
    receipt = compressor._last_compression_audit_record[
        "reusable_prefix_checkpoint"
    ]
    assert receipt["selected"] is False
    assert receipt["reason"] == "transcript_reconstructed_fallback"


@pytest.mark.parametrize("trigger_reason", [
    "token_threshold",
    "final_provider_request_threshold",
])
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
    trigger_reason: str,
):
    compressor = _compressor(provider=provider, api_mode=api_mode)
    compressor.threshold_tokens = 100_000
    compressor.tail_token_budget = 20_000
    messages = _messages()
    captured = {}
    _record(
        compressor,
        messages[:3],
        source_message_count=3,
        provider_input_tokens=70_000,
    )
    _record(
        compressor,
        messages[:4],
        source_message_count=4,
        provider_input_tokens=84_000,
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
        trigger_reason=trigger_reason,
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
    assert compressor._frozen_reusable_prefix_checkpoint is None


@pytest.mark.parametrize("event", ["reset", "end"])
def test_session_lifecycle_deletes_persisted_frozen_anchor(tmp_path, event):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session(f"session-{event}", "discord")
    compressor = _compressor()
    compressor.threshold_tokens = 100_000
    compressor.tail_token_budget = 20_000
    compressor.bind_session_state(db, session_id)
    messages = _messages()

    for source_end, provider_tokens in ((3, 70_000), (4, 84_000)):
        provider_messages = [message.copy() for message in messages[:source_end]]
        _record(
            compressor,
            provider_messages,
            source_message_count=source_end,
            provider_input_tokens=provider_tokens,
        )

    assert db.get_compression_replay_anchor(session_id) is not None
    if event == "reset":
        compressor.on_session_reset()
    else:
        compressor.on_session_end(session_id, messages)

    assert db.get_compression_replay_anchor(session_id) is None
    assert compressor._frozen_reusable_prefix_checkpoint is None


def test_auto_compression_moves_cut_forward_to_latest_successful_endpoint(monkeypatch):
    compressor = _compressor()
    messages = _messages()
    replay_messages = [message.copy() for message in messages[:6]]
    replay_messages[1]["content"] += "\n\nturn-local context"
    captured = {}

    compressor.threshold_tokens = 100_000
    compressor.tail_token_budget = 20_000
    _record(
        compressor,
        messages[:4],
        source_message_count=4,
        provider_input_tokens=70_000,
    )
    _record(
        compressor,
        replay_messages,
        source_message_count=6,
        provider_input_tokens=84_000,
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


def test_unreplayable_checkpoint_uses_transcript_fallback(monkeypatch):
    compressor = _compressor()
    messages = _messages()
    messages[2] = {"role": "assistant", "content": "rewritten cleanup result"}
    summary_called = False

    monkeypatch.setattr(compressor, "_find_tail_cut_by_tokens", lambda _messages, _start: 5)

    def _summary(*_args, **_kwargs):
        nonlocal summary_called
        summary_called = True
        return "## Current Work\n- transcript fallback"

    monkeypatch.setattr(compressor, "_generate_summary", _summary)

    result = compressor.compress(
        messages,
        current_tokens=80_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert result is not messages
    assert summary_called is True
    assert compressor._last_compress_deferred is False
    receipt = compressor._last_compression_audit_record[
        "reusable_prefix_checkpoint"
    ]
    assert receipt["selected"] is False
    assert receipt["reason"] == "transcript_reconstructed_fallback"


def test_runtime_invalid_anchor_falls_back_to_transcript_without_defer(monkeypatch):
    compressor = _compressor()
    compressor.threshold_tokens = 100_000
    compressor.tail_token_budget = 20_000
    messages = _messages()
    provider_messages = [message.copy() for message in messages[:4]]
    _record(
        compressor,
        messages[:3],
        source_message_count=3,
        provider_input_tokens=70_000,
    )
    _record(
        compressor,
        provider_messages,
        source_message_count=4,
        provider_input_tokens=84_000,
    )
    compressor.bind_summary_runtime_factory(
        lambda: SimpleNamespace(
            fingerprint_provider_request=lambda _request: "different-lineage",
        )
    )
    captured = {}

    monkeypatch.setattr(compressor, "_find_tail_cut_by_tokens", lambda _messages, _start: 5)

    def _summary(_turns, *_args, **kwargs):
        captured["compress_end"] = kwargs["compress_end"]
        return "## Current Work\n- transcript fallback"

    monkeypatch.setattr(compressor, "_generate_summary", _summary)

    result = compressor.compress(
        messages,
        current_tokens=80_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert result is not messages
    assert captured["compress_end"] == 5
    assert compressor._last_compress_deferred is False
    assert compressor._last_compression_audit_record["result"] == "success"
    receipt = compressor._last_compression_audit_record[
        "reusable_prefix_checkpoint"
    ]
    assert receipt["selected"] is False
    assert receipt["reason"] == "transcript_reconstructed_fallback"


def test_freezes_endpoint_nearest_tail_target_and_later_success_does_not_overwrite():
    compressor = _compressor()
    compressor.threshold_tokens = 100_000
    compressor.tail_token_budget = 20_000
    messages = [
        {"role": "user", "content": f"message-{index}"}
        for index in range(8)
    ]

    def record(source_end: int, provider_tokens: int) -> None:
        provider_messages = [message.copy() for message in messages[:source_end]]
        _record(
            compressor,
            provider_messages,
            source_message_count=source_end,
            provider_input_tokens=provider_tokens,
        )

    record(3, 70_000)
    assert compressor._frozen_reusable_prefix_checkpoint is None

    record(4, 84_000)
    anchor = compressor._frozen_reusable_prefix_checkpoint
    assert anchor is not None
    assert anchor.source_message_count == 4
    assert anchor.provider_input_tokens == 84_000

    record(6, 95_000)
    anchor = compressor._frozen_reusable_prefix_checkpoint
    assert anchor is not None
    assert anchor.source_message_count == 4
    assert len(compressor._reusable_prefix_checkpoints) == 1


def test_candidate_lineage_change_restarts_anchor_cycle():
    compressor = _compressor()
    compressor.threshold_tokens = 100_000
    compressor.tail_token_budget = 20_000
    messages = _messages()

    _record(
        compressor,
        messages[:3],
        source_message_count=3,
        provider_input_tokens=70_000,
        calibration_fingerprint="route-a",
    )
    assert compressor._replay_anchor_candidate is not None
    assert compressor._replay_anchor_candidate.calibration_fingerprint == "route-a"

    _record(
        compressor,
        messages[:4],
        source_message_count=4,
        provider_input_tokens=72_000,
        calibration_fingerprint="route-b",
    )

    assert compressor._frozen_reusable_prefix_checkpoint is None
    candidate = compressor._replay_anchor_candidate
    assert candidate is not None
    assert candidate.source_message_count == 4
    assert candidate.calibration_fingerprint == "route-b"
    assert compressor._reusable_prefix_checkpoints == [candidate]


def test_frozen_lineage_change_deletes_durable_anchor_and_restarts_cycle(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session("session-lineage", "discord")
    compressor = _compressor()
    compressor.threshold_tokens = 100_000
    compressor.tail_token_budget = 20_000
    compressor.bind_session_state(db, session_id)
    messages = _messages()

    _record(
        compressor,
        messages[:3],
        source_message_count=3,
        provider_input_tokens=70_000,
        calibration_fingerprint="route-a",
    )
    _record(
        compressor,
        messages[:4],
        source_message_count=4,
        provider_input_tokens=84_000,
        calibration_fingerprint="route-a",
    )
    assert compressor._frozen_reusable_prefix_checkpoint is not None
    assert db.get_compression_replay_anchor(session_id) is not None

    _record(
        compressor,
        messages[:5],
        source_message_count=5,
        provider_input_tokens=73_000,
        calibration_fingerprint="route-b",
    )

    assert db.get_compression_replay_anchor(session_id) is None
    assert compressor._frozen_reusable_prefix_checkpoint is None
    candidate = compressor._replay_anchor_candidate
    assert candidate is not None
    assert candidate.source_message_count == 5
    assert candidate.calibration_fingerprint == "route-b"


def test_frozen_anchor_reloads_when_compressor_is_recreated(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session("session-1", "discord")
    messages = [
        {"role": "user", "content": f"message-{index}"}
        for index in range(8)
    ]
    before = _compressor()
    before.threshold_tokens = 100_000
    before.tail_token_budget = 20_000
    before.bind_session_state(db, session_id)

    for source_end, provider_tokens in ((3, 70_000), (4, 84_000)):
        provider_messages = [message.copy() for message in messages[:source_end]]
        _record(
            before,
            provider_messages,
            source_message_count=source_end,
            provider_input_tokens=provider_tokens,
        )

    after = _compressor()
    after.threshold_tokens = 100_000
    after.tail_token_budget = 20_000
    after.bind_session_state(db, session_id)

    anchor = after._frozen_reusable_prefix_checkpoint
    assert anchor is not None
    assert anchor.source_message_count == 4
    assert anchor.provider_input_tokens == 84_000
    assert anchor.provider_request == {"messages": messages[:4]}
    assert after._select_reusable_prefix_checkpoint(messages, 5) == anchor


def test_first_success_above_tail_target_does_not_become_latest_fallback_anchor():
    compressor = _compressor()
    compressor.threshold_tokens = 100_000
    compressor.tail_token_budget = 20_000
    provider_messages = [{"role": "user", "content": "already over target"}]

    _record(
        compressor,
        provider_messages,
        source_message_count=1,
        provider_input_tokens=90_000,
    )

    assert compressor._frozen_reusable_prefix_checkpoint is None
    assert compressor._select_reusable_prefix_checkpoint(provider_messages, 1) is None


def test_success_without_provider_usage_does_not_freeze_latest_endpoint():
    compressor = _compressor()
    provider_messages = [{"role": "user", "content": "usage unavailable"}]

    _record(
        compressor,
        provider_messages,
        source_message_count=1,
        provider_input_tokens=0,
    )

    assert compressor._frozen_reusable_prefix_checkpoint is None
    assert compressor._replay_anchor_candidate is None
    assert compressor._latest_reusable_prefix_checkpoint is None
    assert compressor._reusable_prefix_checkpoints == []


def test_success_without_provider_usage_keeps_accepted_baseline_unchanged():
    compressor = _compressor()
    compressor.threshold_tokens = 100_000
    compressor.tail_token_budget = 20_000
    accepted_messages = [{"role": "user", "content": "accepted"}]
    _record(
        compressor,
        accepted_messages,
        source_message_count=1,
        provider_input_tokens=70_000,
        calibration_fingerprint="route-a",
    )
    baseline = (
        compressor._last_successful_request_actual_tokens,
        compressor._last_successful_request_semantic_tokens,
        compressor._last_successful_request_fingerprint,
    )

    _record(
        compressor,
        accepted_messages + [{"role": "assistant", "content": "unmetered"}],
        source_message_count=2,
        provider_input_tokens=0,
        calibration_fingerprint="route-b",
    )

    assert (
        compressor._last_successful_request_actual_tokens,
        compressor._last_successful_request_semantic_tokens,
        compressor._last_successful_request_fingerprint,
    ) == baseline
    assert compressor._replay_anchor_candidate is not None
    assert compressor._replay_anchor_candidate.calibration_fingerprint == "route-a"
