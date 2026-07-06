from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    CheapToolResultCleanupConfig,
    ContextCompressor,
)


def _compressor(**kwargs) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.80,
            protect_first_n=1,
            protect_last_n=2,
            summary_target_ratio=0.10,
            quiet_mode=True,
            **kwargs,
        )


def test_cheap_cleanup_config_defaults_disabled():
    c = _compressor()

    cfg = c.cheap_tool_result_cleanup

    assert cfg.enabled is False
    assert cfg.keep_recent == 5
    assert cfg.min_tokens_saved == 20_000
    assert cfg.replacement_mode == "persisted_handle_or_sentinel"
    assert cfg.skip_llm_summary_when_below_threshold is True


def test_cheap_cleanup_config_can_be_injected():
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=3,
        min_tokens_saved=1234,
        replacement_mode="persisted_handle_or_sentinel",
        skip_llm_summary_when_below_threshold=False,
    )

    c = _compressor(cheap_tool_result_cleanup=cfg)

    assert c.cheap_tool_result_cleanup == cfg


def _assistant_call(call_id: str, name: str = "terminal") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _tool(call_id: str, content: str, *, row_id: int | None = None) -> dict:
    msg: dict[str, object] = {"role": "tool", "tool_call_id": call_id, "content": content}
    if row_id is not None:
        msg["id"] = row_id
    return msg


def test_cleanup_counts_tail_tools_against_keep_recent():
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=5, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_id="sess-1")
    big = "x" * 4000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-0"),
        _tool("old-0", big, row_id=10),
        _assistant_call("old-1"),
        _tool("old-1", big, row_id=11),
        _assistant_call("old-2"),
        _tool("old-2", big, row_id=12),
        _assistant_call("old-3"),
        _tool("old-3", big, row_id=13),
        _assistant_call("tail-1"),
        _tool("tail-1", big, row_id=21),
        _assistant_call("tail-2"),
        _tool("tail-2", big, row_id=22),
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=10)

    assert result.applied is True
    assert result.messages[3]["content"] == "[Old tool result content cleared]"
    assert result.audit["replacement_counts"]["sentinel"] == 1
    assert result.audit["sentinel_fallback_reasons"]["unverified_row_id"] == 1
    assert result.messages[5]["content"] == big
    assert result.messages[7]["content"] == big
    assert result.messages[9]["content"] == big
    assert result.messages[11]["content"] == big
    assert result.messages[13]["content"] == big
    assert result.audit["tail_tool_result_count"] == 2
    assert result.audit["extra_pre_tail_keep_count"] == 3
    assert result.audit["cleared_count"] == 1
    assert result.audit["protected_tail_cleared_count"] == 0


def test_cleanup_uses_sentinel_when_row_id_missing():
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_id="sess-1")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "x" * 4000),
        {"role": "user", "content": "tail"},
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    assert result.applied is True
    assert result.messages[3]["content"] == "[Old tool result content cleared]"
    assert result.audit["replacement_counts"]["sentinel"] == 1
    assert result.audit["sentinel_fallback_reasons"]["missing_row_id"] == 1


def test_persisted_handle_points_to_existing_message_row(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session("cheap-cleanup-test", "cli")
    raw_tool_output = "recover me " + ("x" * 4000)
    raw_messages = [
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", raw_tool_output),
    ]
    db.replace_messages(session_id, raw_messages)
    loaded = db.get_messages(session_id)
    tool_row = next(msg for msg in loaded if msg.get("role") == "tool")

    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_db=db, session_id=session_id)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        tool_row,
        {"role": "user", "content": "tail user"},
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    cleaned_content = result.messages[3]["content"]
    assert f"hermes://session/{session_id}/message/{tool_row['id']}" in cleaned_content
    assert "session_search(" in cleaned_content
    assert raw_tool_output not in cleaned_content
    archived_window = db.get_messages_around(session_id, int(tool_row["id"]), window=1)
    assert any(msg.get("content") == raw_tool_output for msg in archived_window["window"])


def test_cleanup_uses_sentinel_when_row_id_is_not_in_bound_session_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session("cheap-cleanup-test", "cli")

    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_db=db, session_id=session_id)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "x" * 4000, row_id=999_999),
        {"role": "user", "content": "tail"},
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    assert result.applied is True
    assert result.messages[3]["content"] == "[Old tool result content cleared]"
    assert "hermes://session/" not in result.messages[3]["content"]
    assert result.audit["replacement_counts"]["sentinel"] == 1
    assert result.audit["sentinel_fallback_reasons"]["unverified_row_id"] == 1


def test_cleanup_below_min_tokens_saved_does_not_mutate():
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=20_000)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_id="sess-1")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "small", row_id=11),
        {"role": "user", "content": "tail"},
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    assert result.applied is False
    assert result.messages is messages
    assert result.audit["result"] == "below_min_tokens_saved"


def test_audit_records_cleanup_block_for_disabled_feature():
    c = _compressor()
    c.bind_session_state(session_db=None, session_id="sess-1")
    record = c._build_compression_audit_record(
        result="skipped",
        entrypoint="auto",
        input_messages=2,
        output_messages=2,
        before_estimate=10,
        after_estimate=10,
        before_messages=[{"role": "user", "content": "hi"}],
        after_messages=[{"role": "user", "content": "hi"}],
    )

    block = record["cheap_tool_result_cleanup"]
    assert block["enabled"] is False
    assert block["applied"] is False
    assert block["result"] in {"not_attempted", "disabled"}
    assert block["scope"] == "summary_window_before_protected_tail"
    assert block["tail_tool_result_count"] == 0
    assert block["tail_tool_count"] == 0
    assert block["extra_pre_tail_keep_count"] == 0
    assert block["candidate_count"] == 0
    assert block["clear_candidate_count"] == 0
    assert block["cleared_count"] == 0
    assert block["tokens_saved_estimate"] == 0
    assert block["tokens_saved"] == 0
    assert block["replacement_counts"] == {"persisted_handle": 0, "sentinel": 0}
    assert block["sentinel_fallback_reasons"] == {}
    assert block["protected_tail_cleared_count"] == 0
    assert block["summary_source_view"] == "not_applicable"
    assert block["raw_tool_results_restored_for_summary"] is False
    assert block["llm_summary_skipped_after_cleanup"] is False
    assert block["llm_summary_ran_on_cleaned_view"] is False


def test_summary_source_uses_cleaned_tool_result_when_cleanup_applies(monkeypatch):
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=0,
        min_tokens_saved=1,
        skip_llm_summary_when_below_threshold=False,
    )
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.tail_token_budget = 1
    c.bind_session_state(session_db=None, session_id="sess-1")
    captured = {}

    def fake_generate(turns, focus_topic=None):
        captured["turns"] = turns
        captured["serialized"] = c._serialize_for_summary(turns)
        return "## Current Work\n- compressed\n\n" + captured["serialized"]

    monkeypatch.setattr(c, "_generate_summary", fake_generate)
    big = "RAW_TOOL_OUTPUT_SHOULD_NOT_REACH_SUMMARY " * 300
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", big, row_id=11),
        {"role": "assistant", "content": "bridge before protected tail"},
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    out = c.compress(messages, current_tokens=90_000, force=False, trigger_reason="token_threshold")

    serialized_turns = str(captured["turns"])
    assert "RAW_TOOL_OUTPUT_SHOULD_NOT_REACH_SUMMARY" not in serialized_turns
    assert "RAW_TOOL_OUTPUT_SHOULD_NOT_REACH_SUMMARY" not in captured["serialized"]
    assert "[Old tool result content cleared]" in serialized_turns
    assert "[Old tool result content cleared]" in captured["serialized"]
    assert any("[Old tool result content cleared]" in str(msg.get("content")) for msg in out)
    assert c._last_cheap_tool_cleanup_audit["applied"] is True
    assert c._last_cheap_tool_cleanup_audit["result"] == "summary_on_cleaned_view"
    assert c._last_cheap_tool_cleanup_audit["raw_tool_results_restored_for_summary"] is False
    assert c._last_summary_source_audit["view"] == "cleaned_after_cheap_tool_result_cleanup"
    assert c._last_summary_source_audit["raw_tool_results_restored_for_summary"] is False


def test_audit_records_no_tail_clearing_after_applied_cleanup(monkeypatch):
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=0,
        min_tokens_saved=1,
        skip_llm_summary_when_below_threshold=False,
    )
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.tail_token_budget = 1
    c.bind_session_state(session_db=None, session_id="sess-1")
    monkeypatch.setattr(c, "_generate_summary", lambda turns, focus_topic=None: "## Current Work\n- compressed\n")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "x" * 4000, row_id=11),
        {"role": "assistant", "content": "bridge before protected tail"},
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    c.compress(messages, current_tokens=80_000, force=False, trigger_reason="token_threshold")

    assert c._last_compression_audit_record is not None
    block = c._last_compression_audit_record["cheap_tool_result_cleanup"]
    assert block["applied"] is True
    assert block["result"] == "summary_on_cleaned_view"
    assert block["tail_tool_count"] == block["tail_tool_result_count"]
    assert block["tokens_saved"] == block["tokens_saved_estimate"]
    assert block["protected_tail_cleared_count"] == 0
    assert block["raw_tool_results_restored_for_summary"] is False


def test_cleanup_abort_preserves_raw_transcript_when_summary_generation_fails(monkeypatch):
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=0,
        min_tokens_saved=1,
        skip_llm_summary_when_below_threshold=False,
    )
    c = _compressor(
        abort_on_summary_failure=True,
        cheap_tool_result_cleanup=cfg,
    )
    c.tail_token_budget = 1
    c.bind_session_state(session_db=None, session_id="sess-1")
    captured = {}

    def fake_generate(turns, focus_topic=None):
        captured["turns"] = turns
        return None

    monkeypatch.setattr(c, "_generate_summary", fake_generate)
    big = "RAW_TOOL_OUTPUT_MUST_SURVIVE_ABORT " * 300
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", big, row_id=11),
        {"role": "assistant", "content": "bridge before protected tail"},
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    out = c.compress(messages, current_tokens=90_000, force=False, trigger_reason="token_threshold")

    summary_source = str(captured["turns"])
    assert "[Old tool result content cleared]" in summary_source
    assert "RAW_TOOL_OUTPUT_MUST_SURVIVE_ABORT" not in summary_source
    assert c._last_cheap_tool_cleanup_audit["applied"] is True
    assert c._last_compress_aborted is True
    assert out is messages
    assert any("RAW_TOOL_OUTPUT_MUST_SURVIVE_ABORT" in str(msg.get("content")) for msg in out)
    assert not any("[Old tool result content cleared]" in str(msg.get("content")) for msg in out)


def test_auto_cleanup_only_skips_summary_when_below_threshold(monkeypatch):
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=0,
        min_tokens_saved=1,
        skip_llm_summary_when_below_threshold=True,
    )
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.tail_token_budget = 1
    c.bind_session_state(session_db=None, session_id="sess-1")
    called = {"summary": False}

    def fail_if_called(turns, focus_topic=None):
        called["summary"] = True
        raise AssertionError("summary should be skipped")

    monkeypatch.setattr(c, "_generate_summary", fail_if_called)
    monkeypatch.setattr(c, "threshold_tokens", 50_000)
    big = "x" * 220_000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", big, row_id=11),
        {"role": "assistant", "content": "bridge before protected tail"},
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    out = c.compress(
        messages,
        current_tokens=80_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert called["summary"] is False
    assert any("[Old tool result content cleared]" in str(msg.get("content")) for msg in out)
    audit = c._last_compression_audit_record
    assert audit is not None
    assert audit["result"] == "cheap_cleanup_only"
    assert audit["cheap_tool_result_cleanup"]["result"] == "cheap_cleanup_only"
    assert audit["cheap_tool_result_cleanup"]["llm_summary_skipped_after_cleanup"] is True


@pytest.mark.parametrize(
    "trigger_reason",
    ["message_count_hard_limit", "token_threshold_and_message_count_hard_limit"],
)
def test_hard_message_limit_triggers_do_not_use_cleanup_only(monkeypatch, trigger_reason):
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=0,
        min_tokens_saved=1,
        skip_llm_summary_when_below_threshold=True,
    )
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.tail_token_budget = 1
    c.bind_session_state(session_db=None, session_id="sess-1")
    called = {"summary": False}

    def fake_generate(turns, focus_topic=None):
        called["summary"] = True
        return "## Current Work\n- hard-limit summary\n"

    monkeypatch.setattr(c, "_generate_summary", fake_generate)
    monkeypatch.setattr(c, "threshold_tokens", 50_000)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "x" * 220_000, row_id=11),
        {"role": "assistant", "content": "bridge before protected tail"},
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    c.compress(messages, current_tokens=80_000, force=False, trigger_reason=trigger_reason)

    assert called["summary"] is True
    audit = c._last_compression_audit_record
    assert audit is not None
    assert audit["result"] != "cheap_cleanup_only"
    assert audit["cheap_tool_result_cleanup"]["result"] == "summary_on_cleaned_view"
    assert audit["cheap_tool_result_cleanup"]["llm_summary_skipped_after_cleanup"] is False


def test_manual_compress_does_not_use_cleanup_only(monkeypatch):
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=0,
        min_tokens_saved=1,
        skip_llm_summary_when_below_threshold=True,
    )
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.tail_token_budget = 1
    c.bind_session_state(session_db=None, session_id="sess-1")
    called = {"summary": False}

    def fake_generate(turns, focus_topic=None):
        called["summary"] = True
        return "## Current Work\n- manual summary\n"

    monkeypatch.setattr(c, "_generate_summary", fake_generate)
    monkeypatch.setattr(c, "threshold_tokens", 50_000)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "x" * 220_000, row_id=11),
        {"role": "assistant", "content": "bridge before protected tail"},
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    c.compress(messages, current_tokens=80_000, force=True, trigger_reason="manual")

    assert called["summary"] is True
