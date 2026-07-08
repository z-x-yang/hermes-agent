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


def test_cleanup_ignores_tools_outside_claude_code_cleanup_list():
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_id="sess-1")
    eligible_raw = "ELIGIBLE_READ_OUTPUT " + ("x" * 4000)
    ineligible_raw = "INELIGIBLE_DELEGATE_OUTPUT " + ("y" * 4000)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("read-call", name="read_file"),
        _tool("read-call", eligible_raw, row_id=10),
        _assistant_call("delegate-call", name="delegate_task"),
        _tool("delegate-call", ineligible_raw, row_id=11),
        {"role": "user", "content": "tail"},
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=6)

    assert result.applied is True
    assert result.messages[3]["content"] == "[Old tool result content cleared]"
    assert result.messages[5]["content"] == ineligible_raw
    assert result.audit["candidate_count"] == 1
    assert result.audit["clear_candidate_count"] == 1
    assert result.audit["ineligible_tool_result_count"] == 1


def test_cleanup_keep_recent_can_clear_older_protected_tail_tool_results():
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=5, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_id="sess-1")
    big = "x" * 4000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("pre-tail-0", name="read_file"),
        _tool("pre-tail-0", f"PRE_TAIL_RAW\n{big}", row_id=10),
    ]
    for i in range(6):
        messages.append(_assistant_call(f"tail-{i}", name="read_file"))
        messages.append(_tool(f"tail-{i}", f"TAIL_RAW_{i}\n{big}", row_id=20 + i))

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    assert result.applied is True
    assert result.messages[3]["content"] == "[Old tool result content cleared]"
    # keep_recent=5 is global, so the oldest retained-tail tool result is also
    # cleared when the tail itself has more than 5 eligible tool results.
    assert result.messages[5]["content"] == "[Old tool result content cleared]"
    for index, tail_i in zip([7, 9, 11, 13, 15], range(1, 6), strict=True):
        assert f"TAIL_RAW_{tail_i}" in result.messages[index]["content"]
    assert result.audit["tail_tool_result_count"] == 6
    assert result.audit["candidate_count"] == 7
    assert result.audit["kept_recent_count"] == 5
    assert result.audit["cleared_count"] == 2
    assert result.audit["protected_tail_cleared_count"] == 1


def test_cleanup_below_min_tokens_saved_does_not_report_tail_cleared():
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=5,
        min_tokens_saved=1_000_000,
    )
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_id="sess-1")
    raw = "x" * 4000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("pre-tail-0", name="read_file"),
        _tool("pre-tail-0", f"PRE_TAIL_RAW\n{raw}", row_id=10),
    ]
    for i in range(6):
        messages.append(_assistant_call(f"tail-{i}", name="read_file"))
        messages.append(_tool(f"tail-{i}", f"TAIL_RAW_{i}\n{raw}", row_id=20 + i))

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    assert result.applied is False
    assert result.audit["result"] == "below_min_tokens_saved"
    assert result.audit["clear_candidate_count"] == 2
    assert result.audit["cleared_count"] == 0
    assert result.audit["replacement_counts"] == {"persisted_handle": 0, "sentinel": 0}
    assert result.audit["protected_tail_cleared_count"] == 0
    assert result.messages is messages
    assert "TAIL_RAW_0" in result.messages[5]["content"]


def test_cleanup_keep_recent_counts_only_eligible_tail_tools():
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=2, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_id="sess-1")
    big = "x" * 4000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-read", name="read_file"),
        _tool("old-read", f"OLD_READ\n{big}", row_id=10),
        _assistant_call("tail-delegate", name="delegate_task"),
        _tool("tail-delegate", f"TAIL_DELEGATE\n{big}", row_id=20),
        _assistant_call("tail-read-0", name="read_file"),
        _tool("tail-read-0", f"TAIL_READ_0\n{big}", row_id=21),
        _assistant_call("tail-read-1", name="read_file"),
        _tool("tail-read-1", f"TAIL_READ_1\n{big}", row_id=22),
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    assert result.applied is True
    assert result.messages[3]["content"] == "[Old tool result content cleared]"
    assert "TAIL_DELEGATE" in result.messages[5]["content"]
    assert "TAIL_READ_0" in result.messages[7]["content"]
    assert "TAIL_READ_1" in result.messages[9]["content"]
    assert result.audit["tail_tool_result_count"] == 2
    assert result.audit["ineligible_tool_result_count"] == 1
    assert result.audit["kept_recent_count"] == 2


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


def test_persisted_handle_resolves_missing_row_id_from_bound_session_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session("cheap-cleanup-test", "cli")
    raw_tool_output = "recover me without live row id " + ("x" * 4000)
    raw_messages = [
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", raw_tool_output),
    ]
    db.replace_messages(session_id, raw_messages)
    loaded = db.get_messages(session_id)
    persisted_tool_row = next(msg for msg in loaded if msg.get("role") == "tool")
    live_tool_msg = dict(persisted_tool_row)
    live_tool_msg.pop("id", None)

    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_db=db, session_id=session_id)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        live_tool_msg,
        {"role": "user", "content": "tail user"},
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    cleaned_content = result.messages[3]["content"]
    assert f"hermes://session/{session_id}/message/{persisted_tool_row['id']}" in cleaned_content
    assert "session_search(" in cleaned_content
    assert raw_tool_output not in cleaned_content
    assert result.audit["replacement_counts"] == {"persisted_handle": 1, "sentinel": 0}
    assert result.audit["sentinel_fallback_reasons"] == {}


def test_missing_row_id_lookup_is_built_once_per_cleanup_pass(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session("cheap-cleanup-test", "cli")
    raw_messages = [
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "first recoverable " + ("x" * 4000)),
        _assistant_call("old-2"),
        _tool("old-2", "second recoverable " + ("y" * 4000)),
    ]
    db.replace_messages(session_id, raw_messages)
    loaded = db.get_messages(session_id)
    live_messages = []
    for msg in loaded:
        live_msg = dict(msg)
        live_msg.pop("id", None)
        live_messages.append(live_msg)

    get_messages_calls = 0
    original_get_messages = db.get_messages

    def counted_get_messages(*args, **kwargs):
        nonlocal get_messages_calls
        get_messages_calls += 1
        return original_get_messages(*args, **kwargs)

    monkeypatch.setattr(db, "get_messages", counted_get_messages)
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_db=db, session_id=session_id)
    messages = [
        {"role": "system", "content": "sys"},
        *live_messages,
        {"role": "user", "content": "tail user"},
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=6)

    assert get_messages_calls == 1
    assert result.audit["replacement_counts"] == {"persisted_handle": 2, "sentinel": 0}
    assert result.audit["sentinel_fallback_reasons"] == {}



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
    assert block["scope"] == "eligible_tool_results_across_provider_history"
    assert "read_file" in block["eligible_tool_names"]
    assert "delegate_task" not in block["eligible_tool_names"]
    assert block["tail_tool_result_count"] == 0
    assert block["tail_tool_count"] == 0
    assert block["extra_pre_tail_keep_count"] == 0
    assert block["candidate_count"] == 0
    assert block["eligible_tool_result_count"] == 0
    assert block["ineligible_tool_result_count"] == 0
    assert block["clear_candidate_count"] == 0
    assert block["kept_recent_count"] == 0
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


def test_auto_cleanup_only_runs_when_tail_floor_leaves_no_summary_window(monkeypatch):
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=1,
        min_tokens_saved=1,
        skip_llm_summary_when_below_threshold=True,
    )
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.80,
            protect_first_n=1,
            protect_last_n=50,
            summary_target_ratio=0.10,
            quiet_mode=True,
            cheap_tool_result_cleanup=cfg,
        )
    c.bind_session_state(session_db=None, session_id="sess-1")
    monkeypatch.setattr(c, "threshold_tokens", 80_000)
    monkeypatch.setattr(
        c,
        "_generate_summary",
        lambda turns, focus_topic=None: (_ for _ in ()).throw(
            AssertionError("summary should not run for cleanup-only relief")
        ),
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
    ]
    for i in range(4):
        messages.append(_assistant_call(f"tail-read-{i}", name="read_file"))
        messages.append(_tool(f"tail-read-{i}", f"TAIL_READ_{i}\n" + ("x" * 220_000), row_id=20 + i))
    messages.append({"role": "user", "content": "latest tail user"})

    out = c.compress(
        messages,
        current_tokens=90_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert any("[Old tool result content cleared]" in str(msg.get("content")) for msg in out)
    assert "TAIL_READ_3" in str(out[-2]["content"])
    audit = c._last_compression_audit_record
    assert audit is not None
    assert audit["result"] == "cheap_cleanup_only"
    block = audit["cheap_tool_result_cleanup"]
    assert block["cleared_count"] == 3
    assert block["protected_tail_cleared_count"] == 3
    assert block["kept_recent_count"] == 1
    assert block["llm_summary_skipped_after_cleanup"] is True


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


def test_manual_compress_resolves_missing_row_ids_to_persisted_handles(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session("cheap-cleanup-test", "cli")
    raw_tool_output = "MANUAL_RAW_TOOL_OUTPUT_SHOULD_BE_HANDLE " * 300
    raw_messages = [
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", raw_tool_output),
    ]
    db.replace_messages(session_id, raw_messages)
    persisted_tool_row = next(
        msg for msg in db.get_messages(session_id) if msg.get("role") == "tool"
    )
    live_tool_msg = dict(persisted_tool_row)
    live_tool_msg.pop("id", None)

    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=0,
        min_tokens_saved=1,
        skip_llm_summary_when_below_threshold=True,
    )
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.tail_token_budget = 1
    c.bind_session_state(session_db=db, session_id=session_id)
    captured = {}

    def fake_generate(turns, focus_topic=None):
        captured["serialized"] = c._serialize_for_summary(turns)
        return "## Current Work\n- manual summary\n"

    monkeypatch.setattr(c, "_generate_summary", fake_generate)
    monkeypatch.setattr(c, "threshold_tokens", 50_000)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        live_tool_msg,
        {"role": "assistant", "content": "bridge before protected tail"},
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    c.compress(messages, current_tokens=80_000, force=True, trigger_reason="manual")

    persisted_uri = f"hermes://session/{session_id}/message/{persisted_tool_row['id']}"
    assert persisted_uri in captured["serialized"]
    assert "MANUAL_RAW_TOOL_OUTPUT_SHOULD_BE_HANDLE" not in captured["serialized"]
    audit = c._last_compression_audit_record
    assert audit is not None
    assert audit["result"] != "cheap_cleanup_only"
    block = audit["cheap_tool_result_cleanup"]
    assert block["result"] == "summary_on_cleaned_view"
    assert block["replacement_counts"] == {"persisted_handle": 1, "sentinel": 0}
    assert block["sentinel_fallback_reasons"] == {}
    assert block["raw_tool_results_restored_for_summary"] is False
