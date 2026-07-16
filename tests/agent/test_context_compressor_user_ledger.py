"""Tests for user-message preservation in compaction summaries.

The LLM writes the ``## All User Messages`` section (Claude-Code-aligned:
every real user message quoted with a reply-context annotation, carried
forward across iterative updates). Deterministic code no longer replaces
that section; instead it extracts the window's real user messages as ground
truth for the audit sidecar (``logs/compression_user_messages.jsonl``) and
still renders the verbatim ledger for the no-LLM static fallback summary.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


LEDGER_SCOPE_CHANGE = (
    "行，那第一，我们先把这个 skill 的名字统一一下\n"
    "然后降低这个 warning 的触发\n"
    "最后，设计一个 Cron 定期检查和清理陈旧台账，如何？"
)


def _make_compressor(**overrides) -> ContextCompressor:
    kwargs = dict(
        model="test/model",
        threshold_percent=0.85,
        protect_first_n=1,
        protect_last_n=1,
        quiet_mode=True,
    )
    kwargs.update(overrides)
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(**kwargs)


def _llm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _llm_summary_without_user_message() -> str:
    return """## Primary Request and Intent
Continue the old repo context-link implementation.

## Key Technical Concepts
Context compression uses a nine-section summary.

## Files and Code Sections
agent/context_compressor.py

## Errors and Fixes
None.

## Problem Solving
The LLM summary body used for mocking.

## All User Messages
1. "同意" — approving the proposed plan.
2. "审查一下" — asking for a review of the change just made.

## Pending Tasks
Continue repo: context-link work.

## Current Work
Editing active-task-ledger context link support.

## Optional Next Step
Continue the old repo task."""


# ---------------------------------------------------------------------------
# LLM owns the ## All User Messages section (Claude-Code-aligned)
# ---------------------------------------------------------------------------


def test_generate_summary_keeps_llm_written_all_user_messages_section():
    compressor = _make_compressor()
    turns = [
        {"role": "user", "content": "审查一下"},
        {"role": "assistant", "content": "我来查。"},
        {"role": "user", "content": LEDGER_SCOPE_CHANGE},
    ]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_llm_response(_llm_summary_without_user_message()),
    ):
        summary = compressor._generate_summary(turns)

    assert summary is not None
    assert compressor._previous_summary is not None
    all_user_section = summary.split("## All User Messages", 1)[1].split(
        "## Pending Tasks", 1
    )[0]
    # The LLM-written entries survive untouched — including their annotations.
    assert "同意" in all_user_section
    assert "approving the proposed plan" in all_user_section
    # No deterministic fenced ledger is injected over the LLM's section.
    assert "Latest/last user message in compacted range:" not in all_user_section
    assert "```text" not in all_user_section
    # The stored previous summary carries the same LLM-written section forward.
    prev_section = compressor._previous_summary.split("## All User Messages", 1)[1].split(
        "## Pending Tasks", 1
    )[0]
    assert "同意" in prev_section


def test_first_compaction_prompt_instructs_full_user_message_listing():
    compressor = _make_compressor()
    user_message = "本轮用户消息只应该出现在 TURNS TO SUMMARIZE 里。"
    captured_prompt = ""

    def fake_call_llm(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs["messages"][0]["content"]
        return _llm_response(_llm_summary_without_user_message())

    turns = [{"role": "user", "content": user_message}]

    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        summary = compressor._generate_summary(turns)

    assert summary is not None
    # The section is now the LLM's responsibility, with explicit requirements.
    assert "List EVERY real user message" in captured_prompt
    assert "add a brief annotation" in captured_prompt
    assert "never drop or reorder entries" in captured_prompt
    assert "never merge entries that carry distinct content" in captured_prompt
    # The only sanctioned exception: folding consecutive pure-continuation
    # messages into a range entry without renumbering.
    assert "folded into a single entry" in captured_prompt
    assert "never renumber" in captured_prompt
    # Referential closure: annotations must resolve against the whole summary
    # (an LLM reader), not against turns the compaction dropped.
    assert (
        "resolvable by a reader who has this whole summary but not the compacted turns"
        in captured_prompt
    )
    assert (
        "never point at something that survives only in the now-compacted turns"
        in captured_prompt
    )
    assert "defined elsewhere in this summary" in captured_prompt
    # The exclusion list mirrors the deterministic synthetic-note filter:
    # whole synthetic user-role rows and chat-platform scaffolding.
    assert "[Your active task list was preserved" in captured_prompt
    assert "[ASYNC DELEGATION" in captured_prompt
    assert "[IMPORTANT: Background process" in captured_prompt
    assert "[New message]" in captured_prompt
    assert "[Replying to:" in captured_prompt
    # The old deterministic-replacement contract is gone from the prompt.
    assert "replaced deterministically" not in captured_prompt
    assert "USER MESSAGE EVIDENCE LEDGER" not in captured_prompt
    # User messages appear exactly once (in the serialized turns).
    turns_block = captured_prompt.split("TURNS TO SUMMARIZE:", 1)[1].split(
        "Use this exact structure:", 1
    )[0]
    assert user_message in turns_block
    assert captured_prompt.count(user_message) == 1


def test_iterative_prompt_instructs_carry_forward_of_previous_entries():
    compressor = _make_compressor()
    previous_user_message = "上一轮用户消息" + ("甲" * 200)
    current_user_message = "本轮用户消息"
    compressor._previous_summary = f"""## Primary Request and Intent
Previous intent.

## Key Technical Concepts
Compression.

## Files and Code Sections
agent/context_compressor.py

## Errors and Fixes
None.

## Problem Solving
None.

## All User Messages
1. "{previous_user_message}" — prior scope instruction.

## Pending Tasks
Continue.

## Current Work
Previous work.

## Optional Next Step
Continue."""
    captured_prompt = ""

    def fake_call_llm(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs["messages"][0]["content"]
        return _llm_response(_llm_summary_without_user_message())

    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        compressor._generate_summary([{"role": "user", "content": current_user_message}])

    # Carry-forward is an explicit prompt obligation now.
    assert "carry forward every entry" in captured_prompt.lower()
    assert "replaced deterministically" not in captured_prompt
    assert "already-deterministic" not in captured_prompt
    assert "USER MESSAGE EVIDENCE LEDGER" not in captured_prompt
    # Previous ledger content and the new turn each appear exactly once.
    previous_summary_prompt = captured_prompt.split("PREVIOUS SUMMARY:", 1)[1].split(
        "NEW TURNS TO INCORPORATE:", 1
    )[0]
    new_turns_prompt = captured_prompt.split("NEW TURNS TO INCORPORATE:", 1)[1].split(
        "Role=user messages in NEW TURNS", 1
    )[0]
    assert previous_user_message in previous_summary_prompt
    assert current_user_message in new_turns_prompt
    assert captured_prompt.count(previous_user_message) == 1
    assert captured_prompt.count(current_user_message) == 1


# ---------------------------------------------------------------------------
# Deterministic ground-truth extraction (audit insurance)
# ---------------------------------------------------------------------------


def test_extract_current_user_ledger_entries_skips_synthetic_notes():
    synthetic_todo_note = (
        "[Your active task list was preserved across context compression]\n"
        "- [>] inspect. Check the thing (in_progress)"
    )
    synthetic_delegation_note = (
        "[ASYNC DELEGATION BATCH COMPLETE — deleg_test]\n"
        "A background fan-out of 1 subagent(s) you dispatched earlier has finished."
    )
    background_note = (
        "[Need222Say] [IMPORTANT: Background process proc_bcf159723904 "
        "completed normally (exit code 0).\n"
        "Command: python3 monitor.py\n"
        "Output:\n"
        "{\"event\": \"O2_FILE_AGENT_LIVE\"}\n"
        "]"
    )
    real_first = "这是真人用户的新要求，必须保留。"
    real_latest = "这条真人用户消息应该仍然是 latest。"
    turns = [
        {"role": "user", "content": synthetic_todo_note},
        {"role": "user", "content": real_first},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": synthetic_delegation_note},
        {"role": "user", "content": real_latest},
        {"role": "user", "content": background_note},
    ]

    entries = ContextCompressor._extract_current_user_ledger_entries(turns)

    assert [e["text"] for e in entries] == [real_first, real_latest]
    assert entries[-1]["is_latest"] is True
    assert entries[0]["is_latest"] is False


def test_user_ledger_strips_trigger_metadata_and_skips_triggered_runtime_notes():
    trigger_only = (
        "[Triggering message id: `1523072422226428076` — use as `message_id` "
        "for reply/react/pin via the discord tools.]"
    )
    triggered_background_note = (
        trigger_only
        + "\n\n[IMPORTANT: Background process proc_3c2c2ab1a7fd matched watch pattern "
        "\"Uvicorn running on\".\n"
        "Command: ./scripts/run-noema.sh\n"
        "Matched output:\n"
        "INFO:     Uvicorn running on http://127.0.0.1:8790\n]"
    )
    triggered_real_message = trigger_only + "\n\n继续修这个压缩摘要问题。"

    entries = ContextCompressor._extract_current_user_ledger_entries(
        [
            {"role": "user", "content": trigger_only},
            {"role": "user", "content": triggered_background_note},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": triggered_real_message},
        ]
    )

    assert [e["text"] for e in entries] == ["继续修这个压缩摘要问题。"]
    assert entries[-1]["is_latest"] is True


def test_retained_tail_skips_triggered_runtime_notes():
    triggered_background_note = (
        "[Triggering message id: `1523072422226428076` — use as `message_id` "
        "for reply/react/pin via the discord tools.]\n\n"
        "[IMPORTANT: Background process proc_3c2c2ab1a7fd matched watch pattern "
        "\"Uvicorn running on\".\n"
        "Command: ./scripts/run-noema.sh\n"
        "Matched output:\n"
        "INFO:     Uvicorn running on http://127.0.0.1:8790\n]"
    )
    interrupted_background_note = (
        "[System note: The previous turn was interrupted by a gateway shutdown; "
        "the gateway is now back online. Any restart/shutdown command in the "
        "history has already run — do NOT re-execute or verify it.]\n\n"
        + triggered_background_note
    )
    restarted_background_note = (
        "[System note: The previous turn was interrupted by a gateway restart; "
        "the gateway is now back online. Any restart/shutdown command in the "
        "history has already run — do NOT re-execute or verify it.]\n\n"
        + triggered_background_note
    )

    assert ContextCompressor._is_synthetic_retained_user_note(
        {"role": "user", "content": triggered_background_note}
    )
    assert ContextCompressor._is_synthetic_retained_user_note(
        {"role": "user", "content": interrupted_background_note}
    )
    assert ContextCompressor._is_synthetic_retained_user_note(
        {"role": "user", "content": restarted_background_note}
    )


def test_retained_tail_keeps_real_user_text_without_trigger_metadata():
    msg = {
        "role": "user",
        "content": (
            "[Triggering message id: `1523082458323619862` — use as `message_id` "
            "for reply/react/pin via the discord tools.]\n\n"
            "noema-router实现好了？"
        ),
    }

    cleaned = ContextCompressor._sanitize_retained_user_tail_message(dict(msg))

    assert cleaned is not None
    assert cleaned["content"] == "noema-router实现好了？"


def test_generate_summary_records_user_message_ground_truth():
    compressor = _make_compressor()
    synthetic_todo_note = (
        "[Your active task list was preserved across context compression]\n"
        "- [>] verify. Run checks (in_progress)"
    )
    turns = [
        {"role": "user", "content": "审查一下"},
        {"role": "assistant", "content": "我来查。"},
        {"role": "user", "content": synthetic_todo_note},
        {"role": "user", "content": LEDGER_SCOPE_CHANGE},
    ]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_llm_response(_llm_summary_without_user_message()),
    ):
        summary = compressor._generate_summary(turns)

    assert summary is not None
    assert compressor._last_summary_user_message_ground_truth == [
        "审查一下",
        LEDGER_SCOPE_CHANGE,
    ]


def test_ground_truth_is_none_after_summary_failure():
    compressor = _make_compressor()
    # Seed a stale value to prove failure clears it rather than leaking the
    # previous window's messages into the next audit record.
    compressor._last_summary_user_message_ground_truth = ["stale"]

    with patch(
        "agent.context_compressor.call_llm",
        side_effect=RuntimeError("boom"),
    ):
        summary = compressor._generate_summary(
            [{"role": "user", "content": "这条会失败"}]
        )

    assert summary is None
    assert compressor._last_summary_user_message_ground_truth is None


def test_compress_writes_ground_truth_sidecar_with_matching_compression_id(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "agent.context_compressor.get_hermes_home",
        lambda: tmp_path,
        raising=False,
    )
    compressor = _make_compressor(protect_first_n=1, protect_last_n=2)
    compressor._compression_audit_session_id = "sidecar-session"
    window_user_message = "窗口内的真实用户消息 GROUND_TRUTH_SENTINEL"
    msgs = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "initial ask"},
        {"role": "assistant", "content": "middle assistant"},
        {"role": "user", "content": window_user_message},
        {"role": "assistant", "content": "middle reply"},
        {"role": "user", "content": "latest protected ask"},
        {"role": "assistant", "content": "latest protected answer"},
    ]

    with (
        patch.object(compressor, "_prune_old_tool_results", return_value=(msgs, 0)),
        patch.object(compressor, "_find_tail_cut_by_tokens", return_value=5),
        patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(_llm_summary_without_user_message()),
        ),
    ):
        compressor.compress(msgs, current_tokens=90_000, force=True)

    audit_path = tmp_path / "logs" / "compression_audit.jsonl"
    audit_records = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    [main_record] = [r for r in audit_records if r["event"] == "context_compression"]

    sidecar_path = tmp_path / "logs" / "compression_user_messages.jsonl"
    sidecar_records = [
        json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines()
    ]
    [sidecar] = sidecar_records

    assert sidecar["event"] == "compression_user_message_ground_truth"
    assert sidecar["compression_id"] == main_record["compression_id"]
    assert sidecar["session_id"] == "sidecar-session"
    assert sidecar["count"] == 1
    assert sidecar["messages"] == [window_user_message]
    # The main audit stays content-free: count only, never message text.
    assert main_record["user_messages_in_window"] == 1
    assert "GROUND_TRUTH_SENTINEL" not in json.dumps(main_record, ensure_ascii=False)
    # Ground truth is consumed by the write — no stale carry-over.
    assert compressor._last_summary_user_message_ground_truth is None


def test_compress_abort_does_not_write_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.context_compressor.get_hermes_home",
        lambda: tmp_path,
        raising=False,
    )
    compressor = _make_compressor(
        protect_first_n=1, protect_last_n=2, abort_on_summary_failure=True
    )
    msgs = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "initial ask"},
        {"role": "assistant", "content": "middle assistant"},
        {"role": "user", "content": "窗口内消息"},
        {"role": "assistant", "content": "middle reply"},
        {"role": "user", "content": "latest protected ask"},
        {"role": "assistant", "content": "latest protected answer"},
    ]

    with (
        patch.object(compressor, "_prune_old_tool_results", return_value=(msgs, 0)),
        patch.object(compressor, "_find_tail_cut_by_tokens", return_value=5),
        patch(
            "agent.context_compressor.call_llm",
            side_effect=RuntimeError("summary provider down"),
        ),
    ):
        result = compressor.compress(msgs, current_tokens=90_000, force=True)

    assert result == msgs
    assert compressor._last_compress_aborted is True
    assert not (tmp_path / "logs" / "compression_user_messages.jsonl").exists()


# ---------------------------------------------------------------------------
# Static fallback keeps the deterministic verbatim ledger (no LLM available)
# ---------------------------------------------------------------------------


def test_static_fallback_summary_uses_deterministic_user_ledger():
    compressor = _make_compressor()

    summary = compressor._build_static_fallback_summary(
        [{"role": "user", "content": LEDGER_SCOPE_CHANGE}], reason="test failure"
    )

    assert "Latest/last user message in compacted range:" in summary
    assert LEDGER_SCOPE_CHANGE in summary
    all_user_section = summary.split("## All User Messages", 1)[1].split(
        "## Pending Tasks", 1
    )[0]
    assert all_user_section.lstrip().startswith(
        "1. Latest/last user message in compacted range:"
    )


def test_static_fallback_does_not_truncate_kept_user_ledger_entry_mid_message():
    compressor = _make_compressor()
    long_user_message = "START-" + ("中" * 300) + "-END"

    with patch("agent.context_compressor._FALLBACK_SUMMARY_MAX_CHARS", 220):
        summary = compressor._build_static_fallback_summary(
            [{"role": "user", "content": long_user_message}], reason="test failure"
        )

    assert long_user_message in summary
    assert "[fallback summary truncated]" not in summary
    assert "## Optional Next Step" in summary


def test_static_fallback_user_ledger_skips_synthetic_active_task_preservation_note():
    compressor = _make_compressor()
    synthetic_todo_note = (
        "[Your active task list was preserved across context compression]\n"
        "- [>] verify. Run checks (in_progress)"
    )
    real_user_message = "fallback 里这条真人用户消息必须保留。"

    summary = compressor._build_static_fallback_summary(
        [
            {"role": "user", "content": synthetic_todo_note},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": real_user_message},
        ],
        reason="test failure",
    )

    all_user_section = summary.split("## All User Messages", 1)[1].split(
        "## Pending Tasks", 1
    )[0]

    assert "Latest/last user message in compacted range:" in all_user_section
    assert real_user_message in all_user_section
    assert "Your active task list was preserved" not in all_user_section


def test_static_fallback_inherits_previous_ledger_and_filters_synthetic_notes():
    """The fallback's previous-summary parse (deterministic and legacy formats)
    keeps real user evidence and drops system-injected notes."""
    compressor = _make_compressor()
    previous_real_user_message = "上一轮真人用户约束：只保留真人消息。"
    legacy_constraint = "legacy exact user constraint"
    compressor._previous_summary = f"""## Primary Request and Intent
Previous summary.

## Key Technical Concepts
Compression.

## Files and Code Sections
agent/context_compressor.py

## Errors and Fixes
None.

## Problem Solving
None.

## All User Messages
1. User message:
```text
[Your active task list was preserved across context compression]
- [>] inspect. Check synthetic notes (in_progress)
```

2. User message:
```text
{previous_real_user_message}
```

3. Latest/last user message in compacted range:
```text
{legacy_constraint}
```

## Pending Tasks
Continue.

## Current Work
Previous work.

## Optional Next Step
Continue."""

    summary = compressor._build_static_fallback_summary(
        [{"role": "user", "content": "本轮真人用户消息：继续修 ledger。"}],
        reason="test failure",
    )

    all_user_section = summary.split("## All User Messages", 1)[1].split(
        "## Pending Tasks", 1
    )[0]
    assert previous_real_user_message in all_user_section
    assert legacy_constraint in all_user_section
    assert "本轮真人用户消息：继续修 ledger。" in all_user_section
    assert "Your active task list was preserved" not in all_user_section
    assert all_user_section.count("Latest/last user message in compacted range:") == 1


def test_legacy_previous_ledger_parses_for_static_fallback():
    """Legacy (pre-fenced) previous summaries still feed the fallback ledger."""
    compressor = _make_compressor()
    legacy_constraint = "legacy exact user constraint"
    legacy_multiline = "legacy line one\nlegacy line two"
    compressor._previous_summary = f"""## Primary Request and Intent
Old task.

## Key Technical Concepts
Compression.

## Files and Code Sections
agent/context_compressor.py

## Errors and Fixes
None.

## Problem Solving
None.

## All User Messages
1. \"{legacy_multiline}\"
2. Latest/last user message in compacted range: \"{legacy_constraint}\"

## Pending Tasks
Continue.

## Current Work
Old work.

## Optional Next Step
Continue."""

    summary = compressor._build_static_fallback_summary(
        [{"role": "assistant", "content": "no new user"}], reason="test failure"
    )

    all_user_section = summary.split("## All User Messages", 1)[1].split(
        "## Pending Tasks", 1
    )[0]
    assert legacy_multiline in all_user_section
    assert legacy_constraint in all_user_section


# ---------------------------------------------------------------------------
# Ledger budget helpers (still used by the static fallback)
# ---------------------------------------------------------------------------


def test_render_user_ledger_reports_omission_when_all_entries_exceed_cap():
    huge_entry = {
        "source": "new",
        "ordinal": 1,
        "text": "这是一条超过预算的完整用户消息" * 20,
        "is_latest": True,
    }

    kept, omitted_count = ContextCompressor._cap_user_ledger_entries(
        [huge_entry], max_chars=10
    )
    rendered = ContextCompressor._render_user_message_ledger(
        kept, omitted_count=omitted_count
    )

    assert kept == []
    assert "omitted_count=1" in rendered
    assert "USER_LEDGER_MAX_ROUGH_TOKENS=20000" in rendered
    assert "None." not in rendered


def test_user_ledger_budget_omits_whole_old_entries_and_keeps_newer_entries():
    old_entry = {
        "source": "previous",
        "ordinal": 1,
        "text": "旧消息" * 50,
        "is_latest": False,
    }
    new_entry = {
        "source": "new",
        "ordinal": 2,
        "text": "新的短约束",
        "is_latest": True,
    }

    kept, omitted_count = ContextCompressor._cap_user_ledger_entries(
        [old_entry, new_entry], max_chars=ContextCompressor._entry_rough_chars(new_entry)
    )
    rendered = ContextCompressor._render_user_message_ledger(
        kept, omitted_count=omitted_count
    )

    assert [entry["text"] for entry in kept] == ["新的短约束"]
    assert "旧消息" not in rendered
    assert "新的短约束" in rendered
    assert "omitted_count=1" in rendered
