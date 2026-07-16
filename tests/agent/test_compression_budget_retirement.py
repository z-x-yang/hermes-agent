"""Tests for summary_budget retirement, truncation fail-fast, and the AUM ledger backstop.

Covers the 2026-07-16 compression overhaul:
  - No numeric token target is injected into the summary prompt; SummaryRules
    carries no budget and _build_summary_rules takes no turn/budget inputs.
  - Summary calls request a fixed generous output allowance instead of an
    input-delta-derived budget.
  - A truncated summary response (finish_reason=length / status=incomplete) is
    a hard failure, never silently accepted as a checkpoint.
  - The ``## All User Messages`` section is bounded by a deterministic token
    backstop that drops oldest whole entries behind an explicit, cumulative
    omission marker.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    _AUM_LEDGER_MAX_TOKENS,
    _SUMMARY_OUTPUT_TOKENS_ALLOWANCE,
    ContextCompressor,
    SummaryRules,
)
from agent.compression_summary_runtime import extract_summary_response_content


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(model="test/model", quiet_mode=True)


# ---------------------------------------------------------------------------
# summary_budget retirement
# ---------------------------------------------------------------------------


def test_summary_rules_carry_no_numeric_budget():
    compressor = _compressor()
    rules = compressor._build_summary_rules()
    assert isinstance(rules, SummaryRules)
    assert not hasattr(rules, "summary_budget")
    assert "Target ~" not in rules.template_sections
    # The concreteness guidance survives budget removal.
    assert "Be CONCRETE" in rules.template_sections


def test_no_numeric_target_in_either_prompt_path():
    compressor = _compressor()
    rules = compressor._build_summary_rules()
    serialized = compressor._build_serialized_summary_prompt(
        rules, "[user] hello", focus_topic=None
    )
    append_instruction = compressor._build_append_cached_summary_instruction(
        rules, focus_topic=None
    )
    assert "Target ~" not in serialized
    assert "Target ~" not in append_instruction


def test_compute_summary_budget_is_retired():
    compressor = _compressor()
    assert not hasattr(compressor, "max_summary_tokens")
    assert not hasattr(ContextCompressor, "_compute_summary_budget")
    assert not hasattr(ContextCompressor, "_compute_summary_budget_from_source")


def test_serialized_summary_call_requests_fixed_output_allowance():
    compressor = _compressor()
    captured: dict = {}

    def _fake_call_llm(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="## Primary Request and Intent\nok")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")]
        )

    with patch("agent.context_compressor.call_llm", _fake_call_llm):
        summary = compressor._generate_summary(
            [{"role": "user", "content": "hello"}]
        )
    assert summary is not None
    assert captured["max_tokens"] == _SUMMARY_OUTPUT_TOKENS_ALLOWANCE


# ---------------------------------------------------------------------------
# truncation fail-fast
# ---------------------------------------------------------------------------


def test_extract_summary_response_content_flags_chat_length_truncation():
    message = SimpleNamespace(content="partial summary", tool_calls=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="length")]
    )
    content, tool_call_violation, truncated = extract_summary_response_content(
        response
    )
    assert content == "partial summary"
    assert not tool_call_violation
    assert truncated


def test_extract_summary_response_content_flags_responses_incomplete():
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="partial")],
            )
        ],
    )
    content, tool_call_violation, truncated = extract_summary_response_content(
        response
    )
    assert content == "partial"
    assert not tool_call_violation
    assert truncated


def test_extract_summary_response_content_clean_response_not_truncated():
    message = SimpleNamespace(content="full summary", tool_calls=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")]
    )
    content, tool_call_violation, truncated = extract_summary_response_content(
        response
    )
    assert content == "full summary"
    assert not tool_call_violation
    assert not truncated


def test_serialized_path_rejects_truncated_summary():
    compressor = _compressor()

    def _truncated_call_llm(**kwargs):
        message = SimpleNamespace(content="## Primary Request and Intent\ncut off mid-")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="length")]
        )

    with patch("agent.context_compressor.call_llm", _truncated_call_llm):
        summary = compressor._generate_summary(
            [{"role": "user", "content": "hello"}]
        )
    assert summary is None
    assert "truncat" in (compressor._last_summary_error or "").lower()


# ---------------------------------------------------------------------------
# AUM ledger backstop
# ---------------------------------------------------------------------------


def _summary_with_entries(count: int, *, marker: str | None = None) -> str:
    lines = ["## Primary Request and Intent", "Fix the bug.", "", "## All User Messages"]
    if marker:
        lines.append(marker)
    for i in range(1, count + 1):
        lines.append(f'{i}. "user message number {i} with some words" — annotation {i}.')
    lines.extend(["", "## Pending Tasks", "None."])
    return "\n".join(lines)


def test_aum_backstop_constant_matches_agreed_cap():
    assert _AUM_LEDGER_MAX_TOKENS == 20_000


def test_aum_backstop_noop_under_cap():
    summary = _summary_with_entries(5)
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(summary)
    assert result == summary
    assert omitted == 0


def test_aum_backstop_drops_oldest_and_marks():
    summary = _summary_with_entries(40)
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(
        summary, max_tokens=200
    )
    assert omitted > 0
    section = ContextCompressor._extract_all_user_messages_section(result)
    first_line = section.splitlines()[0]
    assert first_line.startswith("[Ledger backstop: earliest")
    assert f"earliest {omitted} user messages omitted" in first_line
    # Oldest entries dropped, newest kept with original numbering.
    assert '"user message number 1 ' not in section
    assert '"user message number 40 ' in section
    assert f"{40 - 0}." in section
    # Other sections untouched.
    assert "## Primary Request and Intent\nFix the bug." in result
    assert "## Pending Tasks\nNone." in result


def test_aum_backstop_keeps_at_least_latest_entry():
    summary = _summary_with_entries(3)
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(
        summary, max_tokens=1
    )
    section = ContextCompressor._extract_all_user_messages_section(result)
    assert '"user message number 3 ' in section
    assert omitted == 2


def test_aum_backstop_accumulates_marker_count():
    marker = (
        "[Ledger backstop: earliest 40 user messages omitted to bound this "
        "section; full history remains in the session transcript]"
    )
    summary = _summary_with_entries(30, marker=marker)
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(
        summary, max_tokens=200
    )
    assert omitted > 0
    section = ContextCompressor._extract_all_user_messages_section(result)
    first_line = section.splitlines()[0]
    assert f"earliest {40 + omitted} user messages omitted" in first_line
    assert section.count("[Ledger backstop:") == 1


def test_aum_backstop_reinserts_marker_dropped_by_model():
    marker = (
        "[Ledger backstop: earliest 12 user messages omitted to bound this "
        "section; full history remains in the session transcript]"
    )
    previous_summary = _summary_with_entries(5, marker=marker)
    current = _summary_with_entries(6)  # model dropped the marker line
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(
        current, previous_summary=previous_summary
    )
    assert omitted == 0
    section = ContextCompressor._extract_all_user_messages_section(result)
    assert "earliest 12 user messages omitted" in section.splitlines()[0]


def test_aum_backstop_repairs_duplicated_marker():
    marker = (
        "[Ledger backstop: earliest 7 user messages omitted to bound this "
        "section; full history remains in the session transcript]"
    )
    summary = _summary_with_entries(4, marker=marker)
    # Model copied the marker into the body as well.
    summary = summary.replace(
        '3. "user message number 3',
        marker + '\n3. "user message number 3',
    )
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(summary)
    assert omitted == 0
    section = ContextCompressor._extract_all_user_messages_section(result)
    assert section.count("[Ledger backstop:") == 1
    assert section.splitlines()[0].startswith("[Ledger backstop: earliest 7 ")


def test_aum_backstop_repairs_marker_moved_into_body():
    marker = (
        "[Ledger backstop: earliest 9 user messages omitted to bound this "
        "section; full history remains in the session transcript]"
    )
    summary = _summary_with_entries(4)
    summary = summary.replace(
        '3. "user message number 3',
        marker + '\n3. "user message number 3',
    )
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(summary)
    assert omitted == 0
    section = ContextCompressor._extract_all_user_messages_section(result)
    assert section.count("[Ledger backstop:") == 1
    assert section.splitlines()[0].startswith("[Ledger backstop: earliest 9 ")
    # Entries keep their original numbering and order.
    assert '"user message number 1 ' in section
    assert '"user message number 4 ' in section


def test_aum_backstop_ignores_marker_lookalike_inside_fenced_entry():
    summary = "\n".join(
        [
            "## Primary Request and Intent",
            "Fix the bug.",
            "",
            "## All User Messages",
            "1. User quoted a log excerpt:",
            "```",
            "[Ledger backstop: earliest 99 user messages omitted to bound this "
            "section; full history remains in the session transcript]",
            "```",
            '2. "real follow-up" — new instruction.',
            "",
            "## Pending Tasks",
            "None.",
        ]
    )
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(summary)
    assert result == summary
    assert omitted == 0


def test_aum_backstop_counts_folded_range_as_covered_messages():
    lines = ["## All User Messages"]
    lines.append('1. "first long message ' + "x" * 400 + '" — initial.')
    lines.append("2–100. (ninety-nine consecutive pure continuation approvals)")
    lines.append('101. "latest message ' + "x" * 400 + '" — newest instruction.')
    summary = "\n".join(lines)
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(
        summary, max_tokens=120
    )
    # Entries 1 and 2–100 dropped: 1 + 99 = 100 user messages omitted.
    assert omitted == 100
    section = ContextCompressor._extract_all_user_messages_section(result)
    assert "earliest 100 user messages omitted" in section.splitlines()[0]
    assert '"latest message' in section


def test_aum_backstop_marker_count_is_monotonic_against_model_regression():
    regressed_marker = (
        "[Ledger backstop: earliest 10 user messages omitted to bound this "
        "section; full history remains in the session transcript]"
    )
    true_marker = (
        "[Ledger backstop: earliest 100 user messages omitted to bound this "
        "section; full history remains in the session transcript]"
    )
    previous_summary = _summary_with_entries(3, marker=true_marker)
    current = _summary_with_entries(4, marker=regressed_marker)
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(
        current, previous_summary=previous_summary
    )
    assert omitted == 0
    section = ContextCompressor._extract_all_user_messages_section(result)
    assert "earliest 100 user messages omitted" in section.splitlines()[0]
    assert section.count("[Ledger backstop:") == 1


def test_aum_backstop_preserves_marker_when_model_wipes_section_to_none():
    marker = (
        "[Ledger backstop: earliest 15 user messages omitted to bound this "
        "section; full history remains in the session transcript]"
    )
    previous_summary = _summary_with_entries(2, marker=marker)
    current = "\n".join(
        [
            "## Primary Request and Intent",
            "Fix the bug.",
            "",
            "## All User Messages",
            "None.",
            "",
            "## Pending Tasks",
            "None.",
        ]
    )
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(
        current, previous_summary=previous_summary
    )
    assert omitted == 0
    section = ContextCompressor._extract_all_user_messages_section(result)
    assert "earliest 15 user messages omitted" in section.splitlines()[0]
    assert "## Pending Tasks\nNone." in result


def test_aum_backstop_tracks_fence_opener_char_and_length():
    summary = "\n".join(
        [
            "## All User Messages",
            "1. User pasted a nested markdown example:",
            "````",
            "```",
            "7. this numbered line is quoted text, not an entry",
            "```",
            "````",
            '2. "real follow-up" — new instruction.',
        ]
    )
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(summary)
    assert result == summary
    assert omitted == 0
    # Force pruning: entry 1 (with its whole nested fence) drops as one unit.
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(
        summary, max_tokens=15
    )
    assert omitted == 1
    section = ContextCompressor._extract_all_user_messages_section(result)
    assert "````" not in section
    assert "quoted text" not in section
    assert '"real follow-up"' in section


def test_aum_backstop_recognizes_paren_numbered_entries():
    lines = ["## All User Messages"]
    for i in range(1, 31):
        lines.append(f'{i}) "user message number {i} with some words" — annotation {i}.')
    summary = "\n".join(lines)
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(
        summary, max_tokens=100
    )
    assert omitted > 0
    section = ContextCompressor._extract_all_user_messages_section(result)
    assert section.splitlines()[0].startswith("[Ledger backstop: earliest")
    assert '"user message number 30 ' in section


def test_aum_backstop_without_section_is_noop():
    summary = "## Primary Request and Intent\nFix the bug."
    result, omitted = ContextCompressor._enforce_user_ledger_backstop(summary)
    assert result == summary
    assert omitted == 0


def test_normalize_rewrites_canonical_heading_variants_to_exact_form():
    """Heading variants the normalizer tolerates must be rewritten to the exact
    canonical text, or exact-match consumers (section extraction, the ledger
    backstop, retained-tail sanitizing) silently skip the section."""
    summary = "\n".join(
        [
            "## Primary Request and Intent:",
            "Fix the bug.",
            "",
            "## all user messages",
            '1. "hello" — greeting.',
            "",
            "## Not A Canonical Heading",
            "leaked tool output",
        ]
    )
    normalized, demoted = ContextCompressor._normalize_summary_sections(summary)
    assert demoted == 1
    assert "## Primary Request and Intent\n" in normalized
    assert "## All User Messages\n" in normalized
    assert "**Not A Canonical Heading**" in normalized
    # The backstop now sees the section through its exact-match extraction.
    section = ContextCompressor._extract_all_user_messages_section(normalized)
    assert '"hello"' in section


def test_legacy_ledger_parser_keeps_folded_range_entries():
    summary = "\n".join(
        [
            "## All User Messages",
            '1. "fix the parser" — initial request.',
            "2–5. (four consecutive pure continuation approvals, no new content)",
            '6. "now add tests" — new instruction.',
        ]
    )
    entries = ContextCompressor._parse_previous_user_ledger_entries(summary)
    texts = [entry["text"] for entry in entries]
    assert any("fix the parser" in text for text in texts)
    assert any("pure continuation approvals" in text for text in texts)
    assert any("now add tests" in text for text in texts)


# ---------------------------------------------------------------------------
# prompt content: folding rule, boundary naming, anti-drift
# ---------------------------------------------------------------------------


def test_aum_template_includes_folding_rule_and_marker_preservation():
    compressor = _compressor()
    rules = compressor._build_summary_rules()
    aum = rules.template_sections.split("## All User Messages", 1)[1]
    aum = aum.split("## Pending Tasks", 1)[0]
    assert "fold" in aum.lower()
    assert "never renumber" in aum.lower()
    assert "[Ledger backstop:" in aum


def test_append_instruction_names_previous_summary_boundary():
    compressor = _compressor()
    rules = compressor._build_summary_rules()
    instruction = compressor._build_append_cached_summary_instruction(
        rules, focus_topic=None
    )
    assert "[CONTEXT COMPACTION]" in instruction


def test_iterative_paths_carry_anti_drift_rule():
    compressor = _compressor()
    rules = compressor._build_summary_rules()
    instruction = compressor._build_append_cached_summary_instruction(
        rules, focus_topic=None
    )
    serialized_with_previous = compressor._build_serialized_summary_prompt(
        rules,
        "[user] delta",
        focus_topic=None,
        previous_summary="## Primary Request and Intent\nold state",
    )
    for prompt in (instruction, serialized_with_previous):
        assert "carry its wording forward unchanged" in prompt
