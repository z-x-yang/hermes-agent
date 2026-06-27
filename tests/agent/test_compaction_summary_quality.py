"""Lightweight quality checks for nine-section context compaction summaries.

These tests intentionally avoid calling an LLM.  They codify the structural
invariants that make a strong summarizer's checkpoint unambiguous: stable
sections, user-message provenance, and summary/tail attribution boundaries.
"""

from tests.agent.compaction_quality import (
    evaluate_compacted_messages,
    evaluate_nine_section_summary,
)


GOOD_SUMMARY = """[CONTEXT COMPACTION] Earlier turns were compacted into the summary below; treat it as working context, not as a new user request.
## Primary Request and Intent
User asked to keep the nine-section compression structure and evaluate boundary quality.

## Key Technical Concepts
Context compaction is a lossy checkpoint with source attribution boundaries.

## Files and Code Sections
agent/context_compressor.py and compression tests.

## Errors and Fixes
No runtime fix in this fixture.

## Problem Solving
The evaluator checks structure, not model intelligence.

## All User Messages
1. "Keep the nine-section structure."
2. Latest/last user message in compacted range: "Build the minimal evaluator."

## Pending Tasks
None.

## Current Work
Preparing a small pytest-based evaluator.

## Optional Next Step
Run the evaluator against representative fixtures.
"""


def test_quality_eval_accepts_minimal_good_summary():
    failures = evaluate_nine_section_summary(
        GOOD_SUMMARY,
        user_messages=[
            "Keep the nine-section structure.",
            "Build the minimal evaluator.",
        ],
        forbidden_user_attribution_texts=["assistant self note"],
    )
    assert failures == []


def test_quality_eval_rejects_missing_latest_user_anchor():
    bad = GOOD_SUMMARY.replace(
        '2. Latest/last user message in compacted range: "Build the minimal evaluator."',
        '2. "Build the minimal evaluator."',
    )

    failures = evaluate_nine_section_summary(
        bad,
        user_messages=["Keep the nine-section structure.", "Build the minimal evaluator."],
    )

    assert any("latest/last user message" in failure for failure in failures)


def test_quality_eval_rejects_assistant_text_inside_all_user_messages():
    bad = GOOD_SUMMARY.replace(
        '2. Latest/last user message in compacted range: "Build the minimal evaluator."',
        '2. Latest/last user message in compacted range: "Build the minimal evaluator."\n3. "assistant self note"',
    )

    failures = evaluate_nine_section_summary(
        bad,
        user_messages=["Keep the nine-section structure.", "Build the minimal evaluator."],
        forbidden_user_attribution_texts=["assistant self note"],
    )

    assert any("forbidden non-user text" in failure for failure in failures)


def test_quality_eval_rejects_unlabelled_merged_assistant_tail():
    messages = [
        {
            "role": "assistant",
            "content": "[CONTEXT COMPACTION] summary\n\n--- END OF COMPACTED CONTEXT ---\n\nassistant self note",
            "tool_calls": [{"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]

    failures = evaluate_compacted_messages(messages)

    assert any("retained assistant continuation" in failure for failure in failures)


def test_quality_eval_accepts_labelled_merged_assistant_tail():
    messages = [
        {
            "role": "assistant",
            "content": "[CONTEXT COMPACTION] summary\n\n--- END OF COMPACTED CONTEXT ---\n\n[RETAINED ASSISTANT CONTINUATION — not user-provided text]\nassistant self note",
            "tool_calls": [{"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]

    failures = evaluate_compacted_messages(messages)

    assert failures == []
