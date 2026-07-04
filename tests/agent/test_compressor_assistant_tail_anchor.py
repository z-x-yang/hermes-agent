"""Regression coverage for removing raw user/assistant tail anchors.

Historically this file pinned #29824's assistant-tail anchor helper. The
nine-section summary contract intentionally removes both the last-user and
last-assistant raw tail anchors: task continuity now belongs in the compaction
summary, while the protected tail stays bounded by token budget and tool-pair
alignment.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX


@pytest.fixture()
def compressor():
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        c.tail_token_budget = 50
        return c


def _content_text(messages):
    return "\n".join(
        m.get("content") for m in messages if isinstance(m.get("content"), str)
    )


class TestAnchorRemovalSourceGuardrail:
    @pytest.fixture()
    def source(self):
        return Path("agent/context_compressor.py").read_text()

    def test_raw_user_and_assistant_anchor_helpers_are_removed(self, source):
        assert "def _ensure_last_user_message_in_tail(" not in source
        assert "def _ensure_last_assistant_message_in_tail(" not in source
        assert "self._ensure_last_user_message_in_tail(" not in source
        assert "self._ensure_last_assistant_message_in_tail(" not in source

    def test_tail_cut_doc_explains_summary_continuation_not_raw_anchors(self, source):
        assert "It does not anchor user or" in source
        assert "active continuation is carried by the compaction" in source
        assert "last user message is always in the tail" not in source
        assert "last assistant message is always in the tail" not in source


class TestFindTailCutByTokensWithoutRawAnchors:
    def test_budget_cut_does_not_pull_old_visible_reply_back_into_tail(self, compressor):
        c = compressor
        c.tail_token_budget = 10
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "PREVIOUSLY VISIBLE REPLY"},
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "c1",
                    "function": {"name": "t", "arguments": "{}"},
                }],
            },
            {"role": "tool", "content": "x" * 200, "tool_call_id": "c1"},
        ]

        cut = c._find_tail_cut_by_tokens(messages, head_end=2)
        tail_text = _content_text(messages[cut:])

        assert cut > 3
        assert "PREVIOUSLY VISIBLE REPLY" not in tail_text
        assert "q2" in tail_text

    def test_tool_call_result_group_alignment_still_holds_without_anchors(self, compressor):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "assistant", "content": "recent continuation"},
        ]

        # A boundary at the tool result should be pulled back before its parent
        # assistant, preserving the protocol pair without relying on user or
        # assistant anchors.
        assert compressor._align_boundary_backward(messages, 3) == 2


class TestCompactionSummaryCarriesFormerVisibleReply:
    def test_compress_keeps_old_reply_only_via_summary_not_raw_tail(self, compressor):
        c = compressor
        c.tail_token_budget = 10
        visible_reply = "THE VISIBLE REPLY THE USER JUST READ"
        summary = f"""{SUMMARY_PREFIX}
## Primary Request and Intent
Continue the long tool-heavy task.

## Key Technical Concepts
Context compression.

## Files and Code Sections
None.

## Errors and Fixes
None.

## Problem Solving
The old visible reply was summarized instead of pinned raw.

## All User Messages
- initial
- follow up

## Pending Tasks
- Continue after the tool result.

## Current Work
Previous assistant reply: {visible_reply}

## Optional Next Step
Inspect the latest tool output."""
        messages = (
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "initial"},
            ]
            + [
                {"role": "user", "content": f"middle q{i}"}
                if i % 2 == 0
                else {"role": "assistant", "content": f"middle reply {i}"}
                for i in range(12)
            ]
            + [
                {"role": "user", "content": "the visible question"},
                {"role": "assistant", "content": visible_reply},
                {"role": "user", "content": "follow up"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "c1",
                        "function": {"name": "t", "arguments": "{}"},
                    }],
                },
                {"role": "tool", "content": "z" * 500, "tool_call_id": "c1"},
            ]
        )

        with patch.object(c, "_generate_summary", return_value=summary):
            result = c.compress(messages, current_tokens=90_000)

        summary_text = _content_text(
            m for m in result
            if isinstance(m.get("content"), str)
            and m["content"].startswith(SUMMARY_PREFIX)
        )
        non_summary_text = _content_text(
            m for m in result
            if not (
                isinstance(m.get("content"), str)
                and m["content"].startswith(SUMMARY_PREFIX)
            )
        )

        assert visible_reply in summary_text
        assert visible_reply not in non_summary_text
