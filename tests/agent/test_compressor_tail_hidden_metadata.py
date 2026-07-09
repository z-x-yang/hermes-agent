"""Retained tail provider replay metadata is not a cheap-cleanup target.

Claude-like cheap cleanup only clears old eligible tool-result bodies.  It does
not clear signed/encrypted reasoning replay, assistant message replay items, or
assistant tool-call arguments.  Hermes used to run an extra Stage-1b hygiene pass
after cleanup/compaction that bounded these fields; that saved storage-looking
tokens but diverged from Claude Code and could remove provider-visible replay
state.  These tests pin the boundary so retained-tail behavior is deterministic.
"""

from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    SUMMARY_PREFIX,
    _estimate_msg_budget_tokens,
)
from agent.model_metadata import estimate_messages_tokens_rough


def _make_compressor(protect_last_n=4, tail_budget=1_000):
    with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=protect_last_n,
            quiet_mode=True,
        )
    c.tail_token_budget = tail_budget
    return c


def _heavy_assistant_tool_call(call_id, name="terminal"):
    """Assistant tool-call turn carrying very large non-visible metadata."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                # oversized arguments blob (mirrors 23,571-char tool_calls field)
                "arguments": '{"command":"' + ("run_step.py " * 2_000) + '"}',
            },
        }],
        "reasoning": "R" * 8_000,
        "reasoning_content": "C" * 8_000,
        "codex_reasoning_items": [
            {"id": f"rs_{call_id}", "type": "reasoning",
             "encrypted_content": "E" * 40_000}
        ],
        "codex_message_items": [
            {"id": f"mi_{call_id}", "type": "message", "content": "M" * 6_000}
        ],
    }


def _build_history():
    """system + summarizable middle + protected tail with heavy hidden metadata."""
    return [
        {"role": "system", "content": "System prompt"},          # 0 head
        {"role": "user", "content": "early ask to summarize"},    # 1 middle
        {"role": "assistant", "content": "early reply"},          # 2 middle
        {"role": "user", "content": "mid ask"},                   # 3 middle
        {"role": "assistant", "content": "mid reply"},            # 4 middle
        {"role": "user", "content": "latest protected ask"},      # 5 tail start
        _heavy_assistant_tool_call("call_a"),                     # 6
        {"role": "tool", "tool_call_id": "call_a", "content": "ok a"},   # 7
        _heavy_assistant_tool_call("call_b"),                     # 8
        {"role": "tool", "tool_call_id": "call_b", "content": "ok b"},   # 9
        {                                                         # 10 visible reply + metadata
            "role": "assistant",
            "content": "final visible reply",
            "reasoning": "R" * 8_000,
            "reasoning_content": "C" * 8_000,
            "codex_reasoning_items": [
                {"id": "rs_final", "type": "reasoning", "encrypted_content": "E" * 40_000}
            ],
        },
    ]


TAIL_START = 5


def _compress(c, msgs):
    with (
        patch.object(c, "_prune_old_tool_results", return_value=(msgs, 0)),
        patch.object(c, "_find_tail_cut_by_tokens", return_value=TAIL_START),
        patch.object(c, "_generate_summary", return_value=f"{SUMMARY_PREFIX}\nsummary"),
    ):
        return c.compress(msgs, current_tokens=200_000)


class TestRetainedTailHiddenMetadata:
    def test_tail_budget_counts_unbounded_replay_metadata(self):
        msg = _heavy_assistant_tool_call("call_budget")

        budget_tokens = _estimate_msg_budget_tokens(msg)

        assert budget_tokens > 10_000

    def test_preserves_encrypted_reasoning_and_message_items_in_tail(self):
        c = _make_compressor()
        result = _compress(c, _build_history())

        seen_reasoning = False
        seen_message_items = False
        for m in result:
            if m.get("role") == "assistant":
                if m.get("codex_reasoning_items"):
                    seen_reasoning = True
                if m.get("codex_message_items"):
                    seen_message_items = True
        assert seen_reasoning
        assert seen_message_items

    def test_preserves_reasoning_text_fields(self):
        c = _make_compressor()
        result = _compress(c, _build_history())

        preserved_any = False
        for m in result:
            if m.get("role") != "assistant":
                continue
            for key in ("reasoning", "reasoning_content"):
                v = m.get(key)
                if v is None:
                    continue
                assert isinstance(v, str)
                assert len(v) == 8_000
                if key == "reasoning_content":
                    assert v.strip(), "reasoning_content must remain non-empty"
                preserved_any = True
        assert preserved_any, "expected the heavy tail reasoning fields to be retained"

    def test_preserves_oversized_tool_args_and_tool_pairing(self):
        c = _make_compressor()
        result = _compress(c, _build_history())

        tool_call_ids = set()
        for m in result:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                tool_call_ids.add(tc["id"])
                assert tc["function"]["name"] == "terminal"
                args = tc["function"]["arguments"]
                assert len(args) > 20_000

        # tool-result pairing stays valid: every tool result still has a
        # matching assistant tool_call id, and both heavy calls survive.
        assert {"call_a", "call_b"} <= tool_call_ids
        for m in result:
            if m.get("role") == "tool":
                assert m["tool_call_id"] in tool_call_ids

    def test_compression_does_not_claim_stage1b_storage_savings(self):
        c = _make_compressor()
        msgs = _build_history()
        pre = estimate_messages_tokens_rough(msgs)
        result = _compress(c, msgs)
        post = estimate_messages_tokens_rough(result)

        assert post > pre // 2, (
            "retained reasoning/tool-call replay metadata should remain in the "
            f"rough storage estimate: {pre:,} -> {post:,} tokens"
        )

    def test_no_orphan_tool_results(self):
        """Bounding metadata must not break tool_call/tool_result structure."""
        c = _make_compressor()
        result = _compress(c, _build_history())

        assistant_call_ids = {
            tc["id"]
            for m in result
            if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
        }
        tool_result_ids = [
            m.get("tool_call_id") for m in result if m.get("role") == "tool"
        ]
        for rid in tool_result_ids:
            assert rid in assistant_call_ids, f"orphan tool result: {rid}"


class TestEndToEndFreesSpaceWithRealBoundary:
    """No pinned boundary — exercises the real tail-cut + assembly path while
    preserving provider replay metadata in retained assistant messages."""

    def _long_history(self, n_turns=14):
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Start the long-running task."},
            {"role": "assistant", "content": "On it."},
        ]
        for i in range(n_turns):
            msgs.append({"role": "user", "content": f"continue step {i}"})
            msgs.append(_heavy_assistant_tool_call(f"call_{i}"))
            msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "ok"})
            msgs.append({"role": "assistant", "content": f"done step {i}"})
        return msgs

    def test_real_path_drops_well_under_threshold(self):
        c = _make_compressor(protect_last_n=20, tail_budget=2_000)
        msgs = self._long_history()
        pre = estimate_messages_tokens_rough(msgs)

        with patch.object(c, "_generate_summary", return_value=f"{SUMMARY_PREFIX}\nsummary"):
            result = c.compress(msgs, current_tokens=pre)

        post = estimate_messages_tokens_rough(result)
        # Progress signal the conversation loop relies on (len < original_len).
        assert len(result) < len(msgs)
        assert post > pre // 3
        # Encrypted reasoning replay survives in retained assistant messages.
        assert any(
            m.get("codex_reasoning_items") or m.get("codex_message_items")
            for m in result
            if m.get("role") == "assistant"
        )
