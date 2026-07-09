from __future__ import annotations


class _FakeCompressor:
    def __init__(self) -> None:
        self.calls = []

    def estimate_provider_request_tokens(self, messages, *, system_prompt="", tools=None):
        self.calls.append(
            {
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
            }
        )
        return 4242


class _FakeAgent:
    def __init__(self) -> None:
        self.context_compressor = _FakeCompressor()
        self.tools = [{"type": "function", "function": {"name": "terminal"}}]


def test_hygiene_compression_history_preserves_provider_visible_tool_turns():
    from gateway.run import _build_hygiene_compression_history

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "terminal", "arguments": "{\"command\":\"pwd\"}"},
        }
    ]
    codex_reasoning_items = [{"type": "reasoning", "encrypted_content": "sealed"}]
    history = [
        {"role": "system", "content": "not replayed from transcript"},
        {"role": "user", "content": "run pwd", "timestamp": 1.0},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
            "codex_reasoning_items": codex_reasoning_items,
            "finish_reason": "tool_calls",
            "timestamp": 2.0,
        },
        {
            "role": "tool",
            "content": "/tmp/project",
            "tool_call_id": "call_1",
            "tool_name": "terminal",
            "timestamp": 3.0,
        },
        {"role": "assistant", "content": "done", "timestamp": 4.0},
    ]

    prepared = _build_hygiene_compression_history(history)

    assert [m["role"] for m in prepared] == ["user", "assistant", "tool", "assistant"]
    assert prepared[1]["tool_calls"] == tool_calls
    assert prepared[1]["codex_reasoning_items"] == codex_reasoning_items
    assert prepared[1]["finish_reason"] == "tool_calls"
    assert prepared[2] == {
        "role": "tool",
        "content": "/tmp/project",
        "tool_call_id": "call_1",
        "tool_name": "terminal",
    }
    assert all("timestamp" not in m for m in prepared)


def test_hygiene_compression_request_estimate_uses_full_provider_shape():
    from gateway.run import _estimate_hygiene_compressed_request_tokens

    agent = _FakeAgent()
    compressed = [{"role": "user", "content": "summary"}]

    tokens = _estimate_hygiene_compressed_request_tokens(
        agent,
        compressed,
        system_prompt="SYSTEM",
    )

    assert tokens == 4242
    assert agent.context_compressor.calls == [
        {
            "messages": compressed,
            "system_prompt": "SYSTEM",
            "tools": agent.tools,
        }
    ]
