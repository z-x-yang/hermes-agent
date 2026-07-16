"""Tests for live session context breakdown."""

from unittest.mock import MagicMock, patch

import tiktoken

from agent.context_breakdown import compute_session_context_breakdown


def _make_agent(
    *,
    stable: str = "identity and guidance",
    context: str = "",
    volatile: str = "timestamp line",
    tools: list | None = None,
    context_length: int = 200_000,
    compression_context_length: int | None = None,
    last_prompt_tokens: int = 0,
):
    agent = MagicMock()
    agent.model = "openai/gpt-5.4"
    agent.tools = tools or [
        {"type": "function", "function": {"name": "terminal", "description": "run"}},
        {"type": "function", "function": {"name": "mcp_demo_tool", "description": "mcp"}},
        {"type": "function", "function": {"name": "delegate_task", "description": "spawn"}},
    ]
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = True
    agent.context_compressor = MagicMock(
        context_length=context_length,
        compression_context_length=(
            compression_context_length
            if compression_context_length is not None
            else context_length
        ),
        last_prompt_tokens=last_prompt_tokens,
    )
    return agent, {"stable": stable, "context": context, "volatile": volatile}


def test_breakdown_includes_major_categories():
    stable = (
        "base guidance\n"
        "<available_skills>\n  demo:\n    - hello: hi\n</available_skills>"
    )
    context = "# Project Context\nFollow AGENTS.md"
    volatile = "Current time: now"
    history = [{"role": "user", "content": "hello there"}]
    agent, parts = _make_agent(stable=stable, context=context, volatile=volatile)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, history)

    ids = {item["id"] for item in data["categories"]}
    assert {"system_prompt", "tool_definitions", "rules", "skills", "mcp", "subagent_definitions", "conversation"} <= ids
    assert data["context_max"] == 200_000
    assert data["estimated_total"] > 0


def test_breakdown_uses_o200k_for_multilingual_system_prompt():
    stable = "这是一个用于验证中文上下文估算的句子。" * 100
    agent, parts = _make_agent(stable=stable, context="", volatile="", tools=[])
    agent.tools = []

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    expected = len(tiktoken.get_encoding("o200k_base").encode(stable))
    system_prompt = next(
        item for item in data["categories"] if item["id"] == "system_prompt"
    )
    assert system_prompt["tokens"] == expected


def test_breakdown_uses_measured_context_when_available():
    agent, parts = _make_agent(last_prompt_tokens=42_000)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    assert data["context_used"] == 42_000
    assert data["context_percent"] == 21


def test_breakdown_uses_internal_context_window():
    agent, parts = _make_agent(
        context_length=1_000_000,
        compression_context_length=272_000,
        last_prompt_tokens=68_000,
    )

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    assert data["context_max"] == 272_000
    assert data["context_percent"] == 25


def test_breakdown_no_usage_uses_provider_visible_conversation_estimator():
    agent, parts = _make_agent(last_prompt_tokens=0)
    history = [{
        "role": "assistant",
        "content": "persisted shadow",
        "codex_message_items": [{"type": "message", "role": "assistant"}],
    }]

    class _ProviderShapedCompressor:
        context_length = 200_000
        compression_context_length = 200_000
        last_prompt_tokens = 0

        def __init__(self):
            self.seen = None

        def estimate_provider_messages_tokens(self, messages):
            self.seen = messages
            return 1_729

    compressor = _ProviderShapedCompressor()
    agent.context_compressor = compressor

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, history)

    conversation = next(item for item in data["categories"] if item["id"] == "conversation")
    assert conversation["tokens"] == 1_729
    assert compressor.seen is history


def test_breakdown_categories_do_not_exceed_measured_context():
    """Measured provider input tokens are the display total; categories must not contradict it."""
    agent, parts = _make_agent(
        stable="",
        context="",
        volatile="",
        tools=[],
        last_prompt_tokens=100,
    )
    agent.tools = []
    history = [{"role": "user", "content": "x" * 2_000}]  # rough estimate: 500 tokens

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, history)

    assert data["context_used"] == 100
    assert data["estimated_total"] == 100
    assert sum(item["tokens"] for item in data["categories"]) == 100
    conversation = next(item for item in data["categories"] if item["id"] == "conversation")
    assert conversation["tokens"] == 100


def test_breakdown_scales_estimated_prompt_categories_to_measured_context():
    """Even non-conversation estimates must stay on the measured /usage total basis."""
    agent, parts = _make_agent(
        stable="s" * 800,  # rough estimate: 200 tokens
        context="",
        volatile="",
        tools=[],
        last_prompt_tokens=100,
    )
    agent.tools = []

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    assert data["context_used"] == 100
    assert data["estimated_total"] == 100
    assert sum(item["tokens"] for item in data["categories"]) == 100
    assert next(item for item in data["categories"] if item["id"] == "system_prompt")["tokens"] == 100
