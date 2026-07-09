from types import SimpleNamespace

from agent.conversation_loop import _format_truncated_response_for_user


def test_high_context_truncation_message_is_actionable_and_not_misleading():
    agent = SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=272_000),
    )

    message = _format_truncated_response_for_user(
        agent,
        approx_tokens=255_337,
        finish_reason="length",
        has_tool_calls=True,
    )

    assert "Response truncated due to output length limit" not in message
    assert "255,337 / 272,000" in message
    assert "94%" in message
    assert "tool call" in message.lower()
    assert "No partial tool call was executed" in message
    assert "/compress" in message
