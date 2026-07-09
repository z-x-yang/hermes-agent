"""Tests for gateway /compress user-facing messaging."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str = "/compress") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def _make_tool_history() -> list[dict]:
    return [
        {
            "role": "user",
            "content": "run pwd",
            "timestamp": 1.0,
            "message_id": "platform-user-1",
            "token_count": 3,
            "active": 1,
        },
        {
            "role": "assistant",
            "content": "",
            "timestamp": 2.0,
            "platform_message_id": "platform-assistant-1",
            "observed": True,
            "tool_calls": [
                {
                    "id": "call_pwd",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "pwd"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "/Users/zongxin/project",
            "timestamp": 3.0,
            "tool_call_id": "call_pwd",
            "tool_name": "terminal",
            "compacted": 0,
        },
        {
            "role": "assistant",
            "content": "done",
            "timestamp": 4.0,
            "token_count": 1,
        },
    ]


def _make_runner(history: list[dict]):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store._save = MagicMock()
    runner._session_db = None
    return runner


@pytest.mark.asyncio
async def test_compress_command_reports_noop_without_success_banner():
    history = _make_history()
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")

    def _estimate(messages, **_kwargs):
        assert messages == history
        return 100

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "No changes from compression" in result
    assert "Compressed:" not in result
    assert "Approx request size: ~100 tokens (unchanged)" in result
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_explains_when_token_estimate_rises():
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "Dense summary that still counts as more tokens."},
        history[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 120
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed: 4 → 3 messages" in result
    assert "Approx request size: ~100 → ~120 tokens" in result
    assert "denser summaries" in result
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_builds_prompt_before_estimating_tokens():
    """Manual /compress feedback must compare before/after estimates with
    the same request-shape basis.  A fresh temporary agent can have an empty
    cached prompt; estimating the pre-compress request with that empty prompt
    but the post-compress request with a rebuilt prompt makes compression look
    like it increased tokens for the wrong reason.
    """
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance._build_system_prompt.return_value = "BUILT SYSTEM PROMPT"
    agent_instance.tools = [{"type": "function", "function": {"name": "demo"}}]
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.session_id = "sess-1"

    def _compress(messages, system_message, **_kwargs):
        assert system_message == ""
        agent_instance._cached_system_prompt = "POST-COMPRESSION SYSTEM PROMPT"
        return compressed, "POST-COMPRESSION SYSTEM PROMPT"

    agent_instance._compress_context.side_effect = _compress
    estimate_calls = []

    def _estimate(messages, **kwargs):
        estimate_calls.append((messages, kwargs))
        if messages == history:
            return 100
        if messages == compressed:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed:" in result
    assert estimate_calls[0][1]["system_prompt"] == "BUILT SYSTEM PROMPT"
    assert estimate_calls[1][1]["system_prompt"] == "POST-COMPRESSION SYSTEM PROMPT"
    assert estimate_calls[0][1]["tools"] is agent_instance.tools
    assert estimate_calls[1][1]["tools"] is agent_instance.tools


@pytest.mark.asyncio
async def test_compress_command_reports_provider_visible_token_estimates():
    """Manual /compress feedback must use provider-visible request accounting,
    not raw DB message accounting, for before/after token estimates.
    """
    history = _make_tool_history()
    filtered_history = [
        {"role": "user", "content": "run pwd"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": history[1]["tool_calls"],
        },
        {
            "role": "tool",
            "content": "/Users/zongxin/project",
            "tool_call_id": "call_pwd",
            "tool_name": "terminal",
        },
        {"role": "assistant", "content": "done"},
    ]
    compressed = [
        {"role": "assistant", "content": "[summary]"},
        history[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = "SYSTEM"
    agent_instance.tools = [{"type": "function", "function": {"name": "demo"}}]
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "SYSTEM")

    estimate_calls = []

    def _provider_visible_estimate(messages, **kwargs):
        estimate_calls.append((messages, kwargs))
        if messages == filtered_history:
            return 90
        if messages == compressed:
            return 30
        raise AssertionError(f"unexpected transcript: {messages!r}")

    agent_instance.context_compressor.estimate_provider_request_tokens.side_effect = (
        _provider_visible_estimate
    )

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            side_effect=AssertionError("raw estimator must not be used"),
        ),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed: 4 → 2 messages" in result
    assert "Approx request size: ~90 → ~30 tokens" in result
    assert estimate_calls[0][1]["system_prompt"] == "SYSTEM"
    assert estimate_calls[0][1]["tools"] is agent_instance.tools


@pytest.mark.asyncio
async def test_compress_command_estimates_full_next_turn_not_memory_only_agent():
    """Gateway /compress uses a memory-only temporary agent to run the
    summariser, but the displayed request size should describe the next normal
    turn with the full system prompt and tool schemas.
    """
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)

    compression_agent = MagicMock()
    compression_agent.shutdown_memory_provider = MagicMock()
    compression_agent.close = MagicMock()
    compression_agent._cached_system_prompt = "MEMORY ONLY SYSTEM"
    compression_agent.tools = [{"type": "function", "function": {"name": "memory"}}]
    compression_agent.context_compressor.has_content_to_compress.return_value = True
    compression_agent.context_compressor._last_compress_aborted = False
    compression_agent.context_compressor._last_aux_model_failure_model = None
    compression_agent.context_compressor.estimate_provider_request_tokens.side_effect = [
        40,
        20,
    ]
    compression_agent.session_id = "sess-1"
    compression_agent._compress_context.return_value = (compressed, "MEMORY ONLY SYSTEM")

    full_agent = MagicMock()
    full_agent.shutdown_memory_provider = MagicMock()
    full_agent.close = MagicMock()
    full_agent._cached_system_prompt = "FULL SYSTEM"
    full_agent.tools = [{"type": "function", "function": {"name": "terminal"}}]
    full_agent.context_compressor.estimate_provider_request_tokens.side_effect = [
        400,
        300,
    ]

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", side_effect=[compression_agent, full_agent]) as cls,
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Approx request size: ~400 → ~300 tokens" in result
    assert cls.call_args_list[0].kwargs["enabled_toolsets"] == ["memory"]
    assert cls.call_args_list[1].kwargs["enabled_toolsets"] != ["memory"]
    assert "terminal" in cls.call_args_list[1].kwargs["enabled_toolsets"]
    assert cls.call_args_list[1].kwargs["skip_memory"] is True
    assert cls.call_args_list[1].kwargs["session_id"].endswith(":compress-estimate")
    compression_agent.context_compressor.estimate_provider_request_tokens.assert_not_called()
    full_agent.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_restores_session_context_after_estimate_agent():
    """The full-tool estimate helper uses a synthetic session id; constructing
    it must not leave the live gateway turn bound to that fake id.
    """
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)

    compression_agent = MagicMock()
    compression_agent.shutdown_memory_provider = MagicMock()
    compression_agent.close = MagicMock()
    compression_agent._cached_system_prompt = "MEMORY ONLY SYSTEM"
    compression_agent.tools = [{"type": "function", "function": {"name": "memory"}}]
    compression_agent.context_compressor.has_content_to_compress.return_value = True
    compression_agent.context_compressor._last_compress_aborted = False
    compression_agent.context_compressor._last_aux_model_failure_model = None
    compression_agent.session_id = "sess-1"
    compression_agent._compress_context.return_value = (compressed, "MEMORY ONLY SYSTEM")

    full_agent = MagicMock()
    full_agent.close = MagicMock()
    full_agent._cached_system_prompt = "FULL SYSTEM"
    full_agent.tools = [{"type": "function", "function": {"name": "terminal"}}]
    full_agent.context_compressor.estimate_provider_request_tokens.side_effect = [
        400,
        300,
    ]

    constructed = []

    def _construct_agent(**kwargs):
        constructed.append(kwargs)
        from gateway.session_context import set_current_session_id

        set_current_session_id(kwargs["session_id"])
        return compression_agent if len(constructed) == 1 else full_agent

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", side_effect=_construct_agent),
        patch("gateway.session_context.set_current_session_id") as set_sid,
        patch("hermes_logging.set_session_context") as set_log_sid,
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed:" in result
    assert constructed[1]["session_id"].endswith(":compress-estimate")
    assert set_sid.call_args_list[-1].args[0] == "sess-1"
    assert set_log_sid.call_args_list[-1].args[0] == "sess-1"


@pytest.mark.asyncio
async def test_compress_command_appends_warning_when_compression_aborts():
    """When the auxiliary summariser fails and the compressor ABORTS (returns
    messages unchanged), /compress must append a visible ⚠️ warning to its
    reply telling the user nothing was dropped and how to retry. Otherwise
    the failure is silently logged and the user has no idea why nothing
    happened."""
    history = _make_history()
    # Abort path: compressor returns the input messages unchanged.
    compressed = list(history)
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    # Simulate compression aborting (force=True bypassed cooldown but the
    # aux LLM is genuinely broken).
    agent_instance.context_compressor._last_compress_aborted = True
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = (
        "404 model not found: gemini-3-flash-preview"
    )
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 100
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    # A clearly-marked warning must be appended.
    assert "⚠️" in result
    assert "Compression aborted" in result
    # Underlying error must surface so users can fix their config.
    assert "404 model not found" in result
    # User must be told nothing was dropped — the whole point of the
    # new behavior is no silent data loss.
    assert "No messages were dropped" in result
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_surfaces_aux_model_failure_even_when_recovered():
    """When the user's configured ``auxiliary.compression.model`` errors out
    but compression recovers by retrying on the main model, /compress must
    STILL inform the user.  Silent recovery hides broken config the user
    needs to fix."""
    history = _make_history()
    # Compressed transcript — normal successful compression, no placeholder.
    compressed = [
        history[0],
        {"role": "assistant", "content": "summary via main model"},
        history[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    # Fallback placeholder was NOT used — recovery succeeded.
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = None
    # But the configured aux model DID fail before the retry succeeded.
    agent_instance.context_compressor._last_aux_model_failure_model = (
        "gemini-3-flash-preview"
    )
    agent_instance.context_compressor._last_aux_model_failure_error = (
        "404 model not found: gemini-3-flash-preview"
    )
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    # Compression succeeded
    assert "Compressed:" in result
    # No ⚠️ warning (that's reserved for dropped-turns case)
    assert "⚠️" not in result
    # But there IS an info note about the broken aux model
    assert "ℹ️" in result
    assert "gemini-3-flash-preview" in result
    assert "404" in result
    assert "auxiliary.compression.model" in result
    # The user's context is explicitly called out as intact
    assert "intact" in result
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_passes_session_db_and_persists_rotated_session():
    """session_db must be wired into the /compress temp agent so that
    _compress_context can actually rotate the session and persist the
    compressed transcript — without it compression is a silent no-op."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    # a bare object(): the gateway wraps SessionDB in AsyncSessionDB and
    # unwraps via getattr(..., "_db", ...) when passing to the agent; a
    # MagicMock would auto-create ._db and break the identity assertion.
    runner._session_db = object()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = False
    agent_instance.session_id = "sess-1"

    def _compress(messages, *_args, **_kwargs):
        agent_instance.session_id = "sess-2"
        return compressed, ""

    agent_instance._compress_context.side_effect = _compress

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance) as mock_agent_cls,
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed:" in result
    assert mock_agent_cls.call_count == 2
    assert mock_agent_cls.call_args_list[0].kwargs["session_db"] is runner._session_db
    assert mock_agent_cls.call_args_list[1].kwargs["session_db"] is runner._session_db
    runner.session_store._save.assert_called_once()
    runner.session_store.rewrite_transcript.assert_called_once_with(
        "sess-2", compressed
    )
    runner.session_store.update_session.assert_called_once_with(
        build_session_key(_make_source()), last_prompt_tokens=0
    )
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_passes_tool_messages_to_compressor():
    """Manual /compress must use the same replayable transcript shape as
    automatic compression, including assistant tool_calls and tool results.
    """
    history = _make_tool_history()
    expected_replayable = [
        {"role": "user", "content": "run pwd"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": history[1]["tool_calls"],
        },
        {
            "role": "tool",
            "content": "/Users/zongxin/project",
            "tool_call_id": "call_pwd",
            "tool_name": "terminal",
        },
        {"role": "assistant", "content": "done"},
    ]
    compressed = [
        expected_replayable[0],
        {"role": "assistant", "content": "compressed summary"},
        expected_replayable[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.session_id = "sess-1"
    captured = {}

    def _compress(messages, *_args, **_kwargs):
        captured["messages"] = messages
        return compressed, ""

    agent_instance._compress_context.side_effect = _compress

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        await runner._handle_compress_command(_make_event())

    assert captured["messages"] == expected_replayable


@pytest.mark.asyncio
async def test_compress_command_does_not_rewrite_after_in_place_archive():
    """When _compress_context already wrote an in-place archive, gateway
    /compress must not call rewrite_transcript and destroy archived rows.
    """
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner._session_db = MagicMock()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.session_id = "sess-1"
    agent_instance._session_db = runner._session_db
    agent_instance._last_compaction_in_place = True
    agent_instance._compress_context.return_value = (compressed, "")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed:" in result
    runner.session_store.rewrite_transcript.assert_not_called()
    runner.session_store.update_session.assert_called_once_with(
        build_session_key(_make_source()), last_prompt_tokens=0
    )


@pytest.mark.asyncio
async def test_compress_here_in_place_persists_rejoined_tail_without_rewrite():
    """Default in-place /compress here must keep the verbatim tail in the
    active transcript without falling back to destructive rewrite_transcript.
    """
    history = _make_history()
    compressed_head = [
        history[0],
        {"role": "assistant", "content": "compressed head"},
    ]
    expected_rejoined = compressed_head + history[2:]
    runner = _make_runner(history)
    runner._session_db = MagicMock()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.session_id = "sess-1"
    agent_instance._session_db = runner._session_db
    agent_instance._last_compaction_in_place = True
    captured = {}

    def _compress(messages, *_args, **_kwargs):
        captured["messages"] = messages
        return compressed_head, ""

    agent_instance._compress_context.side_effect = _compress

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == expected_rejoined:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event("/compress here 1"))

    assert "Compressed:" in result
    # /compress here 1 compresses only the head before the last user turn.
    assert captured["messages"] == history[:2]
    runner.session_store.rewrite_transcript.assert_not_called()
    # The in-place compressor persisted compressed_head first; the gateway must
    # persist the final rejoined active transcript too, or the tail disappears
    # on the next DB-backed turn.
    runner._session_db.archive_and_compact.assert_called_once_with(
        "sess-1", expected_rejoined
    )
    runner.session_store.update_session.assert_called_once_with(
        build_session_key(_make_source()), last_prompt_tokens=0
    )


@pytest.mark.asyncio
async def test_compress_command_does_not_repoint_session_when_transcript_write_fails():
    """If the canonical transcript write fails after compression produces a new
    continuation session_id, /compress must NOT repoint the live session onto
    that empty session_id, and must report the failure instead of a success
    banner. Otherwise a transient DB/IO error during compression would silently
    drop the user's active conversation while still claiming success."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner._session_db = object()
    session_entry = runner.session_store.get_or_create_session.return_value
    # Simulate the canonical DB write failing (lock contention, ENOSPC, ...).
    runner.session_store.rewrite_transcript = MagicMock(return_value=False)
    # Telegram topic re-binding must never run on the failure path.
    runner._sync_telegram_topic_binding = MagicMock()

    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance._last_compaction_in_place = False
    agent_instance.session_id = "sess-1"

    def _compress(messages, *_args, **_kwargs):
        # Compression rotated the session: the agent now holds a NEW session_id.
        agent_instance.session_id = "sess-2"
        return compressed, ""

    agent_instance._compress_context.side_effect = _compress

    def _estimate(messages, **_kwargs):
        return 100

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    # The user sees a failure banner, not a success banner.
    assert "failed" in result.lower()
    assert "Compressed:" not in result
    # The live session was NOT repointed onto the empty new session_id, so the
    # original conversation stays reachable.
    assert session_entry.session_id == "sess-1"
    runner.session_store._save.assert_not_called()
    runner._sync_telegram_topic_binding.assert_not_called()
    # Resources are still cleaned up even though the command errored.
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()



@pytest.mark.asyncio
async def test_compress_command_preserves_platform_and_gateway_session_key():
    """The temporary compression agent must carry the originating source's
    platform and stable gateway session key, matching a normal gateway turn.
    Without them ``_session_source_for_agent`` falls back to a default "cli"
    host source, so an external context engine misattributes the retained
    transcript tail and later duplicates it on resume (#50422)."""
    history = _make_history()
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance) as mock_agent,
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        await runner._handle_compress_command(_make_event())

    assert mock_agent.call_count == 2
    first_kwargs = mock_agent.call_args_list[0].kwargs
    second_kwargs = mock_agent.call_args_list[1].kwargs
    # Platform preserved as the live turn's config key (TELEGRAM -> "telegram"),
    # not the unbound "cli"/"local" fallback.
    assert first_kwargs.get("platform") == "telegram"
    assert second_kwargs.get("platform") == "telegram"
    # Stable gateway session key preserved, identical to a normal gateway turn.
    assert first_kwargs.get("gateway_session_key") == runner._session_key_for_source(_make_source())
    assert second_kwargs.get("gateway_session_key") == runner._session_key_for_source(_make_source())
    assert first_kwargs["gateway_session_key"]
    assert first_kwargs["enabled_toolsets"] == ["memory"]
    assert second_kwargs["enabled_toolsets"] != ["memory"]
    assert "terminal" in second_kwargs["enabled_toolsets"]
    assert second_kwargs["skip_memory"] is True



@pytest.mark.asyncio
async def test_compress_command_overrides_stale_resolver_identity():
    """If the resolver already supplies platform/gateway_session_key, the
    construction must (a) not raise "got multiple values for keyword argument",
    and (b) let the originating-source identity win — a stale/placeholder
    resolver value must not defeat the attribution fix."""
    history = _make_history()
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")

    # Resolver injects a WRONG platform and a stale session key.
    runtime = {"api_key": "test-key", "platform": "discord", "gateway_session_key": "stale-key"}
    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance) as mock_agent,
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        await runner._handle_compress_command(_make_event())  # must not raise

    assert mock_agent.call_count == 2
    first_kwargs = mock_agent.call_args_list[0].kwargs
    second_kwargs = mock_agent.call_args_list[1].kwargs
    # Source-derived identity overrides the stale resolver values, passed once.
    assert first_kwargs["platform"] == "telegram"
    assert second_kwargs["platform"] == "telegram"
    assert first_kwargs["gateway_session_key"] == runner._session_key_for_source(_make_source())
    assert second_kwargs["gateway_session_key"] == runner._session_key_for_source(_make_source())

