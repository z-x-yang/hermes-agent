from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agent.context_compressor import (
    AppendCachedSummaryConfig,
    ContextCompressor,
    SummaryRules,
)
from agent.compression_summary_runtime import (
    apply_summary_tool_choice_none,
    extract_summary_cache_stats,
    extract_summary_response_content,
    make_summary_runtime,
)
from hermes_cli.config import DEFAULT_CONFIG


def test_append_cached_config_defaults_are_disabled_and_safe():
    cfg = DEFAULT_CONFIG["compression"]
    assert cfg["summary_call_mode"] == "serialized_prompt"
    assert cfg["append_cached_summary"] == {
        "source_scope": "compacted_prefix",
        "require_main_runtime": True,
        "allow_tool_choice_none": True,
        "fallback_to_serialized_prompt": True,
        "transport_retries": 1,
        "audit_sample_summary_chars": 12000,
    }


def test_append_cached_config_normalizes_invalid_values_to_safe_defaults():
    cfg = AppendCachedSummaryConfig.normalized({
        "source_scope": "full_history",
        "require_main_runtime": "yes",
        "allow_tool_choice_none": "0",
        "fallback_to_serialized_prompt": "false",
        "transport_retries": "not-an-int",
        "audit_sample_summary_chars": "not-an-int",
    })
    assert cfg.source_scope == "compacted_prefix"
    assert cfg.require_main_runtime is True
    assert cfg.allow_tool_choice_none is False
    assert cfg.fallback_to_serialized_prompt is False
    assert cfg.transport_retries == 1
    assert cfg.audit_sample_summary_chars == 12000


def test_codex_responses_keeps_tool_choice_auto_for_append_cached_summary():
    api_kwargs = {
        "model": "gpt-5.5",
        "input": [{"role": "user", "content": "summarize"}],
        "tools": [{"type": "function", "name": "noop", "parameters": {"type": "object"}}],
        "tool_choice": "auto",
    }

    updated, requested = apply_summary_tool_choice_none(api_kwargs, "codex_responses")

    assert requested is False
    assert updated["tool_choice"] == "auto"


def test_extract_summary_response_content_reads_codex_responses_output_text_items():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning", summary=[]),
            SimpleNamespace(
                type="message",
                role="assistant",
                content=[
                    SimpleNamespace(type="output_text", text="summary from responses item")
                ],
            ),
        ]
    )

    content, tool_call_violation = extract_summary_response_content(response)

    assert content == "summary from responses item"
    assert tool_call_violation is False


def test_extract_summary_response_content_flags_codex_responses_tool_calls():
    response = SimpleNamespace(
        output=[SimpleNamespace(type="function_call", name="terminal", arguments="{}")]
    )

    content, tool_call_violation = extract_summary_response_content(response)

    assert content == ""
    assert tool_call_violation is True


def test_extract_summary_cache_stats_treats_zero_cached_tokens_as_reported():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1000,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        )
    )

    stats = extract_summary_cache_stats(response)

    assert stats["reported"] is True
    assert stats["read_tokens"] == 0
    assert stats["hit_rate_estimate"] == 0.0


def test_extract_summary_cache_stats_treats_dict_zero_cached_tokens_as_reported():
    response = {
        "usage": {
            "prompt_tokens": 1000,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    }

    stats = extract_summary_cache_stats(response)

    assert stats["reported"] is True
    assert stats["read_tokens"] == 0
    assert stats["hit_rate_estimate"] == 0.0


class FakeCodexAgentForSummaryRuntime:
    provider = "openai-codex"
    model = "gpt-5.5"
    api_mode = "codex_responses"
    base_url = "https://chatgpt.com/backend-api/codex"
    reasoning_effort = None
    tools = [{"type": "function", "name": "noop"}]
    session_api_calls = 7

    class Compressor:
        context_length = 1_000_000

    context_compressor = Compressor()

    def __init__(self):
        self._ephemeral_max_output_tokens = None
        self.seen_ephemeral = None

    def _build_api_kwargs(self, messages):
        self.seen_ephemeral = self._ephemeral_max_output_tokens
        return {
            "model": self.model,
            "input": messages,
            "max_output_tokens": self._ephemeral_max_output_tokens,
        }

    @staticmethod
    def _sanitize_api_messages(messages):
        from agent.agent_runtime_helpers import sanitize_api_messages

        return sanitize_api_messages(messages)

    @staticmethod
    def _drop_thinking_only_and_merge_users(
        messages,
        *,
        drop_codex_reasoning_items=True,
    ):
        from agent.agent_runtime_helpers import drop_thinking_only_and_merge_users

        return drop_thinking_only_and_merge_users(
            messages,
            drop_codex_reasoning_items=drop_codex_reasoning_items,
        )


def test_make_summary_runtime_forwards_ephemeral_output_tokens_to_codex_build_kwargs():
    agent = FakeCodexAgentForSummaryRuntime()
    runtime = make_summary_runtime(agent)

    kwargs = runtime.build_kwargs([{"role": "user", "content": "summarize"}], 12345)

    assert agent.seen_ephemeral == 12345
    assert kwargs["max_output_tokens"] == 12345
    assert getattr(runtime, "main_api_calls_in_process") == 7


def test_make_summary_runtime_canonicalizes_tool_arguments_like_main_api_copy():
    agent = FakeCodexAgentForSummaryRuntime()
    runtime = make_summary_runtime(agent)
    raw_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "arguments": '{"z":2, "a":1}',
                    },
                }
            ],
        }
    ]

    kwargs = runtime.build_kwargs(raw_messages, 12345)

    arguments = kwargs["input"][0]["tool_calls"][0]["function"]["arguments"]
    assert arguments == '{"a":1,"z":2}'
    assert raw_messages[0]["tool_calls"][0]["function"]["arguments"] == '{"z":2, "a":1}'


def test_make_summary_runtime_sanitizer_does_not_mutate_source_tool_calls():
    agent = FakeCodexAgentForSummaryRuntime()
    runtime = make_summary_runtime(agent)
    raw_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_blank",
                    "type": "function",
                    "function": {"name": "", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_blank", "content": "ok"},
    ]

    kwargs = runtime.build_kwargs(raw_messages, 12345)

    assert kwargs["input"][0]["tool_calls"][0]["function"]["name"] == "invalid_tool_call"
    assert raw_messages[0]["tool_calls"][0]["function"]["name"] == ""


def test_make_summary_runtime_strips_content_like_main_api_copy():
    agent = FakeCodexAgentForSummaryRuntime()
    runtime = make_summary_runtime(agent)
    raw_messages = [{"role": "user", "content": "  summarize me  \n"}]

    kwargs = runtime.build_kwargs(raw_messages, 12345)

    assert kwargs["input"][0]["content"] == "summarize me"
    assert raw_messages[0]["content"] == "  summarize me  \n"


def test_make_summary_runtime_drops_thinking_only_turn_like_main_api_copy():
    agent = FakeCodexAgentForSummaryRuntime()
    runtime = make_summary_runtime(agent)
    raw_messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "", "reasoning_content": "hidden"},
        {"role": "user", "content": "second"},
    ]

    kwargs = runtime.build_kwargs(raw_messages, 12345)

    assert kwargs["input"] == [{"role": "user", "content": "first\n\nsecond"}]
    assert len(raw_messages) == 3


def test_make_summary_runtime_exposes_agent_fallback_activation():
    agent = FakeCodexAgentForSummaryRuntime()
    setattr(agent, "_fallback_chain", [{"provider": "openrouter", "model": "fallback/model"}])
    setattr(agent, "_fallback_index", 0)
    seen_reasons: list[Any] = []

    def _activate(reason: Any) -> bool:
        seen_reasons.append(reason)
        return True

    setattr(agent, "_try_activate_fallback", _activate)
    runtime = make_summary_runtime(agent)
    exc = RuntimeError("HTTP 429: primary quota exhausted")
    setattr(exc, "status_code", 429)

    assert runtime.fallback_attempt_budget == 1
    assert runtime.activate_fallback is not None
    assert runtime.activate_fallback(exc) is True
    assert seen_reasons


def test_context_compressor_accepts_summary_call_mode_without_changing_default_behavior():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test/model", quiet_mode=True)
    assert compressor.summary_call_mode == "serialized_prompt"
    assert compressor.append_cached_summary.source_scope == "compacted_prefix"
    assert compressor._summary_runtime_factory is None


def test_serialized_and_append_instruction_share_rules_hash():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test/model", quiet_mode=True)
    turns = [{"role": "user", "content": "remember USER_MARKER"}]
    budget = compressor._compute_summary_budget(turns)
    rules = compressor._build_summary_rules(turns, budget)
    serialized = compressor._build_serialized_summary_prompt(
        rules,
        "[user] remember USER_MARKER",
        focus_topic=None,
    )
    append_instruction = compressor._build_append_cached_summary_instruction(
        rules,
        previous_summary=None,
        focus_topic=None,
    )
    assert isinstance(rules, SummaryRules)
    assert rules.rules_hash.startswith("sha256:")
    assert "## All User Messages" in serialized
    assert "## All User Messages" in append_instruction
    assert rules.rules_hash == compressor._build_summary_rules(turns, budget).rules_hash


def test_append_instruction_does_not_embed_serialized_turns():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test/model", quiet_mode=True)
    turns = [{"role": "user", "content": "UNIQUE_SERIALIZED_HISTORY_MARKER"}]
    rules = compressor._build_summary_rules(turns, compressor._compute_summary_budget(turns))
    append_instruction = compressor._build_append_cached_summary_instruction(
        rules,
        previous_summary=None,
        focus_topic=None,
    )
    assert "TURNS TO SUMMARIZE" not in append_instruction
    assert "UNIQUE_SERIALIZED_HISTORY_MARKER" not in append_instruction


@dataclass
class CapturingRuntime:
    context_limit_tokens: int = 1_000_000
    provider: str = "openai-codex"
    model: str = "gpt-5.5"
    api_mode: str = "codex_responses"
    base_url: str = "https://chatgpt.com/backend-api/codex"
    reasoning_effort: str | None = "medium"
    tools_included: bool = True
    main_api_calls_in_process: int = 0
    captured_messages: list[dict[str, Any]] | None = None
    captured_kwargs: dict[str, Any] | None = None

    def build_kwargs(self, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        self.captured_messages = messages
        self.captured_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}],
            "max_tokens": max_tokens,
        }
        return dict(self.captured_kwargs)

    def invoke(self, api_kwargs: dict[str, Any]) -> Any:
        self.captured_kwargs = dict(api_kwargs)
        message = SimpleNamespace(
            content="## Primary Request and Intent\nUser asked to test append cache.\n\n## Key Technical Concepts\nNone.\n\n## Files and Code Sections\nNone.\n\n## Errors and Fixes\nNone.\n\n## Problem Solving\nNone.\n\n## All User Messages\n1. \"old prefix\" — User supplied source content.\n\n## Pending Tasks\nNone.\n\n## Current Work\nCompression summary generated.\n\n## Optional Next Step\nNone.",
            tool_calls=None,
        )
        usage = SimpleNamespace(prompt_tokens=500, prompt_tokens_details=SimpleNamespace(cached_tokens=300), cache_write_tokens=50)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    def estimate_request_tokens(self, api_kwargs: dict[str, Any]) -> int:
        return 500_000


@dataclass
class FlakyTransportRuntime(CapturingRuntime):
    fail_times: int = 1
    attempts: int = 0

    def invoke(self, api_kwargs: dict[str, Any]) -> Any:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise TimeoutError("simulated transient append-cached transport failure")
        return super().invoke(api_kwargs)


def test_append_cached_retries_retryable_transport_error_before_serialized_fallback():
    runtime = FlakyTransportRuntime(fail_times=1)
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": False, "transport_retries": 1},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)

    summary = compressor._generate_summary(
        [{"role": "user", "content": "old prefix"}],
        source_messages=[{"role": "user", "content": "old prefix"}],
        summarize_start=0,
        compress_end=1,
        focus_topic=None,
    )

    assert summary is not None
    assert runtime.attempts == 2
    audit = compressor._last_summary_call_audit
    assert audit["mode"] == "append_cached"
    assert audit["transport_retry_activated"] is True
    assert audit["transport_retry_attempts"][0]["error_type"] == "TimeoutError"
    assert audit["transport_retry_attempts"][0]["classification_reason"] == "timeout"


def test_append_cached_does_not_reembed_previous_summary_when_visible_in_cached_prefix():
    runtime = CapturingRuntime()
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": False},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    previous_summary = (
        "## Primary Request and Intent\n"
        "PREVIOUS_SUMMARY_MARKER_SHOULD_STAY_IN_CACHED_PREFIX_ONLY\n\n"
        "## All User Messages\n1. \"old ask\" — prior user request."
    )
    compressor._previous_summary = previous_summary
    messages = [
        {"role": "assistant", "content": f"[CONTEXT COMPACTION]\n{previous_summary}"},
        {"role": "user", "content": "new delta"},
        {"role": "assistant", "content": "TAIL_MARKER"},
    ]

    summary = compressor._generate_summary(
        messages[1:2],
        source_messages=messages,
        summarize_start=1,
        compress_end=2,
        focus_topic=None,
    )

    assert summary is not None
    assert runtime.captured_messages is not None
    instruction = runtime.captured_messages[-1]["content"]
    assert "PREVIOUS_SUMMARY_MARKER_SHOULD_STAY_IN_CACHED_PREFIX_ONLY" not in instruction
    assert "previous compaction summary already present" in instruction
    request_audit = compressor._last_summary_call_audit["request"]
    assert request_audit["previous_summary_in_cached_prefix"] is True
    assert request_audit["previous_summary_chars_in_instruction"] == 0


def test_append_cached_keeps_unrelated_legacy_previous_summary_cache_visible():
    runtime = CapturingRuntime()
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": False},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    previous_summary = (
        "## Primary Request and Intent\n"
        "LEGACY_PREVIOUS_SUMMARY_MUST_REMAIN_CACHE_VISIBLE\n\n"
        "## All User Messages\n"
        '1. "older compacted request" — prior user request.\n\n'
        "## Pending Tasks\nContinue."
    )
    compressor._previous_summary = previous_summary
    messages = [
        {"role": "assistant", "content": f"[CONTEXT COMPACTION]\n{previous_summary}"},
        {"role": "assistant", "content": "new delta to summarize"},
        {
            "role": "user",
            "content": "an unrelated retained-tail user message that is absent from the previous summary",
        },
    ]

    summary = compressor._generate_summary(
        messages[1:2],
        source_messages=messages,
        summarize_start=1,
        compress_end=2,
        focus_topic=None,
    )

    assert summary is not None
    assert runtime.captured_messages is not None
    instruction = runtime.captured_messages[-1]["content"]
    assert "LEGACY_PREVIOUS_SUMMARY_MUST_REMAIN_CACHE_VISIBLE" not in instruction
    request_audit = compressor._last_summary_call_audit["request"]
    assert request_audit["previous_summary_in_cached_prefix"] is True
    assert request_audit["previous_summary_chars_in_instruction"] == 0


def test_append_cached_request_uses_prefix_to_compress_end_and_excludes_tail_marker():
    runtime = CapturingRuntime()
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": False},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old prefix"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "TAIL_MARKER_SHOULD_NOT_BE_SENT"},
    ]
    summary = compressor._generate_summary(
        messages[1:3],
        source_messages=messages,
        summarize_start=1,
        compress_end=3,
        focus_topic=None,
    )
    assert summary is not None
    assert runtime.captured_messages is not None
    assert runtime.captured_messages[:-1] == messages[:3]
    assert runtime.captured_messages[-1]["role"] == "user"
    assert "TAIL_MARKER_SHOULD_NOT_BE_SENT" not in runtime.captured_messages[-1]["content"]
    assert "TURNS TO SUMMARIZE" not in runtime.captured_messages[-1]["content"]
    request_audit = compressor._last_summary_call_audit["request"]
    assert request_audit["retained_tail_excluded"] is True
    assert request_audit["rough_tokens_estimate"] == request_audit["tokens_estimate"]
    assert request_audit["request_shape_estimate_tokens"] == request_audit["tokens_estimate"]


def test_append_cached_uses_runtime_context_limit_not_threshold_tokens():
    runtime = CapturingRuntime(context_limit_tokens=1_000_000)
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            threshold_percent=0.272,
            quiet_mode=True,
            summary_call_mode="append_cached",
        )
    compressor.threshold_tokens = 272_000
    compressor.bind_summary_runtime_factory(lambda: runtime)
    summary = compressor._generate_summary(
        [{"role": "user", "content": "old prefix"}],
        source_messages=[{"role": "user", "content": "old prefix"}],
        summarize_start=0,
        compress_end=1,
        focus_topic=None,
    )
    assert summary is not None
    assert compressor._last_summary_call_audit["request"]["runtime_context_limit_tokens"] == 1_000_000
    assert compressor._last_summary_call_audit["fallback_reason"] is None


def test_runtime_context_and_internal_compression_window_are_decoupled():
    runtime = CapturingRuntime(context_limit_tokens=1_000_000)
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            threshold_percent=0.95,
            summary_target_ratio=(20_000 / 272_000),
            compression_context_length=272_000,
            quiet_mode=True,
            summary_call_mode="append_cached",
        )

    assert compressor.context_length == 1_000_000
    assert compressor.compression_context_length == 272_000
    assert compressor.threshold_tokens == int(272_000 * 0.95)
    assert compressor.tail_token_budget == 20_000
    status = compressor.get_status()
    assert status["context_length"] == 272_000
    assert status["compression_context_length"] == 272_000
    assert status["runtime_context_length"] == 1_000_000

    compressor.update_model("gpt-5.6", context_length=1_500_000)
    assert compressor.context_length == 1_500_000
    assert compressor.compression_context_length == 272_000
    assert compressor.threshold_tokens == int(272_000 * 0.95)
    assert compressor.tail_token_budget == 20_000

    compressor.update_model("small-runtime", context_length=200_000)
    assert compressor.context_length == 200_000
    assert compressor.compression_context_length == 200_000
    assert compressor.threshold_tokens == int(200_000 * 0.95)

    compressor.update_model("gpt-5.6", context_length=1_500_000)
    assert compressor.context_length == 1_500_000
    assert compressor.compression_context_length == 272_000

    compressor.update_model("gpt-5.6", context_length=1_000_000, compression_context_length=272_000)
    assert compressor.context_length == 1_000_000
    assert compressor.compression_context_length == 272_000

    compressor.bind_summary_runtime_factory(lambda: runtime)
    summary = compressor._generate_summary(
        [{"role": "user", "content": "old prefix"}],
        source_messages=[{"role": "user", "content": "old prefix"}],
        summarize_start=0,
        compress_end=1,
        focus_topic=None,
    )

    assert summary is not None
    request_audit = compressor._last_summary_call_audit["request"]
    assert request_audit["runtime_context_limit_tokens"] == 1_000_000
    assert compressor._last_summary_call_audit["fallback_reason"] is None


class ToolCallRuntime(CapturingRuntime):
    def invoke(self, api_kwargs: dict[str, Any]) -> Any:
        message = SimpleNamespace(content="", tool_calls=[SimpleNamespace(id="call_1")])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class OverflowRuntime(CapturingRuntime):
    def estimate_request_tokens(self, api_kwargs: dict[str, Any]) -> int:
        return 2_000_000


class ToolChoiceRejectRuntime(CapturingRuntime):
    def invoke(self, api_kwargs: dict[str, Any]) -> Any:
        raise RuntimeError("unsupported parameter: tool_choice")


class FailingFallbackActivatingRuntime(CapturingRuntime):
    provider: str = "openai-codex"
    model: str = "gpt-5.5"
    api_mode: str = "codex_responses"
    fallback_attempt_budget: int = 1

    def __init__(self, **kwargs: Any) -> None:
        fallback_attempt_budget = int(kwargs.pop("fallback_attempt_budget", self.fallback_attempt_budget))
        super().__init__(**kwargs)
        self.fallback_attempt_budget = fallback_attempt_budget
        self.activate_calls: list[BaseException] = []

    def invoke(self, api_kwargs: dict[str, Any]) -> Any:
        exc = RuntimeError("HTTP 429: primary quota exhausted")
        setattr(exc, "status_code", 429)
        raise exc

    def activate_fallback(self, exc: BaseException) -> bool:
        self.activate_calls.append(exc)
        return True


class FallbackSummaryRuntime(CapturingRuntime):
    provider: str = "openrouter"
    model: str = "fallback/model"
    api_mode: str = "chat_completions"

    def invoke(self, api_kwargs: dict[str, Any]) -> Any:
        self.captured_kwargs = dict(api_kwargs)
        message = SimpleNamespace(
            content="## Primary Request and Intent\nFallback summary worked.\n\n## Key Technical Concepts\nNone.\n\n## Files and Code Sections\nNone.\n\n## Errors and Fixes\nNone.\n\n## Problem Solving\nNone.\n\n## All User Messages\n1. \"old prefix\" — User supplied source content.\n\n## Pending Tasks\nNone.\n\n## Current Work\nFallback provider generated the append-cached summary.\n\n## Optional Next Step\nNone.",
            tool_calls=None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def test_append_cached_rejects_tool_call_response_without_silent_success():
    runtime = ToolCallRuntime()
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": False},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    summary = compressor._generate_summary(
        [{"role": "user", "content": "old prefix"}],
        source_messages=[{"role": "user", "content": "old prefix"}],
        summarize_start=0,
        compress_end=1,
        focus_topic=None,
    )
    assert summary is None
    assert compressor._last_summary_call_audit["tool_call_violation"] is True
    assert compressor._last_summary_call_audit["fallback_reason"] == "summary_returned_tool_call"


def test_append_cached_context_overflow_records_fallback_reason():
    runtime = OverflowRuntime(context_limit_tokens=1_000_000)
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": False},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    summary = compressor._generate_summary(
        [{"role": "user", "content": "old prefix"}],
        source_messages=[{"role": "user", "content": "old prefix"}],
        summarize_start=0,
        compress_end=1,
        focus_topic=None,
    )
    assert summary is None
    assert compressor._last_summary_call_audit["fallback_reason"] == "append_cached_context_overflow"


def test_append_cached_tool_choice_rejection_has_specific_reason():
    runtime = ToolChoiceRejectRuntime(context_limit_tokens=1_000_000, api_mode="chat_completions")
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": False},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    summary = compressor._generate_summary(
        [{"role": "user", "content": "old prefix"}],
        source_messages=[{"role": "user", "content": "old prefix"}],
        summarize_start=0,
        compress_end=1,
        focus_topic=None,
    )
    assert summary is None
    assert compressor._last_summary_call_audit["fallback_reason"] == "provider_rejected_tool_choice_none"


def test_append_cached_transport_error_tries_provider_fallback_before_serialized_prompt():
    primary_runtime = FailingFallbackActivatingRuntime()
    fallback_runtime = FallbackSummaryRuntime(
        provider="openrouter",
        model="fallback/model",
        api_mode="chat_completions",
    )
    runtimes = [primary_runtime, fallback_runtime]
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": True},
        )
    compressor.bind_summary_runtime_factory(lambda: runtimes.pop(0))

    with patch(
        "agent.context_compressor.call_llm",
        side_effect=AssertionError("serialized prompt fallback should not run"),
    ) as serialized_call:
        summary = compressor._generate_summary(
            [{"role": "user", "content": "old prefix"}],
            source_messages=[{"role": "user", "content": "old prefix"}],
            summarize_start=0,
            compress_end=1,
            focus_topic=None,
        )

    assert summary is not None
    assert "Fallback summary worked" in summary
    assert primary_runtime.activate_calls
    assert serialized_call.call_count == 0
    assert fallback_runtime.captured_messages is not None
    assert compressor._last_summary_call_audit["mode"] == "append_cached"
    assert compressor._last_summary_call_audit["cache_key_runtime"]["provider"] == "openrouter"


def test_append_cached_provider_fallback_honors_full_configured_chain():
    failing_runtimes = [
        FailingFallbackActivatingRuntime(fallback_attempt_budget=9)
        for _ in range(9)
    ]
    fallback_runtime = FallbackSummaryRuntime(
        provider="openrouter",
        model="ninth-fallback/model",
        api_mode="chat_completions",
    )
    runtimes = [*failing_runtimes, fallback_runtime]
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": True},
        )
    compressor.bind_summary_runtime_factory(lambda: runtimes.pop(0))

    with patch(
        "agent.context_compressor.call_llm",
        side_effect=AssertionError("serialized prompt fallback should not run"),
    ):
        summary = compressor._generate_summary(
            [{"role": "user", "content": "old prefix"}],
            source_messages=[{"role": "user", "content": "old prefix"}],
            summarize_start=0,
            compress_end=1,
            focus_topic=None,
        )

    assert summary is not None
    assert "Fallback summary worked" in summary
    assert sum(len(runtime.activate_calls) for runtime in failing_runtimes) == 9
    assert fallback_runtime.captured_messages is not None
    assert compressor._last_summary_call_audit["cache_key_runtime"]["model"] == "ninth-fallback/model"


def test_compression_audit_contains_summary_call_without_summary_text(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runtime = CapturingRuntime()
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            protect_first_n=0,
            protect_last_n=1,
            summary_target_ratio=0.0,
            quiet_mode=True,
            summary_call_mode="append_cached",
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "middle user"},
        {"role": "assistant", "content": "middle assistant"},
        {"role": "user", "content": "latest tail"},
    ]
    out = compressor.compress(messages, current_tokens=900_000, force=True)
    assert len(out) <= len(messages) + 1
    audit_path = tmp_path / "logs" / "compression_audit.jsonl"
    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    record = records[-1]
    assert record["summary_call"]["mode"] == "append_cached"
    assert record["summary_call"]["cache_key_runtime"]["main_api_calls_in_process"] == 0
    serialized = json.dumps(record, ensure_ascii=False)
    assert "old user" not in serialized
    assert "latest tail" not in serialized
    assert "Primary Request and Intent" not in serialized


def test_summary_sample_sidecar_records_redacted_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runtime = CapturingRuntime()
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            protect_first_n=0,
            protect_last_n=1,
            summary_target_ratio=0.0,
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"audit_sample_summary_chars": 2000},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    compressor.compress([
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "middle user"},
        {"role": "assistant", "content": "middle assistant"},
        {"role": "user", "content": "tail"},
    ], current_tokens=900_000, force=True)
    sample_path = tmp_path / "logs" / "compression_summary_samples.jsonl"
    samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
    sample = samples[-1]
    assert sample["event"] == "compression_summary_sample"
    assert sample["summary_call_mode"] == "append_cached"
    assert sample["section_check"]["has_all_canonical_sections"] is True
    assert sample["summary_excerpt"]
    assert "compression_id" in sample


def test_append_cached_hard_failure_aborts_compress_without_static_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runtime = OverflowRuntime(context_limit_tokens=1_000_000)
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            protect_first_n=0,
            protect_last_n=1,
            summary_target_ratio=0.0,
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": False},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "middle user"},
        {"role": "assistant", "content": "middle assistant"},
        {"role": "user", "content": "latest tail"},
    ]

    out = compressor.compress(messages, current_tokens=900_000, force=True)

    assert out == messages
    audit_path = tmp_path / "logs" / "compression_audit.jsonl"
    record = [json.loads(line) for line in audit_path.read_text().splitlines()][-1]
    assert record["result"] == "abort"
    assert record["abort_reason"] == "append_cached_context_overflow"
    assert record["summary_call"]["fallback_reason"] == "append_cached_context_overflow"


def test_append_cached_tool_call_failure_aborts_compress_without_static_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runtime = ToolCallRuntime()
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            protect_first_n=0,
            protect_last_n=1,
            summary_target_ratio=0.0,
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"fallback_to_serialized_prompt": False},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "middle user"},
        {"role": "assistant", "content": "middle assistant"},
        {"role": "user", "content": "latest tail"},
    ]

    out = compressor.compress(messages, current_tokens=900_000, force=True)

    assert out == messages
    audit_path = tmp_path / "logs" / "compression_audit.jsonl"
    record = [json.loads(line) for line in audit_path.read_text().splitlines()][-1]
    assert record["result"] == "abort"
    assert record["abort_reason"] == "summary_returned_tool_call"
    assert record["summary_call"]["tool_call_violation"] is True
