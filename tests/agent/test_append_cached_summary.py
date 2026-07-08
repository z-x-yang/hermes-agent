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
from hermes_cli.config import DEFAULT_CONFIG


def test_append_cached_config_defaults_are_disabled_and_safe():
    cfg = DEFAULT_CONFIG["compression"]
    assert cfg["summary_call_mode"] == "serialized_prompt"
    assert cfg["append_cached_summary"] == {
        "source_scope": "compacted_prefix",
        "require_main_runtime": True,
        "allow_tool_choice_none": True,
        "fallback_to_serialized_prompt": True,
        "audit_sample_summary_chars": 12000,
    }


def test_append_cached_config_normalizes_invalid_values_to_safe_defaults():
    cfg = AppendCachedSummaryConfig.normalized({
        "source_scope": "full_history",
        "require_main_runtime": "yes",
        "allow_tool_choice_none": "0",
        "fallback_to_serialized_prompt": "false",
        "audit_sample_summary_chars": "not-an-int",
    })
    assert cfg.source_scope == "compacted_prefix"
    assert cfg.require_main_runtime is True
    assert cfg.allow_tool_choice_none is False
    assert cfg.fallback_to_serialized_prompt is False
    assert cfg.audit_sample_summary_chars == 12000


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
    runtime = ToolChoiceRejectRuntime(context_limit_tokens=1_000_000)
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
