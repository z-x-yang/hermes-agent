# Compression Append-Cached Summary Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a config-gated `append_cached` compression summary-generation path that reuses the main-session provider-visible prefix, keeps Hermes' existing summary rules, and adds cache/quality audit tooling for review and debugging.

**Architecture:** Keep the existing serialized-prompt path as the default and fallback. Add a small runtime bridge from `AIAgent` to `ContextCompressor` so append-cached summary calls reuse the active provider payload builder and main runtime API caller. Split durable audit into content-free `compression_audit.jsonl` metadata plus a redacted summary-quality sidecar.

**Tech Stack:** Python 3, Hermes Agent, pytest, JSONL audit logs, existing `agent.chat_completion_helpers`, existing `agent.context_compressor` compression assembly.

## Global Constraints

- Do not change the existing nine-section compression summary rules except for refactoring the same text into reusable builders.
- `compression.summary_call_mode` defaults to `serialized_prompt`; append-cached runs only when explicitly configured.
- v1 source binding is `provider_payload_prefix_to_compress_end`; retained tail is not included in the summary request source.
- Summary request fitting uses the summary runtime's real context limit, not `compression.threshold_tokens`.
- Do not delete the tools array to block tools; use provider-appropriate `tool_choice` when enabled and always reject responses containing tool calls.
- Main `compression_audit.jsonl` remains content-free: no message text, raw tool output, tool args, summary body, or user text.
- Summary-quality sidecar may include redacted summary text up to a cap and structure checks.
- Auth/network summary failures remain fail-closed and preserve the transcript.
- Add focused RED/GREEN tests before implementation changes for each task.
- Commit each task separately; stage only touched files.

---

## File Structure

- Modify `hermes_cli/config.py`
  - Add config defaults for `compression.summary_call_mode` and `compression.append_cached_summary`.
- Modify `agent/agent_init.py`
  - Normalize new compression config and pass it to `ContextCompressor`.
  - Bind a main-runtime summary bridge to the compressor after `agent.context_compressor` is created.
- Modify `agent/context_compressor.py`
  - Add config/dataclass state for summary call mode.
  - Refactor current `_generate_summary()` prompt construction into reusable summary rule and prompt builders without changing the rule text.
  - Add append-cached summary call path, fallback bookkeeping, audit fields, and redacted summary sidecar writer.
- Create `agent/compression_summary_runtime.py`
  - Focused bridge between `AIAgent` and compressor: build main runtime kwargs, apply tool suppression, invoke the existing non-streaming API helper, extract response text/tool-call violations/cache stats, and estimate request size.
- Create `scripts/compression_audit_report.py`
  - Read-only operator/debug report for compression cache hit health, fallback reasons, summary structure, token savings, and persistence row ids.
- Create `tests/agent/test_append_cached_summary.py`
  - Focused unit tests for config normalization, prompt/rules hash, append-cached request shape, context-limit split, tool-call rejection, and audit/sidecar fields.
- Create `tests/scripts/test_compression_audit_report.py`
  - Focused tests for report parsing and no-content leakage.

---

### Task 1: Config plumbing and runtime bridge scaffolding

**Files:**
- Modify: `hermes_cli/config.py:1339-1409`
- Modify: `agent/agent_init.py:1438-1742`
- Modify: `agent/context_compressor.py:1363-1517`
- Create: `agent/compression_summary_runtime.py`
- Test: `tests/agent/test_append_cached_summary.py`

**Interfaces:**
- Produces: `AppendCachedSummaryConfig.normalized(raw: Any) -> AppendCachedSummaryConfig`
- Produces: `SummaryCallMode = Literal["serialized_prompt", "append_cached"]`
- Produces: `SummaryRuntime` with methods `build_kwargs(messages, max_tokens)`, `invoke(api_kwargs)`, `estimate_request_tokens(api_kwargs)`, and `context_limit_tokens` metadata.
- Produces: `make_summary_runtime(agent) -> SummaryRuntime`
- Consumes: existing `agent._build_api_kwargs(api_messages)` and `agent.chat_completion_helpers.interruptible_api_call(agent, api_kwargs)`.

- [ ] **Step 1: Write failing config/default tests**

Create `tests/agent/test_append_cached_summary.py` with this initial content:

```python
from __future__ import annotations

from unittest.mock import patch

from agent.context_compressor import (
    AppendCachedSummaryConfig,
    ContextCompressor,
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
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest tests/agent/test_append_cached_summary.py -q
```

Expected result: fails because `AppendCachedSummaryConfig`, `summary_call_mode`, and `append_cached_summary` do not exist yet.

- [ ] **Step 3: Add config defaults**

In `hermes_cli/config.py`, inside `DEFAULT_CONFIG["compression"]` after `abort_on_summary_failure`, add:

```python
        "summary_call_mode": "serialized_prompt",  # serialized_prompt|append_cached.
                                      # append_cached reuses the main runtime
                                      # provider-visible prefix for the one-off
                                      # summary-generation call, then appends a
                                      # compression instruction as the final
                                      # user message. Keep serialized_prompt as
                                      # default until canary data is good.
        "append_cached_summary": {
            "source_scope": "compacted_prefix",
            "require_main_runtime": True,
            "allow_tool_choice_none": True,
            "fallback_to_serialized_prompt": True,
            "audit_sample_summary_chars": 12000,
        },
```

- [ ] **Step 4: Add config dataclass to `agent/context_compressor.py`**

Near the existing config dataclasses, add:

```python
@dataclass(frozen=True)
class AppendCachedSummaryConfig:
    source_scope: str = "compacted_prefix"
    require_main_runtime: bool = True
    allow_tool_choice_none: bool = True
    fallback_to_serialized_prompt: bool = True
    audit_sample_summary_chars: int = 12000

    @classmethod
    def normalized(cls, raw: Any) -> "AppendCachedSummaryConfig":
        if not isinstance(raw, dict):
            raw = {}

        source_scope = str(raw.get("source_scope", "compacted_prefix") or "").strip().lower()
        if source_scope != "compacted_prefix":
            source_scope = "compacted_prefix"

        def _bool(name: str, default: bool) -> bool:
            value = raw.get(name, default)
            if isinstance(value, bool):
                return value
            if value is None:
                return default
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            return default

        try:
            sample_chars = int(raw.get("audit_sample_summary_chars", 12000))
        except (TypeError, ValueError):
            sample_chars = 12000
        sample_chars = max(0, min(sample_chars, 100000))

        return cls(
            source_scope=source_scope,
            require_main_runtime=_bool("require_main_runtime", True),
            allow_tool_choice_none=_bool("allow_tool_choice_none", True),
            fallback_to_serialized_prompt=_bool("fallback_to_serialized_prompt", True),
            audit_sample_summary_chars=sample_chars,
        )
```

- [ ] **Step 5: Extend `ContextCompressor.__init__`**

Add parameters after `cheap_tool_result_cleanup`:

```python
        summary_call_mode: str = "serialized_prompt",
        append_cached_summary: AppendCachedSummaryConfig | dict[str, Any] | None = None,
```

Inside `__init__`, after `self.cheap_tool_result_cleanup` normalization, add:

```python
        mode = str(summary_call_mode or "serialized_prompt").strip().lower()
        if mode not in {"serialized_prompt", "append_cached"}:
            mode = "serialized_prompt"
        self.summary_call_mode = mode
        self.append_cached_summary = AppendCachedSummaryConfig.normalized(
            append_cached_summary
        )
        self._summary_runtime_factory: Any = None
        self._last_summary_call_audit: dict[str, Any] = {}
        self._last_summary_sample: dict[str, Any] | None = None
```

Add a small binder method near session-state helpers:

```python
    def bind_summary_runtime_factory(self, factory: Any) -> None:
        """Bind a request-local main-runtime bridge used by append_cached summaries."""
        self._summary_runtime_factory = factory
```

- [ ] **Step 6: Create runtime bridge file**

Create `agent/compression_summary_runtime.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SummaryRuntime:
    provider: str
    model: str
    api_mode: str
    base_url: str
    reasoning_effort: str | None
    context_limit_tokens: int | None
    tools_included: bool
    build_kwargs: Callable[[list[dict[str, Any]], int], dict[str, Any]]
    invoke: Callable[[dict[str, Any]], Any]
    estimate_request_tokens: Callable[[dict[str, Any]], int]


def make_summary_runtime(agent: Any) -> SummaryRuntime:
    """Return a lightweight main-runtime bridge for compression summary calls."""
    from agent.chat_completion_helpers import (
        estimate_request_context_tokens,
        interruptible_api_call,
    )

    def _build_kwargs(messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        old_ephemeral = getattr(agent, "_ephemeral_max_output_tokens", None)
        try:
            agent._ephemeral_max_output_tokens = max_tokens
            return agent._build_api_kwargs(messages)
        finally:
            agent._ephemeral_max_output_tokens = old_ephemeral

    def _invoke(api_kwargs: dict[str, Any]) -> Any:
        return interruptible_api_call(agent, api_kwargs)

    return SummaryRuntime(
        provider=getattr(agent, "provider", "") or "",
        model=getattr(agent, "model", "") or "",
        api_mode=getattr(agent, "api_mode", "") or "",
        base_url=getattr(agent, "base_url", "") or "",
        reasoning_effort=getattr(agent, "reasoning_effort", None),
        context_limit_tokens=getattr(getattr(agent, "context_compressor", None), "context_length", None),
        tools_included=bool(getattr(agent, "tools", None)),
        build_kwargs=_build_kwargs,
        invoke=_invoke,
        estimate_request_tokens=estimate_request_context_tokens,
    )
```

- [ ] **Step 7: Wire config in `agent/agent_init.py`**

After `cheap_tool_result_cleanup_cfg` normalization, add:

```python
    compression_summary_call_mode = str(
        _compression_cfg.get("summary_call_mode", "serialized_prompt") or "serialized_prompt"
    ).strip().lower()
    if compression_summary_call_mode not in {"serialized_prompt", "append_cached"}:
        _ra().logger.warning(
            "Invalid compression.summary_call_mode=%r — using 'serialized_prompt'",
            _compression_cfg.get("summary_call_mode"),
        )
        compression_summary_call_mode = "serialized_prompt"

    append_cached_summary_cfg = _compression_cfg.get("append_cached_summary", {})
    if not isinstance(append_cached_summary_cfg, dict):
        append_cached_summary_cfg = {}
```

Pass into the `ContextCompressor` constructor call:

```python
            summary_call_mode=compression_summary_call_mode,
            append_cached_summary=append_cached_summary_cfg,
```

After the `_bind_session_state` call, bind the runtime factory:

```python
    _bind_summary_runtime_factory = getattr(
        agent.context_compressor, "bind_summary_runtime_factory", None
    )
    if callable(_bind_summary_runtime_factory):
        try:
            from agent.compression_summary_runtime import make_summary_runtime
            _bind_summary_runtime_factory(lambda: make_summary_runtime(agent))
        except Exception:
            pass
```

- [ ] **Step 8: Run Task 1 tests and commit**

Run:

```bash
pytest tests/agent/test_append_cached_summary.py -q
python -m py_compile agent/context_compressor.py agent/agent_init.py agent/compression_summary_runtime.py
```

Expected result: tests pass and py_compile exits 0.

Commit:

```bash
git add hermes_cli/config.py agent/agent_init.py agent/context_compressor.py agent/compression_summary_runtime.py tests/agent/test_append_cached_summary.py
git commit -m "feat: add append-cached summary config scaffolding"
```

---

### Task 2: Extract summary rule builders without behavior change

**Files:**
- Modify: `agent/context_compressor.py:3496-3687`
- Test: `tests/agent/test_append_cached_summary.py`
- Existing tests to keep green: `tests/agent/test_context_compressor.py::TestCompress::test_summary_prompt_uses_nine_section_continuation_structure`

**Interfaces:**
- Produces: `SummaryRules` dataclass with fields `preamble`, `minimal_sufficient_state_rule`, `template_sections`, `summary_budget`, `rules_hash`.
- Produces: `_build_summary_rules(turns_to_summarize, summary_budget) -> SummaryRules`
- Produces: `_build_serialized_summary_prompt(rules, content_to_summarize, focus_topic) -> str`
- Produces: `_build_append_cached_summary_instruction(rules, previous_summary, focus_topic) -> str`

- [ ] **Step 1: Add failing tests for rules hash and no history duplication**

Append to `tests/agent/test_append_cached_summary.py`:

```python
from agent.context_compressor import SummaryRules


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
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest tests/agent/test_append_cached_summary.py::test_serialized_and_append_instruction_share_rules_hash tests/agent/test_append_cached_summary.py::test_append_instruction_does_not_embed_serialized_turns -q
```

Expected result: fails because `SummaryRules` and builder methods do not exist.

- [ ] **Step 3: Add `SummaryRules` dataclass and rules hash helper**

In `agent/context_compressor.py`, add near `AppendCachedSummaryConfig`:

```python
@dataclass(frozen=True)
class SummaryRules:
    preamble: str
    minimal_sufficient_state_rule: str
    template_sections: str
    summary_budget: int
    rules_hash: str


def _hash_summary_rules(*parts: str) -> str:
    payload = "\n\n".join(parts).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()
```

Add `import hashlib` at the top of the file if absent.

- [ ] **Step 4: Extract prompt text without editing rule content**

In `ContextCompressor`, create `_build_summary_rules()` by moving these current `_generate_summary()` blocks unchanged:

- date resolution and `_temporal_anchoring_rule`
- `_summarizer_preamble`
- `_minimal_sufficient_state_rule`
- `_template_sections`

The new method must return:

```python
        return SummaryRules(
            preamble=_summarizer_preamble,
            minimal_sufficient_state_rule=_minimal_sufficient_state_rule,
            template_sections=_template_sections,
            summary_budget=summary_budget,
            rules_hash=_hash_summary_rules(
                _summarizer_preamble,
                _minimal_sufficient_state_rule,
                _template_sections,
            ),
        )
```

Do not edit the literal wording inside the moved strings. This preserves the current summary rules and makes the hash a transport-independent guard.

- [ ] **Step 5: Add serialized prompt builder**

Add this method below `_build_summary_rules()`:

```python
    def _build_serialized_summary_prompt(
        self,
        rules: SummaryRules,
        content_to_summarize: str,
        focus_topic: Optional[str] = None,
    ) -> str:
        if self._previous_summary:
            previous_summary_for_prompt = self._previous_summary
            prompt = f"""{rules.preamble}

You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{previous_summary_for_prompt}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

Role=user messages in NEW TURNS TO INCORPORATE are authoritative over PREVIOUS SUMMARY. If they conflict, preserve the newer user state in active sections.

{rules.minimal_sufficient_state_rule}

Update the summary using this exact nine-section structure. Incorporate the new turns, and remove or mark obsolete information when messages inside the summarized slice cancelled, narrowed, or replaced earlier work. Keep "## Pending Tasks" limited to genuinely open work visible from the previous summary plus the new summarized turns. Update "## Current Work" and "## Optional Next Step" to reflect the precise continuation point at the compression boundary. Do not preserve completed or cancelled work as pending. In "## All User Messages", carry forward every entry from the previous summary and append the new turns' real user messages; never drop, merge, or paraphrase away a user's wording.

{rules.template_sections}"""
        else:
            prompt = f"""{rules.preamble}

Create a structured checkpoint summary for the conversation after earlier turns are compacted. The summary should preserve enough detail for continuity without re-reading the original turns.

{rules.minimal_sufficient_state_rule}

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

{rules.template_sections}"""

        return self._append_focus_topic_guidance(prompt, focus_topic)
```

- [ ] **Step 6: Add focus guidance helper**

Move the existing focus-topic append block into:

```python
    def _append_focus_topic_guidance(self, prompt: str, focus_topic: Optional[str]) -> str:
        if not focus_topic:
            return prompt
        return prompt + f"""

FOCUS TOPIC: "{focus_topic}"
This compaction should PRIORITISE preserving all information related to the focus topic above. For content related to "{focus_topic}", include full detail — exact values, file paths, command outputs, error messages, and decisions. For content NOT related to "{focus_topic}", summarise more aggressively (brief one-liners or omit if truly irrelevant). The focus topic sections should receive roughly 60-70% of the summary token budget. Even for the focus topic, NEVER preserve API keys, tokens, passwords, or credentials — use [REDACTED]."""
```

- [ ] **Step 7: Add append instruction builder**

Add:

```python
    def _build_append_cached_summary_instruction(
        self,
        rules: SummaryRules,
        previous_summary: Optional[str],
        focus_topic: Optional[str] = None,
    ) -> str:
        if previous_summary:
            prompt = f"""{rules.preamble}

You are updating a context compaction summary. The conversation messages above are the provider-visible compacted prefix that will be replaced by this summary. The retained tail is not included in this request and will remain verbatim after the summary.

PREVIOUS SUMMARY:
{previous_summary}

Role=user messages in the conversation above are authoritative over PREVIOUS SUMMARY. If they conflict, preserve the newer user state in active sections.

{rules.minimal_sufficient_state_rule}

Update the summary using this exact nine-section structure. Incorporate the conversation above, and remove or mark obsolete information when messages inside the summarized slice cancelled, narrowed, or replaced earlier work. Keep "## Pending Tasks" limited to genuinely open work visible from the previous summary plus the conversation above. Update "## Current Work" and "## Optional Next Step" to reflect the precise continuation point at the compression boundary. Do not preserve completed or cancelled work as pending. In "## All User Messages", carry forward every entry from the previous summary and append the conversation's real user messages; never drop, merge, or paraphrase away a user's wording.

{rules.template_sections}"""
        else:
            prompt = f"""{rules.preamble}

Create a structured checkpoint summary for the conversation messages above. Those messages are the provider-visible compacted prefix that will be replaced by this summary. The retained tail is not included in this request and will remain verbatim after the summary.

{rules.minimal_sufficient_state_rule}

Use this exact structure:

{rules.template_sections}"""

        return self._append_focus_topic_guidance(prompt, focus_topic)
```

- [ ] **Step 8: Rewire existing `_generate_summary()` to use serialized builder**

Replace the inline prompt construction in `_generate_summary()` with:

```python
        summary_budget = self._compute_summary_budget(turns_to_summarize)
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)
        rules = self._build_summary_rules(turns_to_summarize, summary_budget)
        prompt = self._build_serialized_summary_prompt(
            rules,
            content_to_summarize,
            focus_topic=focus_topic,
        )
        self._last_summary_call_audit = {
            "mode": "serialized_prompt",
            "source_binding": "serialized_turns_to_summarize",
            "rules_hash": rules.rules_hash,
            "cache_eligible": False,
            "fallback_reason": None,
            "tool_call_violation": False,
        }
```

Keep the existing `call_llm` invocation, response normalization, summary normalization, ground-truth sidecar prep, and error handling below this point unchanged.

- [ ] **Step 9: Run focused prompt tests and commit**

Run:

```bash
pytest tests/agent/test_append_cached_summary.py tests/agent/test_context_compressor.py::TestCompress::test_summary_prompt_uses_nine_section_continuation_structure tests/agent/test_context_compressor.py::TestCompress::test_summary_prompt_forbids_attributing_assistant_text_to_user -q
python -m py_compile agent/context_compressor.py
```

Expected result: all pass.

Commit:

```bash
git add agent/context_compressor.py tests/agent/test_append_cached_summary.py
git commit -m "refactor: separate compression summary rules from transport"
```

---

### Task 3: Implement append-cached summary call path

**Files:**
- Modify: `agent/context_compressor.py:3496-3880` and `agent/context_compressor.py:4828-4832`
- Modify: `agent/compression_summary_runtime.py`
- Test: `tests/agent/test_append_cached_summary.py`

**Interfaces:**
- Consumes: `SummaryRuntime` from Task 1.
- Produces: `_generate_summary_append_cached(source_messages, turns_to_summarize, summarize_start, compress_end, focus_topic) -> str | None`
- Produces: `extract_summary_response_content(response) -> tuple[str, bool]`
- Produces: `extract_summary_cache_stats(response) -> dict[str, Any]`
- Produces: `apply_summary_tool_choice_none(api_kwargs, api_mode) -> tuple[dict[str, Any], bool]`

- [ ] **Step 1: Add failing append-cached request-shape tests**

Append to `tests/agent/test_append_cached_summary.py`:

```python
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


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
        message = SimpleNamespace(content="## Primary Request and Intent\nUser asked to test append cache.\n\n## Key Technical Concepts\nNone.\n\n## Files and Code Sections\nNone.\n\n## Errors and Fixes\nNone.\n\n## Problem Solving\nNone.\n\n## All User Messages\n1. \"old prefix\" — User supplied source content.\n\n## Pending Tasks\nNone.\n\n## Current Work\nCompression summary generated.\n\n## Optional Next Step\nNone.", tool_calls=None)
        usage = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=300), cache_write_tokens=50)
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
```

- [ ] **Step 2: Add failing tool-call and overflow tests**

Append:

```python
class ToolCallRuntime(CapturingRuntime):
    def invoke(self, api_kwargs: dict[str, Any]) -> Any:
        message = SimpleNamespace(content="", tool_calls=[SimpleNamespace(id="call_1")])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class OverflowRuntime(CapturingRuntime):
    def estimate_request_tokens(self, api_kwargs: dict[str, Any]) -> int:
        return 2_000_000


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
```

- [ ] **Step 3: Run failing tests**

Run:

```bash
pytest tests/agent/test_append_cached_summary.py -q
```

Expected result: append-cached tests fail because `_generate_summary` has no `source_messages`, `summarize_start`, or `compress_end` keyword parameters and no append path.

- [ ] **Step 4: Add response/cache/tool helper functions to runtime file**

In `agent/compression_summary_runtime.py`, add:

```python
def apply_summary_tool_choice_none(
    api_kwargs: dict[str, Any],
    api_mode: str,
) -> tuple[dict[str, Any], bool]:
    if "tools" not in api_kwargs:
        return api_kwargs, False
    updated = dict(api_kwargs)
    if api_mode == "anthropic_messages":
        updated["tool_choice"] = {"type": "none"}
    else:
        updated["tool_choice"] = "none"
    return updated, True


def extract_summary_response_content(response: Any) -> tuple[str, bool]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return "", False
    message = getattr(choices[0], "message", None)
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")
    else:
        tool_calls = getattr(message, "tool_calls", None) or []
        content = getattr(message, "content", message)
    if tool_calls:
        return "", True
    if not isinstance(content, str):
        content = str(content) if content else ""
    return content, False


def extract_summary_cache_stats(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {"reported": False, "read_tokens": None, "write_tokens": None, "hit_rate_estimate": None}

    def _get(obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details")
    read_tokens = (
        _get(details, "cached_tokens")
        or _get(details, "cache_read_input_tokens")
        or _get(usage, "cache_read_tokens")
        or _get(usage, "cache_read_input_tokens")
    )
    write_tokens = (
        _get(usage, "cache_write_tokens")
        or _get(usage, "cache_creation_input_tokens")
        or _get(usage, "cache_creation_tokens")
    )
    prompt_tokens = _get(usage, "prompt_tokens") or _get(usage, "input_tokens")
    read_int = int(read_tokens) if read_tokens is not None else None
    write_int = int(write_tokens) if write_tokens is not None else None
    prompt_int = int(prompt_tokens) if prompt_tokens else None
    hit_rate = None
    if read_int is not None and prompt_int:
        hit_rate = round(read_int / max(prompt_int, 1), 4)
    return {
        "reported": read_int is not None or write_int is not None,
        "read_tokens": read_int,
        "write_tokens": write_int,
        "hit_rate_estimate": hit_rate,
    }
```

- [ ] **Step 5: Extend `_generate_summary` signature and branch**

Change the signature:

```python
    def _generate_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
        *,
        source_messages: Optional[List[Dict[str, Any]]] = None,
        summarize_start: Optional[int] = None,
        compress_end: Optional[int] = None,
    ) -> Optional[str]:
```

After cooldown check and before serialized source construction, add:

```python
        if (
            self.summary_call_mode == "append_cached"
            and source_messages is not None
            and summarize_start is not None
            and compress_end is not None
        ):
            summary = self._generate_summary_append_cached(
                source_messages=source_messages,
                turns_to_summarize=turns_to_summarize,
                summarize_start=summarize_start,
                compress_end=compress_end,
                focus_topic=focus_topic,
            )
            if summary is not None:
                return summary
            if not self.append_cached_summary.fallback_to_serialized_prompt:
                return None
```

- [ ] **Step 6: Implement `_generate_summary_append_cached`**

Add this method before `_generate_summary()`:

```python
    def _generate_summary_append_cached(
        self,
        *,
        source_messages: List[Dict[str, Any]],
        turns_to_summarize: List[Dict[str, Any]],
        summarize_start: int,
        compress_end: int,
        focus_topic: Optional[str],
    ) -> Optional[str]:
        from agent.compression_summary_runtime import (
            apply_summary_tool_choice_none,
            extract_summary_cache_stats,
            extract_summary_response_content,
        )

        summary_budget = self._compute_summary_budget(turns_to_summarize)
        rules = self._build_summary_rules(turns_to_summarize, summary_budget)
        instruction = self._build_append_cached_summary_instruction(
            rules,
            previous_summary=self._previous_summary,
            focus_topic=focus_topic,
        )
        base_audit = {
            "mode": "append_cached",
            "source_binding": "provider_payload_prefix_to_compress_end",
            "rules_hash": rules.rules_hash,
            "cache_eligible": False,
            "cache_key_runtime": {},
            "request": {},
            "cache": {"reported": False, "read_tokens": None, "write_tokens": None, "hit_rate_estimate": None},
            "fallback_reason": None,
            "tool_call_violation": False,
        }
        self._last_summary_call_audit = base_audit

        factory = self._summary_runtime_factory
        if factory is None:
            base_audit["fallback_reason"] = "summary_runtime_not_main"
            return None
        runtime = factory()
        if runtime is None:
            base_audit["fallback_reason"] = "summary_runtime_not_main"
            return None

        prefix_messages = list(source_messages[:compress_end])
        request_messages = prefix_messages + [{"role": "user", "content": instruction}]
        requested_output_tokens = int(summary_budget * 1.3)
        api_kwargs = runtime.build_kwargs(request_messages, requested_output_tokens)
        tool_choice_requested = False
        if self.append_cached_summary.allow_tool_choice_none:
            api_kwargs, tool_choice_requested = apply_summary_tool_choice_none(
                api_kwargs,
                getattr(runtime, "api_mode", ""),
            )

        request_tokens = runtime.estimate_request_tokens(api_kwargs)
        runtime_limit = getattr(runtime, "context_limit_tokens", None)
        base_audit["cache_eligible"] = bool(self.append_cached_summary.require_main_runtime)
        base_audit["cache_key_runtime"] = {
            "provider": getattr(runtime, "provider", "") or "",
            "model": getattr(runtime, "model", "") or "",
            "api_mode": getattr(runtime, "api_mode", "") or "",
            "reasoning_effort": getattr(runtime, "reasoning_effort", None),
            "tools_included": bool(getattr(runtime, "tools_included", False)),
            "tool_choice_none_requested": bool(tool_choice_requested),
        }
        base_audit["request"] = {
            "message_count": len(request_messages),
            "prefix_message_count": len(prefix_messages),
            "instruction_chars": len(instruction),
            "tokens_estimate": int(request_tokens),
            "runtime_context_limit_tokens": int(runtime_limit) if runtime_limit else None,
            "requested_output_tokens": requested_output_tokens,
            "summarize_start": int(summarize_start),
            "compress_end": int(compress_end),
        }
        if runtime_limit is None:
            base_audit["fallback_reason"] = "summary_runtime_context_unknown"
            return None
        if request_tokens + requested_output_tokens > int(runtime_limit):
            base_audit["fallback_reason"] = "append_cached_context_overflow"
            return None

        try:
            with aux_interrupt_protection():
                response = runtime.invoke(api_kwargs)
            content, tool_call_violation = extract_summary_response_content(response)
            base_audit["tool_call_violation"] = bool(tool_call_violation)
            base_audit["cache"] = extract_summary_cache_stats(response)
            if tool_call_violation:
                base_audit["fallback_reason"] = "summary_returned_tool_call"
                self._last_summary_error = "append_cached summary response attempted a tool call"
                return None
            if not content.strip():
                base_audit["fallback_reason"] = "append_cached_validation_failed"
                self._last_summary_error = "append_cached summary returned empty content"
                return None

            summary = redact_sensitive_text(content.strip())
            summary, _demoted_sections = self._normalize_summary_sections(summary)
            self._last_summary_user_message_ground_truth = [
                entry["text"]
                for entry in self._extract_current_user_ledger_entries(turns_to_summarize)
            ]
            self._previous_summary = summary
            self._last_summary_sample = self._build_summary_sample_record(
                summary=summary,
                rules_hash=rules.rules_hash,
                mode="append_cached",
            )
            self._clear_compression_failure_cooldown()
            self._summary_failure_cooldown_error = None
            self._summary_model_fallen_back = False
            self._last_summary_error = None
            self._last_summary_auth_failure = False
            self._last_summary_network_failure = False
            self._summary_skipped_for_cooldown = False
            return self._with_summary_prefix(summary)
        except Exception as exc:
            base_audit["fallback_reason"] = "append_cached_transport_error"
            self._last_summary_error = str(exc)
            return None
```

- [ ] **Step 7: Pass source window from `compress()`**

Change line where `summary` is generated from:

```python
        summary = self._generate_summary(turns_to_summarize, focus_topic=summary_focus_topic)
```

to:

```python
        summary = self._generate_summary(
            turns_to_summarize,
            focus_topic=summary_focus_topic,
            source_messages=messages,
            summarize_start=summarize_start,
            compress_end=compress_end,
        )
```

- [ ] **Step 8: Keep serialized fallback auditable**

When append-cached returns `None` and `fallback_to_serialized_prompt` is true, `_generate_summary()` falls through to the old serialized path. Before the serialized call runs, preserve the append failure in the serialized audit:

```python
        append_failure = dict(self._last_summary_call_audit or {})
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)
        rules = self._build_summary_rules(turns_to_summarize, summary_budget)
        prompt = self._build_serialized_summary_prompt(rules, content_to_summarize, focus_topic=focus_topic)
        self._last_summary_call_audit = {
            "mode": "serialized_prompt",
            "source_binding": "serialized_turns_to_summarize",
            "rules_hash": rules.rules_hash,
            "cache_eligible": False,
            "fallback_from": append_failure if append_failure.get("mode") == "append_cached" else None,
            "fallback_reason": None,
            "tool_call_violation": False,
        }
```

- [ ] **Step 9: Run append-cached tests and commit**

Run:

```bash
pytest tests/agent/test_append_cached_summary.py -q
python -m py_compile agent/context_compressor.py agent/compression_summary_runtime.py
```

Expected result: all pass.

Commit:

```bash
git add agent/context_compressor.py agent/compression_summary_runtime.py tests/agent/test_append_cached_summary.py
git commit -m "feat: generate compression summaries with append-cached mode"
```

---

### Task 4: Add content-free audit fields and redacted summary-quality sidecar

**Files:**
- Modify: `agent/context_compressor.py:2418-2675` and `agent/context_compressor.py:5146-5169`
- Test: `tests/agent/test_append_cached_summary.py`

**Interfaces:**
- Consumes: `self._last_summary_call_audit` from Task 3.
- Produces: `_build_summary_sample_record(summary, rules_hash, mode) -> dict[str, Any]`
- Produces: `_write_summary_sample_audit() -> None`
- Produces: audit field `summary_call` on `context_compression` events.

- [ ] **Step 1: Add failing audit and sidecar tests**

Append to `tests/agent/test_append_cached_summary.py`:

```python
import json
from pathlib import Path


def test_compression_audit_contains_summary_call_without_summary_text(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runtime = CapturingRuntime()
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="gpt-5.5",
            protect_first_n=0,
            protect_last_n=1,
            summary_target_ratio=0.01,
            quiet_mode=True,
            summary_call_mode="append_cached",
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
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
            summary_target_ratio=0.01,
            quiet_mode=True,
            summary_call_mode="append_cached",
            append_cached_summary={"audit_sample_summary_chars": 2000},
        )
    compressor.bind_summary_runtime_factory(lambda: runtime)
    compressor.compress([
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
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
```

- [ ] **Step 2: Run failing audit tests**

Run:

```bash
pytest tests/agent/test_append_cached_summary.py::test_compression_audit_contains_summary_call_without_summary_text tests/agent/test_append_cached_summary.py::test_summary_sample_sidecar_records_redacted_structure -q
```

Expected result: fails because `summary_call` and `compression_summary_samples.jsonl` are not written.

- [ ] **Step 3: Add summary section checker and sample builder**

In `agent/context_compressor.py`, add near audit helpers:

```python
    @classmethod
    def _summary_section_check(cls, summary: str) -> dict[str, Any]:
        canonical = list(_CANONICAL_SUMMARY_SECTIONS)
        found = []
        for line in (summary or "").splitlines():
            if line.startswith("## "):
                found.append(line.strip())
        missing = [heading for heading in canonical if heading not in found]
        noncanonical = [heading for heading in found if heading not in canonical]
        all_user_messages_count = 0
        in_user_section = False
        for line in (summary or "").splitlines():
            if line.startswith("## "):
                in_user_section = line.strip() == "## All User Messages"
                continue
            if in_user_section and re.match(r"^\s*\d+\.\s+", line):
                all_user_messages_count += 1
        return {
            "has_all_canonical_sections": not missing,
            "missing_sections": missing,
            "noncanonical_heading_count": len(noncanonical),
            "all_user_messages_count": all_user_messages_count,
            "pending_tasks_says_none": "## Pending Tasks\nNone." in summary,
            "current_work_present": "## Current Work" in found,
            "optional_next_step_present": "## Optional Next Step" in found,
        }

    def _build_summary_sample_record(
        self,
        *,
        summary: str,
        rules_hash: str,
        mode: str,
    ) -> dict[str, Any]:
        cap = self.append_cached_summary.audit_sample_summary_chars
        redacted = redact_sensitive_text(summary or "")
        truncated = bool(cap and len(redacted) > cap)
        if cap and truncated:
            half = max(1, cap // 2)
            excerpt = redacted[:half].rstrip() + "\n[summary excerpt truncated]\n" + redacted[-half:].lstrip()
        elif cap:
            excerpt = redacted
        else:
            excerpt = ""
        return {
            "event": "compression_summary_sample",
            "schema_version": 1,
            "summary_call_mode": mode,
            "rules_hash": rules_hash,
            "summary_chars": len(summary or ""),
            "summary_excerpt": excerpt,
            "truncated": truncated,
            "section_check": self._summary_section_check(summary or ""),
            "quality_flags": [],
        }
```

- [ ] **Step 4: Add summary sample sidecar writer**

Add after `_write_user_message_ground_truth_audit()`:

```python
    def _write_summary_sample_audit(self) -> None:
        sample = self._last_summary_sample
        if not sample:
            return
        self._last_summary_sample = None
        base = self._last_compression_audit_record or {}
        compression_id = base.get("compression_id")
        if not compression_id:
            return
        record = dict(sample)
        record["compression_id"] = compression_id
        record["session_id"] = base.get("session_id") or getattr(self, "_compression_audit_session_id", None)
        try:
            log_path = get_hermes_home() / "logs" / "compression_summary_samples.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            logger.debug("Failed to write compression summary sample", exc_info=True)
```

- [ ] **Step 5: Add `summary_call` to content-free audit**

In `_build_compression_audit_record()` return dict, add:

```python
            "summary_call": dict(self._last_summary_call_audit or {}),
```

Place it next to the existing `summary_source` field so summary-source and summary-call diagnostics stay together.

- [ ] **Step 6: Set sample for serialized success too**

In the existing serialized success path after `self._previous_summary = summary`, add:

```python
            self._last_summary_sample = self._build_summary_sample_record(
                summary=summary,
                rules_hash=rules.rules_hash,
                mode="serialized_prompt",
            )
```

- [ ] **Step 7: Write sample sidecar after main audit**

At the end of `compress()` after `_write_compression_audit_record` and before `_write_user_message_ground_truth_audit`, add:

```python
        self._write_summary_sample_audit()
```

Keep `_write_user_message_ground_truth_audit()` immediately after it so both sidecars share the same `compression_id` from `_last_compression_audit_record`.

- [ ] **Step 8: Run audit tests and commit**

Run:

```bash
pytest tests/agent/test_append_cached_summary.py -q
python -m py_compile agent/context_compressor.py
```

Expected result: all pass.

Commit:

```bash
git add agent/context_compressor.py tests/agent/test_append_cached_summary.py
git commit -m "feat: audit compression summary call and quality samples"
```

---

### Task 5: Add read-only compression audit report script

**Files:**
- Create: `scripts/compression_audit_report.py`
- Create: `tests/scripts/test_compression_audit_report.py`

**Interfaces:**
- Produces CLI:
  - `python scripts/compression_audit_report.py --last 20`
  - `python scripts/compression_audit_report.py --session <session_id>`
  - `python scripts/compression_audit_report.py --compression-id <id> --show-summary`
- Consumes JSONL logs under `~/.hermes/logs/` or `--home <path>`.

- [ ] **Step 1: Write failing report tests**

Create `tests/scripts/test_compression_audit_report.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/compression_audit_report.py")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_report_summarizes_cache_and_fallbacks(tmp_path):
    logs = tmp_path / "logs"
    _write_jsonl(logs / "compression_audit.jsonl", [
        {
            "event": "context_compression",
            "compression_id": "c1",
            "session_id": "s1",
            "result": "success",
            "tokens": {"before_estimate": 1000, "after_estimate": 300},
            "summary_call": {
                "mode": "append_cached",
                "fallback_reason": None,
                "cache": {"reported": True, "read_tokens": 700, "write_tokens": 100, "hit_rate_estimate": 0.7},
            },
        },
        {
            "event": "context_compression",
            "compression_id": "c2",
            "session_id": "s1",
            "result": "fallback",
            "tokens": {"before_estimate": 1200, "after_estimate": 500},
            "summary_call": {
                "mode": "serialized_prompt",
                "fallback_from": {"fallback_reason": "append_cached_context_overflow"},
                "cache": {"reported": False, "read_tokens": None, "write_tokens": None, "hit_rate_estimate": None},
            },
        },
    ])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--home", str(tmp_path), "--last", "20"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "append_cached: 1" in result.stdout
    assert "serialized_prompt: 1" in result.stdout
    assert "cache reported: 1/2" in result.stdout
    assert "append_cached_context_overflow" in result.stdout


def test_report_show_summary_reads_redacted_sample_only(tmp_path):
    logs = tmp_path / "logs"
    _write_jsonl(logs / "compression_audit.jsonl", [{
        "event": "context_compression",
        "compression_id": "c1",
        "session_id": "s1",
        "result": "success",
        "summary_call": {"mode": "append_cached", "cache": {"reported": True}},
    }])
    _write_jsonl(logs / "compression_summary_samples.jsonl", [{
        "event": "compression_summary_sample",
        "compression_id": "c1",
        "session_id": "s1",
        "summary_excerpt": "## Primary Request and Intent\nredacted sample",
        "section_check": {"has_all_canonical_sections": True, "all_user_messages_count": 1},
        "quality_flags": [],
    }])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--home", str(tmp_path), "--compression-id", "c1", "--show-summary"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "redacted sample" in result.stdout
    assert "raw tool" not in result.stdout.lower()
```

- [ ] **Step 2: Run failing script tests**

Run:

```bash
pytest tests/scripts/test_compression_audit_report.py -q
```

Expected result: fails because script does not exist.

- [ ] **Step 3: Implement report script**

Create `scripts/compression_audit_report.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"event": "invalid_json", "raw_len": len(line)})
    return rows


def _home(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    return Path.home() / ".hermes"


def _fallback_reason(row: dict[str, Any]) -> str | None:
    call = row.get("summary_call") or {}
    reason = call.get("fallback_reason")
    if reason:
        return str(reason)
    fallback_from = call.get("fallback_from") or {}
    if isinstance(fallback_from, dict):
        reason = fallback_from.get("fallback_reason")
        if reason:
            return str(reason)
    return None


def build_report(rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> str:
    compactions = [row for row in rows if row.get("event") == "context_compression"]
    modes = Counter((row.get("summary_call") or {}).get("mode", "unknown") for row in compactions)
    fallbacks = Counter(reason for row in compactions if (reason := _fallback_reason(row)))
    cache_reported = 0
    hit_rates: list[float] = []
    read_tokens = 0
    write_tokens = 0
    for row in compactions:
        cache = (row.get("summary_call") or {}).get("cache") or {}
        if cache.get("reported"):
            cache_reported += 1
        if cache.get("hit_rate_estimate") is not None:
            hit_rates.append(float(cache["hit_rate_estimate"]))
        if cache.get("read_tokens") is not None:
            read_tokens += int(cache["read_tokens"])
        if cache.get("write_tokens") is not None:
            write_tokens += int(cache["write_tokens"])
    lines = [
        "Compression audit report",
        f"records: {len(compactions)}",
        "modes: " + ", ".join(f"{mode}: {count}" for mode, count in sorted(modes.items())),
        f"cache reported: {cache_reported}/{len(compactions)}",
        f"cache read tokens: {read_tokens}",
        f"cache write tokens: {write_tokens}",
        "median cache hit rate: " + (f"{median(hit_rates):.4f}" if hit_rates else "n/a"),
        "fallback reasons: " + (", ".join(f"{reason}: {count}" for reason, count in sorted(fallbacks.items())) or "none"),
        f"summary samples: {len(samples)}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read Hermes compression audit logs.")
    parser.add_argument("--home", default=None, help="Hermes home directory; defaults to ~/.hermes")
    parser.add_argument("--last", type=int, default=20)
    parser.add_argument("--session", default=None)
    parser.add_argument("--compression-id", default=None)
    parser.add_argument("--show-summary", action="store_true")
    args = parser.parse_args()

    home = _home(args.home)
    rows = _read_jsonl(home / "logs" / "compression_audit.jsonl")
    samples = _read_jsonl(home / "logs" / "compression_summary_samples.jsonl")
    if args.session:
        rows = [row for row in rows if row.get("session_id") == args.session]
        samples = [row for row in samples if row.get("session_id") == args.session]
    if args.compression_id:
        rows = [row for row in rows if row.get("compression_id") == args.compression_id]
        samples = [row for row in samples if row.get("compression_id") == args.compression_id]
    if args.last and not args.compression_id:
        rows = rows[-args.last:]

    print(build_report(rows, samples))
    if args.show_summary:
        sample_by_id = {row.get("compression_id"): row for row in samples}
        for row in rows:
            sample = sample_by_id.get(row.get("compression_id"))
            if not sample:
                continue
            print("\n--- summary sample", row.get("compression_id"), "---")
            print(sample.get("summary_excerpt") or "")
            print("section_check:", json.dumps(sample.get("section_check") or {}, ensure_ascii=False, sort_keys=True))
            print("quality_flags:", json.dumps(sample.get("quality_flags") or [], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run script tests and commit**

Run:

```bash
pytest tests/scripts/test_compression_audit_report.py -q
python -m py_compile scripts/compression_audit_report.py
```

Expected result: all pass.

Commit:

```bash
git add scripts/compression_audit_report.py tests/scripts/test_compression_audit_report.py
git commit -m "feat: add compression audit report script"
```

---

### Task 6: Full regression, independent review, and canary notes

**Files:**
- Modify if needed: `.hermes/specs/2026-07-08-compression-append-cached-summary-audit-design.md`
- Modify if needed: `.hermes/plans/2026-07-08-compression-append-cached-summary-audit-implementation-plan.md`

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: final tested branch ready for Zongxin profile canary.

- [ ] **Step 1: Run full compression regression bundle**

Run:

```bash
pytest tests/agent/test_append_cached_summary.py tests/agent/test_*compress*.py tests/run_agent/test_*compress*.py tests/run_agent/test_compression_*.py -q
pytest tests/gateway/test_compress_command.py tests/cli/test_manual_compress.py tests/cli/test_compress_here.py -q
pytest tests/scripts/test_compression_audit_report.py -q
git diff --check
python -m py_compile agent/context_compressor.py agent/conversation_compression.py agent/chat_completion_helpers.py agent/compression_summary_runtime.py scripts/compression_audit_report.py
```

Expected result: all commands exit 0.

- [ ] **Step 2: Inspect staged scope before review**

Run:

```bash
git status --short
git diff --stat $(git merge-base main HEAD)..HEAD
git diff -- . ':(exclude).hermes/plans/2026-07-08-compression-append-cached-summary-audit-implementation-plan.md' | sed -n '1,260p'
```

Expected result: diff only covers the files named in this plan plus the approved Spec/Plan docs.

- [ ] **Step 3: Request Codex review, pass 1/2**

Run the review after all local tests pass:

```bash
codex review --base main --head HEAD
```

Allowed because this touches core compression logic, provider-visible request shape, auditing, and multiple files. This is pass 1/2 under the review budget.

- [ ] **Step 4: Fix clear review findings with focused tests**

For each accepted review finding:

```bash
git add <exact files touched by the fix>
git commit -m "fix: address append-cached compression review finding"
```

Run the focused test that covers the finding before committing. If the finding changes the request shape or failure mode, add or update a test in `tests/agent/test_append_cached_summary.py`.

- [ ] **Step 5: Second review only if the first review found systemic problems**

If the first review forced a broad rewrite of runtime wiring or summary audit shape, run:

```bash
codex review --base main --head HEAD
```

This is pass 2/2 and the hard stop. If the first review only produced local fixes, skip this step and note that the review budget does not require a second pass.

- [ ] **Step 6: Final canary instructions for Zongxin profile**

Add this canary note to the final implementation report, not to default config:

```yaml
compression:
  summary_call_mode: append_cached
```

Manual smoke commands after enabling in a test profile or local config copy:

```bash
python scripts/compression_audit_report.py --last 20
python scripts/compression_audit_report.py --compression-id <id> --show-summary
```

Expected live smoke evidence:

```text
summary_call.mode == append_cached
summary_call.source_binding == provider_payload_prefix_to_compress_end
summary_call.cache.reported == true when provider reports cache stats
summary_call.cache.read_tokens > 0 when provider reports cache stats and prefix was warm
summary_call.fallback_reason == null on successful append-cached runs
compression_summary_samples.jsonl has canonical section_check and redacted summary_excerpt
compression_user_messages.jsonl still correlates by compression_id
```

- [ ] **Step 7: Final commit if any late docs changed**

Run:

```bash
git status --short
git add -f .hermes/specs/2026-07-08-compression-append-cached-summary-audit-design.md .hermes/plans/2026-07-08-compression-append-cached-summary-audit-implementation-plan.md
git commit -m "docs: update append-cached compression rollout notes"
```

Only commit if either doc actually changed. If there are no doc changes, run:

```bash
git status --short
```

Expected result: clean working tree.

---

## Self-Review

**Spec coverage:**
- Cache-friendly append summary call: Task 3.
- Existing summary rules preserved: Task 2.
- 272K threshold versus runtime limit split: Task 3 tests and audit.
- Content-free audit: Task 4.
- Redacted quality sidecar: Task 4.
- Debug report: Task 5.
- Fallback/fail-closed behavior: Task 3 and Task 6 regression bundle.
- Canary and review workflow: Task 6.

**No deferred-slot scan:**
- The plan has no deferred requirement slots.
- Every created file has exact initial content or exact code blocks for the relevant new interfaces.
- Existing huge summary-rule strings are moved without editing from their current code block rather than rewritten from memory.

**Type consistency:**
- `AppendCachedSummaryConfig` is defined in Task 1 and consumed by `ContextCompressor` in Task 1 onward.
- `SummaryRuntime` is defined in Task 1 and consumed by append-cached implementation in Task 3.
- `SummaryRules` is defined in Task 2 and consumed by serialized and append-cached builders.
- `summary_call` audit fields are produced in Task 4 and consumed by the script in Task 5.
