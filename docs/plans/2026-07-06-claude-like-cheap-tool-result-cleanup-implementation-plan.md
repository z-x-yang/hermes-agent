# Claude-like Cheap Tool Result Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a config-gated Claude Code-like cheap old tool-result cleanup stage for Hermes compression that preserves protected tail, replaces old tool results with recoverable handles or Claude-style sentinels, uses the cleaned view for later summary compaction, and emits content-free audit records.

**Architecture:** Add a small cleanup configuration/result layer inside `agent/context_compressor.py`, parse it from `compression.cheap_tool_result_cleanup`, and integrate the cleanup stage between tail-boundary selection and `_generate_summary()`. Reuse existing `state.db` archived message rows via model-visible `hermes://session/<session_id>/message/<row_id>` handles and existing `session_search`; do not add a new always-on core model tool or duplicate large outputs into sidecar files.

**Tech Stack:** Python 3.11, Hermes `ContextCompressor`, `SessionDB`, pytest, existing rough token estimators and compression audit JSONL.

## Global Constraints

- `compression.cheap_tool_result_cleanup.enabled` defaults to `false` in code/default config.
- Claude-like defaults when enabled: `keep_recent=5`, `min_tokens_saved=20000`, sentinel text `[Old tool result content cleared]`, `replacement_mode=persisted_handle_or_sentinel`, `skip_llm_summary_when_below_threshold=true`.
- Protected tail is stronger than `keep_recent`: protected-tail tool results are never cleared and count toward `keep_recent` first.
- Cleanup scope is the pre-tail summary window only: `messages[summarize_start:compress_end]`.
- Cleanup must not affect tail-boundary selection; boundaries are computed from the original payload before cleanup.
- When cleanup applies and LLM summary runs, summary source is the cleaned view; raw old tool results are not automatically restored.
- If cleanup alone gets an eligible auto token-pressure trigger below threshold, return `cheap_cleanup_only` without calling `_generate_summary()`.
- Manual `/compress`, explicit focus compaction, and hard-message-limit hygiene still run LLM summary after cleanup.
- Do not add a new always-on core model tool in v1; persisted handles should instruct the model to use existing `session_search`.
- Do not write duplicate large tool-output sidecar files in v1.
- Audit must never log raw tool result content or raw tool call arguments.

---

## File Structure

- Modify `agent/context_compressor.py`
  - Add config/result dataclasses.
  - Add cleanup helper methods.
  - Integrate cleanup into `compress()`.
  - Add audit block to `_build_compression_audit_record()`.

- Modify `agent/agent_init.py`
  - Parse and normalize `compression.cheap_tool_result_cleanup`.
  - Pass normalized config into `ContextCompressor`.

- Modify `hermes_cli/config.py`
  - Add default config comments under `DEFAULT_CONFIG["compression"]`.
  - Update compression config display to show the nested cleanup state.

- Create `tests/agent/test_context_compressor_cheap_cleanup.py`
  - Focused unit/integration tests for cleanup semantics.

- Modify existing compressor tests only when a helper fixture must accept the new constructor argument.

---

### Task 1: Config model and startup plumbing

**Files:**
- Modify: `agent/context_compressor.py`
- Modify: `agent/agent_init.py`
- Modify: `hermes_cli/config.py`
- Test: `tests/agent/test_context_compressor_cheap_cleanup.py`

**Interfaces:**
- Produces: `CheapToolResultCleanupConfig`
- Produces: `ContextCompressor(..., cheap_tool_result_cleanup: CheapToolResultCleanupConfig | None = None)`
- Consumes later: `self.cheap_tool_result_cleanup`

- [ ] **Step 1: Create the failing config-constructor tests**

Create `tests/agent/test_context_compressor_cheap_cleanup.py` with this initial content:

```python
from __future__ import annotations

from unittest.mock import patch

from agent.context_compressor import (
    CheapToolResultCleanupConfig,
    ContextCompressor,
)


def _compressor(**kwargs) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.80,
            protect_first_n=1,
            protect_last_n=2,
            summary_target_ratio=0.10,
            quiet_mode=True,
            **kwargs,
        )


def test_cheap_cleanup_config_defaults_disabled():
    c = _compressor()

    cfg = c.cheap_tool_result_cleanup

    assert cfg.enabled is False
    assert cfg.keep_recent == 5
    assert cfg.min_tokens_saved == 20_000
    assert cfg.replacement_mode == "persisted_handle_or_sentinel"
    assert cfg.skip_llm_summary_when_below_threshold is True


def test_cheap_cleanup_config_can_be_injected():
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=3,
        min_tokens_saved=1234,
        replacement_mode="persisted_handle_or_sentinel",
        skip_llm_summary_when_below_threshold=False,
    )

    c = _compressor(cheap_tool_result_cleanup=cfg)

    assert c.cheap_tool_result_cleanup == cfg
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py -q -o 'addopts='
```

Expected: import failure for `CheapToolResultCleanupConfig` or constructor argument failure.

- [ ] **Step 3: Add cleanup config dataclass and constructor plumbing**

In `agent/context_compressor.py`, add `dataclass` to imports if absent:

```python
from dataclasses import dataclass
```

Near the compression constants, add:

```python
_CLAUDE_TOOL_RESULT_CLEARED_SENTINEL = "[Old tool result content cleared]"
_CHEAP_TOOL_CLEANUP_REPLACEMENT_MODE = "persisted_handle_or_sentinel"


@dataclass(frozen=True)
class CheapToolResultCleanupConfig:
    """Config for Claude-like deterministic old tool-result cleanup."""

    enabled: bool = False
    keep_recent: int = 5
    min_tokens_saved: int = 20_000
    replacement_mode: str = _CHEAP_TOOL_CLEANUP_REPLACEMENT_MODE
    skip_llm_summary_when_below_threshold: bool = True

    @classmethod
    def normalized(cls, value: Any | None) -> "CheapToolResultCleanupConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls()

        def _bool(raw: Any, default: bool) -> bool:
            if isinstance(raw, bool):
                return raw
            if raw is None:
                return default
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}

        def _non_negative_int(raw: Any, default: int) -> int:
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                return default
            return max(0, parsed)

        mode = str(value.get("replacement_mode") or _CHEAP_TOOL_CLEANUP_REPLACEMENT_MODE)
        if mode != _CHEAP_TOOL_CLEANUP_REPLACEMENT_MODE:
            mode = _CHEAP_TOOL_CLEANUP_REPLACEMENT_MODE

        return cls(
            enabled=_bool(value.get("enabled"), False),
            keep_recent=_non_negative_int(value.get("keep_recent"), 5),
            min_tokens_saved=_non_negative_int(value.get("min_tokens_saved"), 20_000),
            replacement_mode=mode,
            skip_llm_summary_when_below_threshold=_bool(
                value.get("skip_llm_summary_when_below_threshold"), True
            ),
        )
```

Update `ContextCompressor.__init__` signature:

```python
        max_tokens: int | None = None,
        cheap_tool_result_cleanup: CheapToolResultCleanupConfig | dict[str, Any] | None = None,
    ):
```

Inside `__init__`, after `self.abort_on_summary_failure = abort_on_summary_failure`, add:

```python
        self.cheap_tool_result_cleanup = CheapToolResultCleanupConfig.normalized(
            cheap_tool_result_cleanup
        )
        self._last_cheap_tool_cleanup_audit: dict[str, Any] = {
            "enabled": self.cheap_tool_result_cleanup.enabled,
            "applied": False,
            "result": "not_attempted",
        }
```

In `on_session_reset()` and the start of `compress()`, reset `_last_cheap_tool_cleanup_audit` to the same small disabled/not-attempted block using the current config.

- [ ] **Step 4: Parse config in `agent/agent_init.py`**

In `agent/agent_init.py`, after `compression_abort_on_summary_failure` is parsed, add:

```python
    cheap_tool_result_cleanup_cfg = _compression_cfg.get("cheap_tool_result_cleanup", {})
    if not isinstance(cheap_tool_result_cleanup_cfg, dict):
        cheap_tool_result_cleanup_cfg = {}
```

At the `ContextCompressor(...)` call, pass:

```python
            cheap_tool_result_cleanup=cheap_tool_result_cleanup_cfg,
```

Do not import `CheapToolResultCleanupConfig` in `agent_init.py`; let the compressor normalize the dict. That keeps startup wiring simple and avoids a second normalization path.

- [ ] **Step 5: Add default config comments**

In `hermes_cli/config.py`, inside `DEFAULT_CONFIG["compression"]`, add:

```python
        "cheap_tool_result_cleanup": {
            "enabled": False,
            "keep_recent": 5,
            "min_tokens_saved": 20000,
            "replacement_mode": "persisted_handle_or_sentinel",
            "skip_llm_summary_when_below_threshold": True,
        },
```

In the compression display section near the lines that print enabled/threshold/target/protect last, add:

```python
        _cheap = compression.get("cheap_tool_result_cleanup", {})
        if isinstance(_cheap, dict):
            _cheap_enabled = str(_cheap.get("enabled", False)).lower() in {"true", "1", "yes"}
            print(f"  Cheap cleanup: {'yes' if _cheap_enabled else 'no'}")
            if _cheap_enabled:
                print(f"    keep_recent:      {_cheap.get('keep_recent', 5)}")
                print(f"    min_tokens_saved: {_cheap.get('min_tokens_saved', 20000)}")
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py -q -o 'addopts='
```

Expected: both tests pass.

Then commit only touched files:

```bash
git status --short
git add agent/context_compressor.py agent/agent_init.py hermes_cli/config.py tests/agent/test_context_compressor_cheap_cleanup.py
git commit -m "feat(compression): add cheap tool cleanup config"
```

---

### Task 2: Deterministic cleanup helper

**Files:**
- Modify: `agent/context_compressor.py`
- Modify: `tests/agent/test_context_compressor_cheap_cleanup.py`

**Interfaces:**
- Produces: `CheapToolResultCleanupResult`
- Produces: `ContextCompressor._cleanup_old_tool_results(messages, summarize_start, compress_end) -> CheapToolResultCleanupResult`
- Produces: `ContextCompressor._cheap_tool_result_replacement(msg, index) -> tuple[str, str, str | None]`

- [ ] **Step 1: Add failing direct-helper tests**

Append these tests to `tests/agent/test_context_compressor_cheap_cleanup.py`:

```python

def _assistant_call(call_id: str, name: str = "terminal") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _tool(call_id: str, content: str, *, row_id: int | None = None) -> dict:
    msg = {"role": "tool", "tool_call_id": call_id, "content": content}
    if row_id is not None:
        msg["id"] = row_id
    return msg


def test_cleanup_counts_tail_tools_against_keep_recent():
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=5, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_id="sess-1")
    big = "x" * 4000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", big, row_id=11),
        _assistant_call("old-2"),
        _tool("old-2", big, row_id=12),
        _assistant_call("old-3"),
        _tool("old-3", big, row_id=13),
        _assistant_call("tail-1"),
        _tool("tail-1", big, row_id=21),
        _assistant_call("tail-2"),
        _tool("tail-2", big, row_id=22),
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=8)

    assert result.applied is True
    assert result.messages[3]["content"].startswith("<persisted-output>")
    assert result.messages[5]["content"] == big
    assert result.messages[7]["content"] == big
    assert result.messages[9]["content"] == big
    assert result.messages[11]["content"] == big
    assert result.audit["tail_tool_result_count"] == 2
    assert result.audit["extra_pre_tail_keep_count"] == 3
    assert result.audit["cleared_count"] == 1
    assert result.audit["protected_tail_cleared_count"] == 0


def test_cleanup_uses_sentinel_when_row_id_missing():
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_id="sess-1")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "x" * 4000),
        {"role": "user", "content": "tail"},
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    assert result.applied is True
    assert result.messages[3]["content"] == "[Old tool result content cleared]"
    assert result.audit["replacement_counts"]["sentinel"] == 1
    assert result.audit["sentinel_fallback_reasons"]["missing_row_id"] == 1


def test_cleanup_below_min_tokens_saved_does_not_mutate():
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=20_000)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_id="sess-1")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "small", row_id=11),
        {"role": "user", "content": "tail"},
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    assert result.applied is False
    assert result.messages is messages
    assert result.audit["result"] == "below_min_tokens_saved"
```

- [ ] **Step 2: Run tests and verify helper is missing**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py -q -o 'addopts='
```

Expected: failures for missing `bind_session_state` argument support in the test call or missing `_cleanup_old_tool_results`.

If `bind_session_state(session_id="sess-1")` requires `session_db`, call `c.bind_session_state(session_db=None, session_id="sess-1")` in tests.

- [ ] **Step 3: Add helper result dataclass**

In `agent/context_compressor.py`, below `CheapToolResultCleanupConfig`, add:

```python
@dataclass(frozen=True)
class CheapToolResultCleanupResult:
    """Result of deterministic old tool-result cleanup."""

    messages: list[dict[str, Any]]
    applied: bool
    audit: dict[str, Any]
    tokens_saved_estimate: int = 0
    post_tokens_estimate: int | None = None
```

- [ ] **Step 4: Add persisted-output detection and replacement helpers**

Inside `ContextCompressor`, near the legacy tool-output pruning helper section, add:

```python
    @staticmethod
    def _is_persisted_output_block(content: Any) -> bool:
        text = _content_text_for_contains(content).strip()
        return text.startswith("<persisted-output>") and text.endswith("</persisted-output>")

    @staticmethod
    def _short_audit_hash(value: Any) -> str:
        text = str(value or "")
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]

    def _cheap_tool_result_replacement(
        self,
        msg: dict[str, Any],
        index: int,
    ) -> tuple[str, str, str | None]:
        session_id = getattr(self, "_session_id", "") or ""
        row_id = msg.get("id")
        try:
            row_id_int = int(row_id)
        except (TypeError, ValueError):
            row_id_int = 0

        if session_id and row_id_int > 0:
            handle = f"hermes://session/{session_id}/message/{row_id_int}"
            content = (
                "<persisted-output>\n"
                f"Tool result archived at: {handle}\n"
                "Use session_search("
                f"session_id=\"{session_id}\", "
                f"around_message_id={row_id_int}, "
                "window=1, role_filter=\"tool\") to view it if needed.\n"
                "</persisted-output>"
            )
            return "persisted_handle", content, None

        return "sentinel", _CLAUDE_TOOL_RESULT_CLEARED_SENTINEL, "missing_row_id"
```

This first implementation uses row ids already present in loaded messages. If later natural audit shows frequent `missing_row_id`, add a DB matching helper as a follow-up patch.

- [ ] **Step 5: Add cleanup helper**

Inside `ContextCompressor`, add:

```python
    def _cleanup_old_tool_results(
        self,
        messages: list[dict[str, Any]],
        summarize_start: int,
        compress_end: int,
    ) -> CheapToolResultCleanupResult:
        cfg = self.cheap_tool_result_cleanup
        base_audit: dict[str, Any] = {
            "enabled": bool(cfg.enabled),
            "applied": False,
            "result": "disabled" if not cfg.enabled else "not_attempted",
            "scope": "summary_window_before_protected_tail",
            "view": "cleaned_after_cheap_tool_result_cleanup",
            "keep_recent": int(cfg.keep_recent),
            "min_tokens_saved": int(cfg.min_tokens_saved),
            "tail_tool_result_count": 0,
            "extra_pre_tail_keep_count": 0,
            "candidate_count": 0,
            "clear_candidate_count": 0,
            "kept_recent_pre_tail_count": 0,
            "cleared_count": 0,
            "protected_tail_cleared_count": 0,
            "tokens_saved_estimate": 0,
            "replacement_counts": {"persisted_handle": 0, "sentinel": 0},
            "sentinel_fallback_reasons": {},
            "summary_source_view": "cleaned_after_cheap_tool_result_cleanup",
            "raw_tool_results_restored_for_summary": False,
            "llm_summary_skipped_after_cleanup": False,
            "llm_summary_ran_on_cleaned_view": False,
            "cleared_tool_call_id_hashes": [],
        }
        if not cfg.enabled:
            return CheapToolResultCleanupResult(messages, False, base_audit)

        summarize_start = max(0, int(summarize_start))
        compress_end = max(summarize_start, min(int(compress_end), len(messages)))

        tail_tool_count = sum(
            1 for msg in messages[compress_end:]
            if isinstance(msg, dict) and msg.get("role") == "tool"
        )
        extra_keep = max(0, int(cfg.keep_recent) - tail_tool_count)
        base_audit["tail_tool_result_count"] = tail_tool_count
        base_audit["extra_pre_tail_keep_count"] = extra_keep

        candidates: list[tuple[int, dict[str, Any]]] = []
        for index in range(summarize_start, compress_end):
            msg = messages[index]
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if content in (None, "", []):
                continue
            if content == _CLAUDE_TOOL_RESULT_CLEARED_SENTINEL:
                continue
            if self._is_persisted_output_block(content):
                continue
            candidates.append((index, msg))

        base_audit["candidate_count"] = len(candidates)
        if not candidates:
            base_audit["result"] = "no_candidates"
            return CheapToolResultCleanupResult(messages, False, base_audit)

        keep_indices = {index for index, _msg in candidates[-extra_keep:]} if extra_keep else set()
        clear_candidates = [(index, msg) for index, msg in candidates if index not in keep_indices]
        base_audit["kept_recent_pre_tail_count"] = len(keep_indices)
        base_audit["clear_candidate_count"] = len(clear_candidates)
        if not clear_candidates:
            base_audit["result"] = "all_candidates_kept_recent"
            return CheapToolResultCleanupResult(messages, False, base_audit)

        replacements: list[tuple[int, dict[str, Any], str, str, str | None]] = []
        tokens_saved = 0
        for index, msg in clear_candidates:
            replacement_type, replacement_content, fallback_reason = self._cheap_tool_result_replacement(msg, index)
            old_tokens = estimate_messages_tokens_rough([msg])
            new_msg = {**msg, "content": replacement_content}
            new_tokens = estimate_messages_tokens_rough([new_msg])
            tokens_saved += max(0, int(old_tokens) - int(new_tokens))
            replacements.append((index, msg, replacement_type, replacement_content, fallback_reason))

        base_audit["tokens_saved_estimate"] = int(tokens_saved)
        if tokens_saved < int(cfg.min_tokens_saved):
            base_audit["result"] = "below_min_tokens_saved"
            return CheapToolResultCleanupResult(messages, False, base_audit, int(tokens_saved))

        cleaned = [m.copy() if isinstance(m, dict) else m for m in messages]
        fallback_reasons: dict[str, int] = {}
        cleared_hashes: list[str] = []
        replacement_counts = {"persisted_handle": 0, "sentinel": 0}
        for index, old_msg, replacement_type, replacement_content, fallback_reason in replacements:
            cleaned[index] = {**old_msg, "content": replacement_content}
            replacement_counts[replacement_type] = replacement_counts.get(replacement_type, 0) + 1
            if fallback_reason:
                fallback_reasons[fallback_reason] = fallback_reasons.get(fallback_reason, 0) + 1
            if len(cleared_hashes) < 50:
                cleared_hashes.append(self._short_audit_hash(old_msg.get("tool_call_id")))

        post_tokens = int(estimate_messages_tokens_rough(cleaned))
        audit = dict(base_audit)
        audit.update({
            "applied": True,
            "result": "applied",
            "cleared_count": len(replacements),
            "replacement_counts": replacement_counts,
            "sentinel_fallback_reasons": fallback_reasons,
            "post_cleanup_tokens_estimate": post_tokens,
            "cleared_tool_call_id_hashes": cleared_hashes,
        })
        return CheapToolResultCleanupResult(cleaned, True, audit, int(tokens_saved), post_tokens)
```

- [ ] **Step 6: Run helper tests and commit**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py -q -o 'addopts='
```

Expected: all tests in this file pass.

Commit:

```bash
git status --short
git add agent/context_compressor.py tests/agent/test_context_compressor_cheap_cleanup.py
git commit -m "feat(compression): add cheap tool cleanup helper"
```

---

### Task 3: Integrate cleanup into `compress()` and summary source

**Files:**
- Modify: `agent/context_compressor.py`
- Modify: `tests/agent/test_context_compressor_cheap_cleanup.py`

**Interfaces:**
- Consumes: `_cleanup_old_tool_results()` from Task 2
- Produces: `compress()` uses cleaned messages for both live output and `_generate_summary()` source when cleanup applies

- [ ] **Step 1: Add failing summary-source integration test**

Append:

```python

def test_summary_source_uses_cleaned_tool_result_when_cleanup_applies(monkeypatch):
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_db=None, session_id="sess-1")
    captured = {}

    def fake_generate(turns, focus_topic=None):
        captured["turns"] = turns
        return "## Current Work\n- compressed\n"

    monkeypatch.setattr(c, "_generate_summary", fake_generate)
    big = "RAW_TOOL_OUTPUT_SHOULD_NOT_REACH_SUMMARY " * 300
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", big, row_id=11),
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    out = c.compress(messages, current_tokens=90_000, force=False, trigger_reason="token_threshold")

    serialized_turns = str(captured["turns"])
    assert "RAW_TOOL_OUTPUT_SHOULD_NOT_REACH_SUMMARY" not in serialized_turns
    assert "<persisted-output>" in serialized_turns
    assert any("<persisted-output>" in str(msg.get("content")) for msg in out)
    assert c._last_cheap_tool_cleanup_audit["applied"] is True
    assert c._last_cheap_tool_cleanup_audit["raw_tool_results_restored_for_summary"] is False
```

- [ ] **Step 2: Run and verify it fails**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py::test_summary_source_uses_cleaned_tool_result_when_cleanup_applies -q -o 'addopts='
```

Expected: raw tool output reaches fake summarizer or `_last_cheap_tool_cleanup_audit` is not populated.

- [ ] **Step 3: Integrate cleanup after `summarize_start` is known**

In `compress()`, after the block that sets `summarize_start` and `turns_to_summarize`, replace:

```python
        turns_to_summarize = list(turns_to_summarize)
```

with:

```python
        cleanup_result = self._cleanup_old_tool_results(
            messages,
            summarize_start=summarize_start,
            compress_end=compress_end,
        )
        self._last_cheap_tool_cleanup_audit = dict(cleanup_result.audit)
        if cleanup_result.applied:
            messages = cleanup_result.messages
            raw_messages = [m.copy() if isinstance(m, dict) else m for m in messages]
            turns_to_summarize = raw_messages[summarize_start:compress_end]
        else:
            turns_to_summarize = list(turns_to_summarize)
```

This deliberately changes the previous raw-source invariant only when cleanup is applied.

- [ ] **Step 4: Mark summary-source audit as cleaned**

In `_serialize_for_summary()`, initialize `source_audit` with the cleanup view:

```python
        cleanup_audit = getattr(self, "_last_cheap_tool_cleanup_audit", {}) or {}
        source_view = (
            "cleaned_after_cheap_tool_result_cleanup"
            if cleanup_audit.get("applied") else "raw"
        )
        source_audit: dict[str, Any] = {
            "view": source_view,
            "raw_tool_results_restored_for_summary": False if cleanup_audit.get("applied") else None,
            "budget_chars": source_char_budget,
            "budget_tokens_estimate": source_token_budget,
            "raw_chars": 0,
            "final_chars": 0,
            "overflow": False,
            "steps": [],
        }
```

Keep existing budget fields and overflow logic intact.

- [ ] **Step 5: Run summary-source test and commit**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py::test_summary_source_uses_cleaned_tool_result_when_cleanup_applies -q -o 'addopts='
```

Expected: pass.

Then run the whole focused file:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py -q -o 'addopts='
```

Commit:

```bash
git status --short
git add agent/context_compressor.py tests/agent/test_context_compressor_cheap_cleanup.py
git commit -m "feat(compression): summarize cleaned tool cleanup view"
```

---

### Task 4: Cheap-only auto-compression path

**Files:**
- Modify: `agent/context_compressor.py`
- Modify: `tests/agent/test_context_compressor_cheap_cleanup.py`

**Interfaces:**
- Consumes: `_cleanup_old_tool_results()`
- Produces: `compress()` result category `cheap_cleanup_only`

- [ ] **Step 1: Add failing cheap-only tests**

Append:

```python

def test_auto_cleanup_only_skips_summary_when_below_threshold(monkeypatch):
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=0,
        min_tokens_saved=1,
        skip_llm_summary_when_below_threshold=True,
    )
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_db=None, session_id="sess-1")
    called = {"summary": False}

    def fail_if_called(turns, focus_topic=None):
        called["summary"] = True
        raise AssertionError("summary should be skipped")

    monkeypatch.setattr(c, "_generate_summary", fail_if_called)
    monkeypatch.setattr(c, "threshold_tokens", 50_000)
    big = "x" * 220_000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", big, row_id=11),
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    out = c.compress(
        messages,
        current_tokens=80_000,
        force=False,
        trigger_reason="token_threshold",
    )

    assert called["summary"] is False
    assert any("<persisted-output>" in str(msg.get("content")) for msg in out)
    assert c._last_compression_audit_record["result"] == "cheap_cleanup_only"
    assert c._last_compression_audit_record["cheap_tool_result_cleanup"]["llm_summary_skipped_after_cleanup"] is True


def test_manual_compress_does_not_use_cleanup_only(monkeypatch):
    cfg = CheapToolResultCleanupConfig(
        enabled=True,
        keep_recent=0,
        min_tokens_saved=1,
        skip_llm_summary_when_below_threshold=True,
    )
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_db=None, session_id="sess-1")
    called = {"summary": False}

    def fake_generate(turns, focus_topic=None):
        called["summary"] = True
        return "## Current Work\n- manual summary\n"

    monkeypatch.setattr(c, "_generate_summary", fake_generate)
    monkeypatch.setattr(c, "threshold_tokens", 50_000)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "x" * 220_000, row_id=11),
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    c.compress(messages, current_tokens=80_000, force=True, trigger_reason="manual")

    assert called["summary"] is True
```

- [ ] **Step 2: Run and verify cheap-only is absent**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py::test_auto_cleanup_only_skips_summary_when_below_threshold tests/agent/test_context_compressor_cheap_cleanup.py::test_manual_compress_does_not_use_cleanup_only -q -o 'addopts='
```

Expected: first test fails because `_generate_summary()` is called or result is not `cheap_cleanup_only`.

- [ ] **Step 3: Add cheap-only eligibility helper**

Inside `ContextCompressor`, add:

```python
    def _cheap_cleanup_only_allowed(
        self,
        *,
        entrypoint: str,
        trigger_reason: str | None,
        focus_topic: str | None,
        cleanup_result: CheapToolResultCleanupResult,
    ) -> bool:
        cfg = self.cheap_tool_result_cleanup
        if not cfg.enabled or not cfg.skip_llm_summary_when_below_threshold:
            return False
        if not cleanup_result.applied:
            return False
        if entrypoint != "auto":
            return False
        if focus_topic:
            return False
        if trigger_reason in {"manual", "hard_message_limit", "hygiene_hard_message_limit"}:
            return False
        post_tokens = cleanup_result.post_tokens_estimate
        if post_tokens is None:
            return False
        return int(post_tokens) < int(getattr(self, "threshold_tokens", 0) or 0)
```

- [ ] **Step 4: Add cheap-only branch before `_generate_summary()`**

In `compress()`, after cleanup integration and before logging/generating summary, add:

```python
        if self._cheap_cleanup_only_allowed(
            entrypoint=entrypoint,
            trigger_reason=trigger_reason,
            focus_topic=focus_topic,
            cleanup_result=cleanup_result,
        ):
            cheap_messages = self._sanitize_tool_pairs(messages)
            cheap_messages = _strip_historical_media(cheap_messages)
            cheap_messages, metadata_bounded = _bound_retained_nonvisible_metadata(cheap_messages)
            _strip_persistence_markers(cheap_messages)
            cheap_estimate = estimate_messages_tokens_rough(cheap_messages)
            cheap_audit = dict(self._last_cheap_tool_cleanup_audit)
            cheap_audit["llm_summary_skipped_after_cleanup"] = True
            cheap_audit["llm_summary_ran_on_cleaned_view"] = False
            self._last_cheap_tool_cleanup_audit = cheap_audit
            self._write_compression_audit_record(self._build_compression_audit_record(
                result="cheap_cleanup_only",
                entrypoint=entrypoint,
                input_messages=n_messages,
                output_messages=len(cheap_messages),
                summary_start=summarize_start,
                summary_end=compress_end,
                retained_tail_start=compress_end,
                pruned_count=0,
                tail_compacted_count=tail_tool_compacted,
                tail_boundary_promoted=tail_boundary_promoted,
                before_estimate=display_tokens,
                after_estimate=cheap_estimate,
                before_messages=original_messages,
                after_messages=cheap_messages,
                retained_tail_messages=cheap_messages[compress_end:n_messages],
                retained_tail_raw_messages=original_messages[compress_end:n_messages],
                **audit_trigger_kwargs,
            ))
            return cheap_messages
```

If `cheap_messages[compress_end:n_messages]` is wrong after sanitization drops rows, compute the retained tail audit slice using the same retained-tail output-count approach used in the success path. Prefer correct audit over a shorter patch.

- [ ] **Step 5: Mark full-summary cleanup audit**

Before calling `_generate_summary()`, when `cleanup_result.applied` and cheap-only is not taken, update:

```python
        if cleanup_result.applied:
            cleanup_audit = dict(self._last_cheap_tool_cleanup_audit)
            cleanup_audit["llm_summary_skipped_after_cleanup"] = False
            cleanup_audit["llm_summary_ran_on_cleaned_view"] = True
            self._last_cheap_tool_cleanup_audit = cleanup_audit
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py -q -o 'addopts='
```

Expected: all cheap cleanup tests pass.

Commit:

```bash
git status --short
git add agent/context_compressor.py tests/agent/test_context_compressor_cheap_cleanup.py
git commit -m "feat(compression): skip summary after effective cheap cleanup"
```

---

### Task 5: Compression audit block

**Files:**
- Modify: `agent/context_compressor.py`
- Modify: `tests/agent/test_context_compressor_cheap_cleanup.py`

**Interfaces:**
- Consumes: `self._last_cheap_tool_cleanup_audit`
- Produces: `record["cheap_tool_result_cleanup"]`

- [ ] **Step 1: Add audit assertions**

Append:

```python

def test_audit_records_cleanup_block_for_disabled_feature():
    c = _compressor()
    c.bind_session_state(session_db=None, session_id="sess-1")
    record = c._build_compression_audit_record(
        result="skipped",
        entrypoint="auto",
        input_messages=2,
        output_messages=2,
        before_estimate=10,
        after_estimate=10,
        before_messages=[{"role": "user", "content": "hi"}],
        after_messages=[{"role": "user", "content": "hi"}],
    )

    block = record["cheap_tool_result_cleanup"]
    assert block["enabled"] is False
    assert block["applied"] is False
    assert block["result"] in {"not_attempted", "disabled"}


def test_audit_records_no_tail_clearing_after_applied_cleanup(monkeypatch):
    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_db=None, session_id="sess-1")
    monkeypatch.setattr(c, "_generate_summary", lambda turns, focus_topic=None: "## Current Work\n- compressed\n")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "x" * 4000, row_id=11),
        {"role": "user", "content": "tail user"},
        {"role": "assistant", "content": "tail assistant"},
    ]

    c.compress(messages, current_tokens=80_000, force=False, trigger_reason="token_threshold")

    block = c._last_compression_audit_record["cheap_tool_result_cleanup"]
    assert block["applied"] is True
    assert block["protected_tail_cleared_count"] == 0
    assert block["raw_tool_results_restored_for_summary"] is False
```

- [ ] **Step 2: Run and verify audit block is missing**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py::test_audit_records_cleanup_block_for_disabled_feature tests/agent/test_context_compressor_cheap_cleanup.py::test_audit_records_no_tail_clearing_after_applied_cleanup -q -o 'addopts='
```

Expected: failure for missing `cheap_tool_result_cleanup` key.

- [ ] **Step 3: Add audit block to `_build_compression_audit_record()`**

In `_build_compression_audit_record()`, before `return {`, compute:

```python
        cheap_cleanup_audit = dict(getattr(self, "_last_cheap_tool_cleanup_audit", {}) or {})
        if not cheap_cleanup_audit:
            cheap_cleanup_audit = {
                "enabled": bool(getattr(self, "cheap_tool_result_cleanup", CheapToolResultCleanupConfig()).enabled),
                "applied": False,
                "result": "not_attempted",
            }
```

In the returned dict, add:

```python
            "cheap_tool_result_cleanup": cheap_cleanup_audit,
```

Keep existing `tools` block for compatibility.

- [ ] **Step 4: Ensure source audit carries view fields**

In `_serialize_for_summary()`, after `source_audit` is finalized, verify `_last_summary_source_audit` includes:

```python
"view": "cleaned_after_cheap_tool_result_cleanup"
"raw_tool_results_restored_for_summary": false
```

The test in Task 3 should already cover this indirectly. Add a direct assertion to `test_summary_source_uses_cleaned_tool_result_when_cleanup_applies`:

```python
    assert c._last_summary_source_audit["view"] == "cleaned_after_cheap_tool_result_cleanup"
    assert c._last_summary_source_audit["raw_tool_results_restored_for_summary"] is False
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py -q -o 'addopts='
```

Commit:

```bash
git status --short
git add agent/context_compressor.py tests/agent/test_context_compressor_cheap_cleanup.py
git commit -m "feat(compression): audit cheap tool cleanup"
```

---

### Task 6: Persistence-path regression with archived rows

**Files:**
- Modify: `tests/agent/test_context_compressor_cheap_cleanup.py`
- Modify: `agent/context_compressor.py` only if the regression exposes missing row id handling

**Interfaces:**
- Verifies: persisted handles point to recoverable existing session rows

- [ ] **Step 1: Add archived-row recovery regression**

Append:

```python

def test_persisted_handle_points_to_existing_message_row(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session(title="cheap cleanup test")
    raw_messages = [
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        _tool("old-1", "recover me"),
    ]
    db.append_messages(session_id, raw_messages)
    loaded = db.get_messages(session_id)
    tool_row = next(msg for msg in loaded if msg.get("role") == "tool")

    cfg = CheapToolResultCleanupConfig(enabled=True, keep_recent=0, min_tokens_saved=1)
    c = _compressor(cheap_tool_result_cleanup=cfg)
    c.bind_session_state(session_db=db, session_id=session_id)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        _assistant_call("old-1"),
        tool_row,
        {"role": "user", "content": "tail user"},
    ]

    result = c._cleanup_old_tool_results(messages, summarize_start=2, compress_end=4)

    cleaned_content = result.messages[3]["content"]
    assert f"hermes://session/{session_id}/message/{tool_row['id']}" in cleaned_content
    archived_window = db.get_messages_around(session_id, int(tool_row["id"]), window=1)
    assert any(msg.get("content") == "recover me" for msg in archived_window["window"])
```

If `SessionDB.append_messages()` has a different signature in this repo, use the existing helper from nearby session persistence tests. Do not skip the regression; adapt it to the actual SessionDB API.

- [ ] **Step 2: Run regression**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py::test_persisted_handle_points_to_existing_message_row -q -o 'addopts='
```

Expected: pass if loaded messages already carry row ids.

- [ ] **Step 3: Add row-id resolver only if needed**

If loaded tool messages do not carry `id`, add this helper and use it inside `_cheap_tool_result_replacement()` before sentinel fallback:

```python
    def _resolve_tool_result_row_id(self, msg: dict[str, Any]) -> int | None:
        row_id = msg.get("id")
        try:
            parsed = int(row_id)
        except (TypeError, ValueError):
            parsed = 0
        if parsed > 0:
            return parsed

        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "_session_id", "")
        tool_call_id = msg.get("tool_call_id")
        if not session_db or not session_id or not tool_call_id:
            return None

        getter = getattr(session_db, "get_messages", None)
        if getter is None:
            return None
        try:
            rows = getter(session_id, include_inactive=True)
        except TypeError:
            rows = getter(session_id)
        except Exception:
            return None

        matches = [
            row for row in rows
            if isinstance(row, dict)
            and row.get("role") == "tool"
            and row.get("tool_call_id") == tool_call_id
        ]
        if len(matches) != 1:
            return None
        try:
            return int(matches[0].get("id"))
        except (TypeError, ValueError):
            return None
```

Then in `_cheap_tool_result_replacement()` replace direct `msg.get("id")` parsing with:

```python
        row_id_int = self._resolve_tool_result_row_id(msg) or 0
```

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
python -m pytest tests/agent/test_context_compressor_cheap_cleanup.py -q -o 'addopts='
```

Commit:

```bash
git status --short
git add agent/context_compressor.py tests/agent/test_context_compressor_cheap_cleanup.py
git commit -m "test(compression): verify archived tool result handles"
```

---

### Task 7: Broader regression suite and review

**Files:**
- No new files expected unless tests reveal required fixes.

**Interfaces:**
- Verifies: compressor behavior remains compatible with existing final-v2 protected tail and audit behavior.

- [ ] **Step 1: Run targeted compressor suites**

Run:

```bash
python -m pytest \
  tests/agent/test_context_compressor_cheap_cleanup.py \
  tests/agent/test_context_compressor.py \
  tests/agent/test_context_compressor_summary_continuity.py \
  tests/agent/test_compressor_historical_media.py \
  tests/agent/test_context_compression_persistence.py \
  -q -o 'addopts='
```

Expected: pass.

- [ ] **Step 2: Fix any real regressions**

For each failure:

1. Read the failing assertion.
2. Identify whether the test expectation or implementation violates the spec.
3. Patch the smallest implementation or test fixture change.
4. Re-run the failing test first.
5. Re-run the targeted suite from Step 1.

Do not weaken protected-tail assertions. Do not change expected cleanup semantics to preserve an old raw-summary invariant.

- [ ] **Step 3: Run config-focused tests**

Run:

```bash
python -m pytest tests/hermes_cli/test_config.py tests/hermes_cli -q -o 'addopts='
```

If this is too broad or slow, narrow to tests touching config display/defaults after seeing collection names with:

```bash
python -m pytest tests/hermes_cli --collect-only -q -o 'addopts='
```

Expected: relevant config tests pass.

- [ ] **Step 4: Commit test/regression fixes**

If Step 2 or Step 3 required fixes, commit them:

```bash
git status --short
git add <exact changed files>
git commit -m "fix(compression): preserve cleanup invariants"
```

If no fixes were required, do not create an empty commit.

- [ ] **Step 5: Run one Codex review pass**

Because this touches core compression logic and multiple files, run one adversarial review after the implementation commits:

```bash
codex review --commit HEAD
```

This is pass 1/2. Fix clear correctness bugs. Do not run a second review unless the first review exposes broad/systemic problems or forces a materially reshaped diff.

- [ ] **Step 6: Commit review fixes if any**

If fixes were made:

```bash
git status --short
git add <exact changed files>
git commit -m "fix(compression): address cheap cleanup review"
```

---

### Task 8: Local enablement and live smoke after merge approval

**Files:**
- Modify after implementation approval: `/Users/zongxin/.hermes/config.yaml`
- No repo code files expected.

**Interfaces:**
- Verifies: live config can enable feature and audit can explain behavior.

- [ ] **Step 1: Wait for Zongxin approval to enable live config**

Do not modify `/Users/zongxin/.hermes/config.yaml` until Zongxin approves implementation results and local enablement.

- [ ] **Step 2: Enable config locally**

After approval, set:

```yaml
compression:
  cheap_tool_result_cleanup:
    enabled: true
    keep_recent: 5
    min_tokens_saved: 20000
    replacement_mode: persisted_handle_or_sentinel
    skip_llm_summary_when_below_threshold: true
```

Use the repo's config editing path or a targeted patch. Do not rewrite unrelated config sections.

- [ ] **Step 3: Restart Hermes gateway normally**

Use the normal service restart path for this machine. Do not use `launchctl submit` helper.

- [ ] **Step 4: Run a synthetic smoke if natural compression is not imminent**

Create or reuse a temporary session with many old large tool results, trigger compression, then inspect `~/.hermes/logs/compression_audit.jsonl`.

Expected audit facts:

```json
{
  "cheap_tool_result_cleanup": {
    "enabled": true,
    "protected_tail_cleared_count": 0,
    "raw_tool_results_restored_for_summary": false
  }
}
```

If cleanup applies, also expect either:

```json
"result": "cheap_cleanup_only"
```

or:

```json
"summary_source_view": "cleaned_after_cheap_tool_result_cleanup"
```

- [ ] **Step 5: Report live evidence**

Report only verified evidence:

- commit range;
- tests run and pass/fail status;
- config read-back showing feature enabled;
- gateway restart evidence;
- audit row showing cleanup behavior or a clear no-op reason.

---

## Self-Review Notes

- Spec coverage: Tasks 1-8 cover config, cleanup semantics, persisted handles, keep_recent including tail, cleaned summary source, cheap-only path, audit, tests, review, and rollout.
- Red-flag scan result: no unresolved markers or unspecified implementation steps remain.
- Type consistency: `CheapToolResultCleanupConfig`, `CheapToolResultCleanupResult`, `_cleanup_old_tool_results()`, `_cheap_tool_result_replacement()`, and `_cheap_cleanup_only_allowed()` are introduced before later tasks consume them.
- Scope check: the plan stays inside Hermes compression/config/audit and does not add new model tools or duplicate output storage.
