# Claude-like Cheap Tool Result Cleanup for Hermes

Date: 2026-07-06
Status: Design spec for review
Owner: Evelyn / Zongxin
Target repo: `~/.hermes/hermes-agent`

## 1. Goal

Add a Claude Code-like cheap cleanup stage to Hermes compression:

1. When auto-compression is triggered, first remove old tool result bodies from the active context without calling an LLM.
2. Preserve the protected tail exactly: tail tool results are never cleared.
3. Use Claude-like defaults where we have evidence from Claude Code 2.1.195:
   - keep recent tool results: `5`
   - minimum estimated token savings before applying cleanup: `20000`
   - sentinel fallback text: `[Old tool result content cleared]`
   - model-visible replacement via `<persisted-output>...</persisted-output>` when a recoverable handle exists.
4. If cheap cleanup alone brings the active payload below the auto-compression threshold, skip LLM summarization for that auto-compression attempt.
5. If LLM summarization still runs, the summary source is the cleaned view, not the original raw tool-result view.
6. Add content-free audit fields that make cleanup behavior debuggable after the fact.

Done means: Hermes can run with this feature enabled from config, preserve tail invariants, avoid duplicate storage of large tool results, produce useful audit records, pass regression tests, and commit the change.

## 2. Non-goals

- Do not restore cleared old tool results automatically into the LLM summarizer.
- Do not compact, truncate, or replace any protected-tail tool result.
- Do not add a new always-present core model tool unless existing session retrieval is insufficient in practice.
- Do not copy large tool outputs into new sidecar files in v1.
- Do not remove the existing summary-source overflow guard; keep it as a last-resort safety net for pathological prompts.
- Do not change Hermes memory, skill, ledger, or Notion behavior.

## 3. Current Hermes facts this design relies on

Observed in current code:

- `ContextCompressor.compress()` currently preserves `raw_messages` and uses them as summary source.
- Main-flow old tool-result pruning is intentionally disabled; `_prune_old_tool_results()` is a legacy helper and not called in the main compression path.
- Protected tail boundary is selected before summary generation using continuation-payload token accounting and `protect_last_n` as a user/assistant message floor.
- `SessionDB.archive_and_compact()` is non-destructive: pre-compaction active rows become `active=0, compacted=1`; compacted messages are inserted as new active rows.
- Archived rows remain in the same SQLite `state.db`, remain searchable, and can be loaded with `include_inactive=True` or anchored by message row id.

This means Hermes already has a durable original source for old tool results. We should point to archived message rows instead of duplicating large outputs into files.

## 4. Design summary

Introduce a new compression sub-stage:

```text
auto-compression trigger
  -> compute protected head/tail boundaries from original payload
  -> compute actual summary window
  -> cheap cleanup old tool results outside protected tail
  -> re-estimate active payload
      -> if below threshold and trigger is eligible: persist cleaned view, skip LLM summary
      -> otherwise: run LLM summary on cleaned view
  -> audit cleanup + summary-source semantics
```

Key invariant:

> When enabled and applied, the cleaned view becomes the source of truth for subsequent live context and for summary generation.

## 5. Config

Add a nested config section under `compression`:

```yaml
compression:
  cheap_tool_result_cleanup:
    enabled: false
    keep_recent: 5
    min_tokens_saved: 20000
    replacement_mode: persisted_handle_or_sentinel
    skip_llm_summary_when_below_threshold: true
```

### Defaults and deployment

- Code/default config should keep `enabled: false` to avoid surprising upstream users.
- Zongxin's live config can enable it after tests pass.
- Invalid values fail closed to the safe default and log/audit the normalization:
  - `keep_recent` coerced to integer `>= 0`; default `5`.
  - `min_tokens_saved` coerced to integer `>= 0`; default `20000`.
  - `replacement_mode` supports only `persisted_handle_or_sentinel` in v1.
  - `skip_llm_summary_when_below_threshold` coerced as boolean; default `true`.

## 6. Replacement semantics

### 6.1 Preferred replacement: persisted handle

If a tool result has a durable archived source row or can be matched to one, replace its content with:

```text
<persisted-output>
Tool result archived at: hermes://session/<session_id>/message/<row_id>
Use session_search(session_id="<session_id>", around_message_id=<row_id>, window=1, role_filter="tool") to view it if needed.
</persisted-output>
```

This is model-visible text inside the replacement tool result content. It tells a later agent where the old tool result can be recovered, but it does not automatically reload the raw result into context.

The handle should point to the existing session row, not to a duplicated file. Disk growth is limited to the short replacement text in the new active context and content-free audit metadata.

### 6.2 Fallback replacement: sentinel

If a safe handle cannot be created, use the Claude Code sentinel:

```text
[Old tool result content cleared]
```

Fallback is allowed only with explicit audit. It must not be silent.

Expected fallback cases:

- tool result is not yet persisted and cannot be matched safely;
- malformed or missing `tool_call_id`;
- multimodal/image/document result where a text row handle would be misleading;
- resolver lookup error after bounded retry.

### 6.3 What is preserved

For every cleaned tool result, preserve:

- message role (`tool`);
- `tool_call_id`;
- message order;
- neighboring assistant `tool_calls`;
- any non-content metadata needed for provider role/tool-call pairing.

Only the visible `content` body changes.

## 7. `keep_recent` semantics

`keep_recent` counts recent **eligible** tool results globally across the provider-visible history selected for cheap cleanup. The protected tail no longer has higher priority: if the retained tail contains more than `keep_recent` eligible tool results, older tail tool results are cleared too.

Algorithm:

```python
eligible = eligible_tool_results(messages[first_non_system_index:])
keep_indices = newest(eligible, keep_recent)
clear_candidates = eligible - keep_indices
```

Consequences:

- `keep_recent=5` means keep the newest 5 eligible tool results, not newest 5 files and not newest 5 tail results plus extras.
- If protected tail contains 10 eligible tool results, only the newest 5 stay raw and the older 5 are replaced.
- Ineligible tools do not count against `keep_recent` and are never cleared by this pass.
- This intentionally follows Claude Code's `keepRecent=5` behavior rather than Hermes' older protected-tail-wins invariant.

## 8. Eligibility and token-savings gate

A tool result is eligible when all conditions hold:

- message role is `tool`;
- assistant tool name is in the Claude Code-aligned whitelist:
  - `read_file` (Claude `Read`)
  - `terminal` (Claude `Bash` / `PowerShell`)
  - `search_files` (Claude `Grep` / `Glob`)
  - `web_search` (Claude `WebSearch`)
  - `web_extract` (Claude `WebFetch`)
  - `patch` (Claude `Edit`)
  - `write_file` (Claude `Write`)
- content is non-empty;
- content is not already `[Old tool result content cleared]`;
- content is not already a `<persisted-output>...</persisted-output>` block;
- replacing it does not create an orphan tool call/result pair.

Generic Hermes tools (`todo`, `delegate_task`, `clarify`, `cronjob`, `session_search`, memory/fact-store, MCP/Notion/email/Discord tools, `browser_*`, `process`, `execute_code`, etc.) are intentionally out of scope unless future Claude evidence expands the list.

Before mutating messages, estimate token savings for `clear_candidates`:

```python
estimated_saved_tokens = sum(original_tool_result_tokens - replacement_tokens)
```

Apply cleanup only if:

```python
estimated_saved_tokens >= min_tokens_saved
```

If estimated savings are below threshold, do not mutate messages; run existing compression flow unchanged.

## 9. Interaction with LLM summary

### 9.1 Summary source

When cleanup is applied:

- `turns_to_summarize` must be built from the cleaned message list.
- `raw_messages` must not be used as the summary source for cleaned tool results.
- The audit must record `summary_source.view = "cleaned_after_cheap_tool_result_cleanup"` and `raw_tool_results_restored_for_summary = false`.

### 9.2 Cheap-only completion

If cleanup is applied and the full active payload estimate drops below the active compression threshold, auto-compression may complete without LLM summary. This is allowed even when the retained-tail/message-floor policy leaves no summarizable middle window; in that case cheap cleanup is the only deterministic relief layer.

Allowed only when all are true:

- entrypoint is `auto`;
- trigger reason is token pressure / context pressure, not manual `/compress`;
- no explicit `focus_topic` was supplied;
- trigger is not purely hard-message-limit hygiene;
- cleaned active payload is below threshold;
- cleaned message list passes tool-pair sanitization and role alternation checks.

Result category in audit: `cheap_cleanup_only`.

Manual `/compress` should still run full summary after cleanup, because the user explicitly requested compaction rather than just payload relief.

## 10. Archived handle resolution

### 10.1 Preferred source

Use existing session storage:

- `session_id`: current session id bound to the compressor.
- `row_id`: SQLite `messages.id` for the original tool message.

The compressor should prefer row ids already present on loaded message dicts. If absent, it may resolve before archive by matching active rows in the current session using stable identifiers:

1. `role = 'tool'`
2. `tool_call_id`
3. insertion order / message index
4. content hash as a bounded disambiguation signal, without writing the hash of raw content into the replacement text.

If matching is ambiguous, use sentinel and audit the fallback.

### 10.2 Reader path

Do not add a new always-on model tool in v1. Use existing `session_search` as the recovery path in the placeholder.

The handle text tells the future agent to call:

```python
session_search(
    session_id="<session_id>",
    around_message_id=<row_id>,
    window=1,
    role_filter="tool",
)
```

If testing shows this is awkward or unavailable in important contexts, add a later narrow resolver command/tool behind the existing session-search toolset, not as a new broad core tool.

## 11. Audit schema

Add a top-level block to `context_compression` records:

```json
"cheap_tool_result_cleanup": {
  "enabled": true,
  "applied": true,
  "result": "applied",
  "scope": "eligible_tool_results_across_provider_history",
  "view": "cleaned_after_cheap_tool_result_cleanup",
  "eligible_tool_names": ["patch", "read_file", "search_files", "terminal", "web_extract", "web_search", "write_file"],

  "keep_recent": 5,
  "min_tokens_saved": 20000,
  "tail_tool_result_count": 7,
  "extra_pre_tail_keep_count": 0,

  "candidate_count": 41,
  "eligible_tool_result_count": 41,
  "ineligible_tool_result_count": 6,
  "clear_candidate_count": 36,
  "kept_recent_count": 5,
  "kept_recent_pre_tail_count": 0,
  "cleared_count": 36,
  "protected_tail_cleared_count": 4,

  "pre_cleanup_tokens_estimate": 260000,
  "post_cleanup_tokens_estimate": 218000,
  "tokens_saved_estimate": 42000,

  "replacement_counts": {
    "persisted_handle": 31,
    "sentinel": 3
  },
  "sentinel_fallback_reasons": {
    "missing_row_id": 2,
    "multimodal_content": 1
  },

  "summary_source_view": "cleaned_after_cheap_tool_result_cleanup",
  "raw_tool_results_restored_for_summary": false,
  "llm_summary_skipped_after_cleanup": false,
  "llm_summary_ran_on_cleaned_view": true,

  "cleared_tool_call_id_hashes": ["bounded-short-hashes"]
}
```

Audit rules:

- Never log raw tool content.
- Never log raw tool arguments.
- Never log full unbounded tool ids if they can bloat records; use short stable hashes, bounded list length.
- Always record `protected_tail_cleared_count`; value may be `>0` when old eligible tool results inside the retained tail are beyond the global keep-recent window.
- Record fallback reasons whenever sentinel is used.
- If cleanup is disabled or savings are below threshold, emit the block with `applied=false` and a clear `result` such as `disabled` or `below_min_tokens_saved`.

## 12. Existing audit compatibility

Existing fields remain:

- `tools.pruned_old_tool_results`
- `tools.pruned_before_boundary_count`
- `summary_source`
- `message_accounting`
- `retained_tail`

Recommended update:

- Keep legacy `tools.pruned_old_tool_results` at `0` unless the new cleanup applied.
- Add the richer nested block as the authoritative source for new behavior.
- Add `summary_source.view` and `summary_source.raw_tool_results_restored_for_summary` when cleanup applies.

## 13. Implementation touchpoints

Expected files:

- `agent/context_compressor.py`
  - config fields on `ContextCompressor`;
  - helper to identify Claude Code-aligned eligible tool results;
  - helper to compute global `keep_recent` across eligible results, including retained-tail tool results;
  - helper to build persisted handles or sentinel replacements;
  - integration before `_generate_summary()`;
  - cheap-only return path;
  - audit block construction.

- `agent/agent_init.py`
  - parse `compression.cheap_tool_result_cleanup` config;
  - pass normalized values into `ContextCompressor`.

- `hermes_cli/config.py`
  - add default config schema/comments;
  - include config display if appropriate.

- `tests/agent/test_context_compressor.py` or focused new test file
  - unit and integration regressions.

No change to public API or external provider protocol is required.

## 14. Test plan

### 14.1 Config tests

- Missing config keeps feature disabled.
- Enabled config passes normalized defaults to `ContextCompressor`.
- Invalid values fail closed and do not crash startup.

### 14.2 Cleanup eligibility tests

- Disabled feature leaves messages unchanged.
- Savings below `min_tokens_saved` leaves messages unchanged and audit says `below_min_tokens_saved`.
- Already-sentinel and already-persisted-output messages are idempotent.
- Non-tool messages are never replaced.
- Assistant `tool_calls` are preserved.

### 14.3 Protected tail tests

- Large protected-tail tool result remains raw.
- `protected_tail_cleared_count == 0` in audit.
- Tail boundary is identical before and after cleanup.
- Tool-boundary alignment still preserves valid tool call/result pairing.

### 14.4 `keep_recent` tests

- Tail has 5 tool results: no pre-tail tool result is kept for `keep_recent`.
- Tail has 2 tool results: newest 3 pre-tail candidates are kept, older candidates are cleared.
- Tail has 0 tool results: newest 5 pre-tail candidates are kept.
- Tail has more than 5 tool results: all tail tool results remain raw; pre-tail candidates may be cleared.

### 14.5 Persisted handle tests

- Tool message with row id becomes `<persisted-output>` with `hermes://session/.../message/...`.
- Missing or ambiguous row id falls back to sentinel and audits fallback reason.
- No duplicate large sidecar file is created.
- Archived row remains retrievable through session storage after `archive_and_compact()`.

### 14.6 Summary-source tests

- When cleanup applies and LLM summary runs, fake summarizer receives sentinel/persisted-output blocks, not original large tool output.
- Audit says `raw_tool_results_restored_for_summary=false`.
- Existing summary-source overflow fallback still works if cleaned serialization is still too large.

### 14.7 Cheap-only tests

- Auto token-threshold trigger below threshold after cleanup returns `cheap_cleanup_only` and does not call `_generate_summary()`.
- Manual `/compress` still calls `_generate_summary()`.
- Hard-message-limit trigger still calls `_generate_summary()` because cleanup does not reduce message count.

### 14.8 Persistence/audit tests

- Compression audit contains the new block for disabled, below-threshold, applied, and cheap-only cases.
- Persist companion audit still records output row ids.
- User-message ground-truth audit is only written when LLM summary actually runs.

## 15. Rollout plan

1. Implement behind `compression.cheap_tool_result_cleanup.enabled=false` default.
2. Run targeted pytest for compressor/config/persistence tests.
3. Run a local smoke compression with synthetic tool-heavy messages.
4. Commit the implementation.
5. Enable in Zongxin's local config.
6. Restart gateway using normal Hermes service restart path.
7. Inspect the next natural compression audit for:
   - `applied=true` or a clear non-applied reason;
   - `protected_tail_cleared_count=0`;
   - `summary_source_view=cleaned_after_cheap_tool_result_cleanup` when applied;
   - `raw_tool_results_restored_for_summary=false`;
   - cheap-only behavior only on eligible auto triggers.

## 16. Risks and mitigations

### Risk: losing useful old tool details from summaries

Mitigation: persisted handles keep a model-visible recovery path; summary-source audit makes the cleaned-view semantics explicit.

### Risk: breaking provider tool-call pairing

Mitigation: replace only `tool.content`, preserve `tool_call_id`, run existing `_sanitize_tool_pairs()`, and add pairing regression tests.

### Risk: hidden fallback from handle to sentinel

Mitigation: sentinel fallback is allowed but always audited with reason counts.

### Risk: model tries to read many archived handles and re-bloats context

Mitigation: handles are explicit and on-demand; future behavior is visible as normal tool calls. The cheap cleanup does not automatically rehydrate raw results.

### Risk: session_search recovery is too awkward

Mitigation: v1 uses existing `session_search` to avoid adding an always-on core tool. If real usage shows poor ergonomics, add a narrow resolver later under the existing session-search surface.

## 17. Review checklist before implementation

- The protected tail is stronger than `keep_recent`.
- `keep_recent` counts tail tool results first.
- The summary source is cleaned when cleanup applies.
- Raw old tool results are not automatically restored for summarization.
- No large duplicate tool-output files are written in v1.
- Audit can explain every cleanup/no-cleanup decision without content leakage.
- Manual `/compress` still means full summary.
