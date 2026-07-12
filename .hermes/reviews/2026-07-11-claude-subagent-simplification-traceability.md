# Claude-like Subagent Simplification — Traceability

> **HISTORICAL REVIEW SNAPSHOT.** The candidate was later merged and deployed; the original branch/rollout status below is preserved only as point-in-time evidence. Use the live registered schema and current delegation docs for operational instructions.

Date: 2026-07-11
Branch: `feat/claude-subagent-simplification`
Scope boundary: candidate only; **not merged to local main and Gateway not restarted**.

## Decision

Hermes now aligns its model-facing single-task contract with Claude Code 2.1.207 while retaining only one deliberate transport-level extension: Batch. Claude parity does **not** justify deleting Hermes runtime security, governance, data-source, provider-fallback, timeout, race, or retention hardening.

The previous rollout report, `2026-07-11-claude-subagent-rollout-readiness-zh.md`, is superseded because it described dynamic schema text, a universal 12-field result checklist, caller-selected role/retention, and three-state scheduling.

## Model-facing contract

### Single

```python
delegate_task(
    description: str,
    prompt: str,
    subagent_type: Optional[Literal["Explore", "Plan", "general-purpose"]] = None,
    run_in_background: Optional[bool] = None,
)
```

### Batch — INTENTIONAL Hermes divergence

```python
delegate_task(
    tasks=[
        {
            "description": str,
            "prompt": str,
            "subagent_type": Optional[str],
        }
    ],
    run_in_background: Optional[bool] = None,
)
```

One Batch is one concurrent group, one async handle/slot, and one consolidated completion. Batch is retained for Gateway/Discord completion UX; it is not claimed as exact Claude parity.

### Continuation

```python
delegate_continue(
    agent_id: str,
    prompt: str,
    run_in_background: Optional[bool] = None,
)
```

Top-level omission defaults background. Nested omission defaults foreground; nested true rejects before child execution.

## Classification

| Capability | Classification | Evidence / rationale |
|---|---|---|
| `description` + `prompt` + optional `subagent_type`/`run_in_background` | EXACT model-facing alignment | `tools/delegate_tool.py::DELEGATE_TASK_SCHEMA`, `delegate_task` |
| Exactly Explore / Plan / general-purpose | EXACT built-in profile alignment | `tools/subagent_profiles.py` |
| No universal semantic result object | EXACT principle | profile-specific final guidance; parent prompt defines deliverable |
| Plan `### Critical Files for Implementation` | EXACT profile guidance | `tools/subagent_profiles.py::PLAN_FINAL` |
| Batch API | INTENTIONAL | Gateway/messaging grouped lifecycle |
| Complete current `SOUL.md` / `MEMORY.md` / `USER.md` for every profile | INTENTIONAL | Evelyn governance requirement across providers |
| GP repository project context + workspace/git snapshot | INTENTIONAL / parity-supporting | real repo rules are needed for GP execution |
| Explore/Plan skip project context but keep complete governance | INTENTIONAL | read-oriented isolation without losing Evelyn policy |
| Notion/Mail read access for every profile | INTENTIONAL | user data-source requirement; readonly tool ceiling for Explore/Plan |
| Explore/Plan no raw terminal or writes | INTENTIONAL | Hermes safety profile |
| Runtime-derived nested GP delegation | INTENTIONAL | exact authority + kill switch + depth; no caller role protocol |
| Process-local bounded continuation | INTENTIONAL | Gateway follow-up UX; restart-ephemeral |
| Provider fallback per child attempt | INTENTIONAL | provider-agnostic Hermes runtime |
| Independent profile wait/run timeouts | INTENTIONAL | long-task safety and liveness |

## Removed complexity

- `goal` / `context` model-facing split;
- caller `role=leaf|orchestrator`;
- caller `retain_session`;
- `scheduling=auto|foreground|background` and direct-Python compatibility branch;
- dynamic delegation schema-description rebuild;
- universal 12-field prompt checklist;
- profile capability booleans and context-policy metadata;
- `agent/subagent_context_policy.py` and trusted-project-route capsule;
- role fields in child results, async metadata, and retained records.

## Preserved hardening

The simplification leaves these runtime gates intact:

- exact parent authority ceiling and `ToolPolicyDescriptor` identity;
- normalized/frozen arguments and argument-sensitive effects;
- Tool Search unwrap and middleware-mutation reauthorization;
- final registry-lock TOCTOU check;
- backend-zero rejection and Notion `mode=readonly` enforcement;
- Mail write-name exclusion for read-only profiles;
- provider fallback with complete governance and payload-fit checks per attempt;
- async capacity rejection without synchronous fallback;
- continuation claim generation/cancellation and late-worker commit rejection;
- workspace-change warning and visible retention failure;
- Notion/Mail `HANDLE_ONLY` retained-history projection;
- profile-specific foreground wait and run timeout contracts.

## Code anchors and scoped commits

- `284e15afd` — profile-specific prompts and GP project context.
- `562dfdf2c` — static single/Batch schema.
- `39fedc48d` — runtime-derived nesting; role state removed.
- `04623fd03` — profile-fixed lifecycle and boolean continuation input.
- `9e1e4c8c2` — EN/ZH docs, bundled skill, and this trace.
- `04df3d54d` — live continuation adapter fix and UI/ACP/Desktop consumer migration.
- `tools/delegate_tool.py`
- `tools/delegate_continue_tool.py`
- `tools/subagent_profiles.py`
- `tools/subagent_sessions.py`
- `tools/async_delegation.py`
- `agent/subagent_tool_policy.py`

## Fresh verification evidence

- Final expanded high-signal gate (delegation, continuation, provider/effect/race, Tool Search/MCP, skills/media dispatch, compression/display/plugin/CLI consumers): **1438 passed, 1 skipped, 1 known config-sensitive node deselected** in 48.77s.
- Live continuation adapter regression plus Python UI/hook consumer gate: **65 passed**; the regression first reproduced the reviewer finding as `TypeError: ... unexpected keyword argument 'scheduling'`, then passed after `run_agent.py` forwarded `run_in_background`.
- ACP renderer gate: **72 passed**.
- Desktop stream consumer: **6 passed**; Desktop TypeScript typecheck and focused ESLint exit 0.
- Final full current `tests/tools`: **20 failed, 7577 passed, 64 skipped** in 377.28s.
- Detached `00cd0985b` baseline `tests/tools`: **20 failed, 7607 passed, 64 skipped** in 404.43s.
- The final current and detached-baseline failed node-ID sets are **identical**. They contain only the same 20 pre-existing/config/environment-sensitive nodes; no delegation-owned node fails. An earlier current run had three fewer Tool Search failures, demonstrating those nodes are order-sensitive rather than branch regressions.
- The unrelated atomic environment-snapshot concurrency test independently failed 3/3 in both current and baseline checkouts; it is not delegation-owned and did not fail in the final full run.
- Documentation contract: **17 passed**; EN and zh-Hans Docusaurus production builds both emitted `[SUCCESS]` (repository-wide pre-existing broken-anchor warnings remain).
- Final Ruff, `py_compile`, focused Desktop ESLint/typecheck, baseline-range `git diff --check`, and worktree `git diff --check` all exit 0.
- Static readback confirms exact simplified function/schema signatures, no dynamic delegation schema override, no retained role field, and profile-specific prompt guidance.

## Independent review and controller disposition

Two bounded fresh-context Codex passes used the entire review budget:

1. correctness/security — reviewer ran 32 focused checks and returned `BLOCKED` on one High finding;
2. design-contract — reviewer ran 182 focused checks, classified the snapshot `EXACT 6 / INTENTIONAL 12 / PARTIAL 1 / MISSING 0 / DIVERGENT 1`, and returned `BLOCKED` on the same finding plus the then-uncommitted trace.

The shared blocker was real: `run_agent.AIAgent._dispatch_delegate_continue()` still passed removed `scheduling=` to the simplified continuation function. Controller reproduction reached the real adapter and failed before the fix. Commit `04df3d54d` now forwards `run_in_background`; the same adapter test and the expanded high-signal gate pass. Reviewer non-blocking notes about the stale single-task spinner and capacity-fallback comment were also verified and fixed. UI/ACP/Desktop/compression/plugin consumers were swept for the removed `goal`/`role` surface and migrated with behavioral tests.

No third Codex pass was run because the approved two-pass ceiling was exhausted. Controller-owned final disposition after the verified fixes is `EXACT 8 / INTENTIONAL 12 / PARTIAL 0 / MISSING 0 / DIVERGENT 0`; there are no unresolved review blockers.

Independence profile: reviewer context was fresh/different; model family/provider route relative to the controller were not observable; no human/domain-expert sign-off was present.

## Rollout boundary

No merge to local main and no Gateway restart is authorized by this artifact. Final handoff must explicitly state current branch/commit, test and baseline differential, independent review verdicts, remaining caveats, rollback point, and ask Zongxin before merge/restart.
