# Claude-like Subagent Simplification — Traceability

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

- `0479b5a7b` — profile-specific prompts and GP project context.
- `88c8cd9ca` — static single/Batch schema.
- `39fedc48d` — runtime-derived nesting; role state removed.
- `04623fd03` — profile-fixed lifecycle and boolean continuation input.
- `tools/delegate_tool.py`
- `tools/delegate_continue_tool.py`
- `tools/subagent_profiles.py`
- `tools/subagent_sessions.py`
- `tools/async_delegation.py`
- `agent/subagent_tool_policy.py`

## Fresh evidence before final review

- High-signal delegation/lifecycle/race gate: **340 passed**.
- Documentation contract RED→GREEN: simplified EN/ZH contract assertion passes.
- Task-specific Ruff and `py_compile` gates passed after Tasks 1–4.
- Static readback confirms no dynamic schema override, no retained role field, and exact simplified function signatures.

These are intermediate candidate facts. Task 6 must still add current-vs-detached-baseline full `tests/tools` node-ID differential, docs production build, final static checks, and independent review findings before any merge/restart decision.

## Rollout boundary

No merge to local main and no Gateway restart is authorized by this artifact. Final handoff must explicitly state current branch/commit, test and baseline differential, independent review verdicts, remaining caveats, rollback point, and ask Zongxin before merge/restart.
