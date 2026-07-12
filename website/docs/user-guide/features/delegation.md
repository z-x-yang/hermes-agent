---
sidebar_position: 7
title: "Subagent Delegation"
description: "Claude-like subagent types, Batch delivery, continuation, and runtime safety"
---

# Subagent Delegation

Hermes uses `delegate_task` to run isolated child agents. The model-facing contract is intentionally small: `description`, `prompt`, optional `subagent_type`, optional `run_in_background`, plus the intentional Hermes Batch extension. Runtime policy—not caller-supplied permission fields—controls tools, nesting, retention, timeouts, and provider fallback.

## Built-in subagent types

`subagent_type` accepts exactly three values:

| Type | Use it for | Lifecycle and context |
|---|---|---|
| `Explore` | Read-only code/file/source investigation | one-shot; complete governance; skips project context and workspace snapshot |
| `Plan` | Read-only implementation research and planning | one-shot; complete governance; skips project context and workspace snapshot |
| `general-purpose` | Multi-step execution, edits, tests, and permitted external actions | automatically retained after successful completion; complete governance plus project context and workspace snapshot |

Omitting `subagent_type` resolves to `general-purpose`. All profiles receive the active profile's complete `SOUL.md`, `MEMORY.md`, and `USER.md`; they do not inherit the parent transcript or previous tool results.

`Explore` and `Plan` use a runtime-enforced read-oriented tool ceiling. They can use permitted repository/file readers, no-spill web/skill readers, Notion AI with `mode=readonly`, and selected Apple Mail read tools, but no raw terminal or file writes. `general-purpose` receives only tools that survive the exact current-parent tool authority and normal tool safety contracts; it is not a no-side-effect sandbox.

There is no universal mandatory semantic result schema. `Explore` reports findings clearly and concisely, `Plan` ends with `### Critical Files for Implementation`, and the parent `prompt` should state any exact return requirements. The parent must verify important self-reported changes and claims.

## Single task

```python
delegate_task(
    description="inspect auth flow",
    prompt="Find the auth middleware. Return absolute paths and line ranges.",
    subagent_type="Explore",
    run_in_background=False,
)
```

A single task requires both `description` and `prompt`:

- `description` is a short progress label.
- `prompt` is the self-contained task, including paths, constraints, evidence requirements, and expected deliverable.
- `run_in_background` defaults to `True` for top-level calls. Set it to `False` only when the parent needs the result before continuing.

## Batch API: intentional Hermes divergence

Claude Code expresses parallelism as multiple Agent calls in one assistant message. Hermes retains a Batch API because Gateway and messaging transports benefit from one grouped lifecycle:

```python
delegate_task(
    tasks=[
        {
            "description": "inspect backend",
            "prompt": "Find the backend auth path and report evidence.",
            "subagent_type": "Explore",
        },
        {
            "description": "inspect frontend",
            "prompt": "Find the frontend auth path and report evidence.",
            "subagent_type": "Explore",
        },
    ]
)
```

A Batch is one concurrent group with one batch handle, one occupied async slot, and one consolidated completion after all children finish. Results remain ordered by task index. Batch items contain only `description`, `prompt`, and optional `subagent_type`; the whole Batch shares one top-level `run_in_background` choice.

If the background pool is full, Hermes returns a structured `rejected` result and runs no child synchronously. If the endpoint cannot deliver later messages, prepared work runs synchronously with an explicit note rather than silently changing semantics.

## Foreground, background, and timeouts

Scheduling uses only `run_in_background`:

- top-level omitted → background;
- top-level `False` → foreground wait;
- nested omitted or `False` → foreground;
- nested `True` → fail closed before child execution.

Foreground waiting and child execution have separate operator-controlled limits. Default wait/run values are Explore `900/1800` seconds, Plan `1800/3600`, and general-purpose `1800/7200`. The profile run limit applies to every child, including work dispatched directly to background. When a foreground wait limit expires, Hermes backgrounds the same future and later emits exactly one completion; it does not queue or restart the child.

## Context isolation

A delegated child starts with fresh conversation state. Make `prompt` self-contained:

```python
# Too vague
delegate_task(
    description="fix error",
    prompt="Fix the error.",
)

# Better
delegate_task(
    description="fix body parser",
    prompt="""Repository: /home/user/webapp.
Fix the TypeError in api/handlers.py: process_request() receives None from
parse_body() when Content-Type is missing. Add a regression test and run
pytest tests/api/. Return changed files and real test output.""",
    subagent_type="general-purpose",
    run_in_background=False,
)
```

`general-purpose` loads real repository rules (`.hermes.md`, `AGENTS.md`, `CLAUDE.md`, or `.cursorrules` under the normal discovery contract) and a workspace/git snapshot. `Explore` and `Plan` deliberately skip project context while retaining complete governance.

## Retained sessions and `delegate_continue`

Lifecycle is fixed by profile:

- `Explore` and `Plan` are one-shot and never retained.
- A successful `general-purpose` child is automatically retained only when the parent has a nonempty session ID and retention capacity is available.
- Failed retention is visible as `retention_status="failed"` plus `retention_error`; Hermes does not invent an `agent_id`.

A retained result includes an `agent_id`:

```python
delegate_continue(
    agent_id="<agent_id from the completed result>",
    prompt="Now add the missing regression test and rerun the focused suite.",
    run_in_background=False,
)
```

`delegate_continue` accepts only `agent_id`, `prompt`, and optional `run_in_background`. It keeps the original profile/workspace and intersects original and current exact tool authority. The process-local store is TTL/count/byte bounded and restart-ephemeral. Notion and Apple Mail sensitive read results remain `HANDLE_ONLY` in retained history. Claim generation and cancellation prevent interrupted or timed-out late workers from committing stale history.

## Runtime-derived nesting

Nested delegation is runtime-derived, not caller-selected. A child receives `delegate_task` only when all of these hold:

1. it is `general-purpose`;
2. the current parent actually exposes `delegate_task` under its exact resolved policy;
3. `delegation.orchestrator_enabled` is true;
4. `child_depth < max_spawn_depth`.

`Explore` and `Plan` never delegate. `delegate_continue` and `clarify` remain unavailable to children. The default `max_spawn_depth=1` keeps delegation flat; raising it permits another GP layer only under the same gates.

## Interrupts and durability

`/agents` (alias `/tasks`) shows active and recent subagents. `/stop` and shutdown propagate interruption to foreground/background children and continuations. Background delegation and retained sessions are process-local, not durable jobs: use cron or a managed background process for work that must survive `/new`, process exit, or Gateway restart.

## Configuration

Concurrency, depth, kill switch, per-profile model/provider and wait/run timeouts, and retained-store TTL/count/byte budgets live under `delegation` in `~/.hermes/config.yaml`. They are operator controls, not model-facing fields. See [Configuration → Delegation](/user-guide/configuration#delegation).
