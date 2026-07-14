---
sidebar_position: 7
title: "Subagent Delegation"
description: "Claude-like subagent types, Batch delivery, continuation, and runtime safety"
---

# Subagent Delegation

Hermes uses `delegate_task` to run isolated child agents. The model-facing contract is intentionally small: `description`, `prompt`, optional `subagent_type`, optional `run_in_background`, the single-Reviewer-only `review_root`, plus the intentional Hermes Batch extension. Runtime policy—not caller-supplied permission fields—controls tools, nesting, retention, timeouts, and provider fallback.

## Built-in subagent types

`subagent_type` accepts exactly four values:

| Type | Use it for | Lifecycle and context |
|---|---|---|
| `Explore` | Read-only code/file/source investigation | one-shot; lean Core Contract + task capsule; skips personal and project context |
| `Plan` | Read-only implementation research and planning | one-shot; Core Contract + controller-selected task/project summary; skips personal governance |
| `Reviewer` | Fresh-context independent review of a scoped code change | one-shot; foreground by default; repository context + ordinary self-contained prompt; ordinary final response |
| `general-purpose` | Multi-step execution, edits, tests, and permitted external actions | automatically retained after successful completion; Core Contract plus project context and workspace snapshot |

Like Claude Code, each canonical profile owns the routing `description` that tells the parent when to select it. `Reviewer` uses the same ordinary `prompt` field as every other profile; it has no hidden JSON grammar or profile-specific completion tool. Its profile system instructions preload a fixed runtime-owned review method: establish scope and intended behavior, inspect callers/callees plus data/error/cleanup paths, trace runtime wiring and false-green tests, and consider reuse, simplification, and performance only when they expose a concrete defect. This is profile text, not dynamic skill discovery or automatic inheritance of another product's bundled review command. `review_root` remains a top-level Hermes adaptation that binds one local worktree when the current workspace is not the intended target.

Omitting `subagent_type` resolves to `general-purpose`. Children do not inherit the parent transcript, parent tool results, or the active profile's complete `SOUL.md`, `MEMORY.md`, or `USER.md`. `general-purpose` and `Reviewer` load repository rules and a workspace snapshot from their bound root; this normative project context is separate from personal memory or implementation-session rationale.

`Explore` and `Plan` use a runtime-enforced read-oriented tool ceiling. They can use permitted repository/file readers, no-spill web/skill readers, Notion AI with `mode=readonly`, and selected Apple Mail read tools, but no raw terminal or file writes. `Reviewer` requires `read_file`, `search_files`, and `terminal`; readonly/no-spill web tools are added only when the parent currently has that authority. It has no named Notion, Mail, session/memory, MCP, browser, write/patch, process/execute-code, or delegation tools. Raw terminal is deliberately Claude-like, so Reviewer is **not** a mechanical no-write or no-external-side-effect sandbox: its system contract forbids edits, installs, Git mutation, publishing, and private-source access, and the controller must verify the worktree afterward. `general-purpose` likewise receives only tools that survive exact current-parent authority and normal tool safety contracts.

All profiles use prompt-defined return contracts and ordinary final responses. Reviewer should report only newly introduced, evidence-backed Critical or Important candidate blockers with file/line evidence and concrete failure paths; a concise no-findings final is valid. The parent/controller must reproduce or falsify every candidate before treating it as true.

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
- `run_in_background` defaults to the selected profile: `Reviewer` is foreground by default; other top-level profiles default to background. An explicit boolean always wins.

### Reviewer

Give Reviewer an ordinary self-contained task, just as you would a Claude Code custom agent:

```python
delegate_task(
    description="independent auth review",
    prompt="""Review the committed auth-race fix at HEAD.
Inspect src/auth and tests/auth against this contract: no token may refresh
twice concurrently and the public API must not change. Fresh verification:
`pytest tests/auth -q` returned `18 passed`.

Report only newly introduced Critical or Important candidate blockers with
file:line evidence and a concrete failure path. Do not edit the checkout.""",
    subagent_type="Reviewer",
    review_root="/absolute/path/to/local/worktree",  # optional
)
```

`review_root` is a controller-bound tool argument, not prompt syntax. It is valid only for a top-level single Reviewer and must resolve exactly to an existing absolute local Git worktree root; relative paths, repo subdirectories, Batch/nested use, non-Reviewer use, and remote/cluster roots fail closed. Omit it to review the current workspace. The runtime binds the child workspace and loads that root's repository guidance; the prompt names the diff/range/files to inspect in ordinary prose.

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

Live child runners reserve against two atomic ceilings: `max_concurrent_children` limits one root session (including its nested descendants and continuations) and also limits one Batch width; `max_global_concurrent_children` limits the whole Hermes process. Defaults are 5 per root session and 20 process-wide. A reservation that would exceed either ceiling rejects the whole Batch before any child starts.

If the background pool is full, Hermes returns a structured `rejected` result and runs no child synchronously. If the endpoint cannot deliver later messages, prepared work runs synchronously with an explicit note rather than silently changing semantics.

## Foreground, background, and timeouts

Scheduling uses only `run_in_background`:

- top-level omitted → the selected profile default (`Reviewer` foreground; others background);
- top-level `False` → foreground wait;
- nested omitted or `False` → foreground;
- nested `True` → fail closed before child execution.

Foreground waiting and child execution have separate operator-controlled limits. Default wait/run values are Explore `900/1800` seconds, Plan `1800/3600`, Reviewer `1800/3600`, and general-purpose `1800/7200`. Reviewer still uses the operator-configured `delegation.max_iterations`; it has no hidden lower turn cap. The profile run limit applies to every child, including work dispatched directly to background. When a foreground wait limit expires, Hermes backgrounds the same future and later emits exactly one completion; it does not queue or restart the child.

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

`general-purpose` and `Reviewer` load real repository rules (`.hermes.md`, `AGENTS.md`, `CLAUDE.md`, or `.cursorrules` under the normal discovery contract) and a workspace/git snapshot. `Explore` and `Plan` skip automatic project context. Every profile skips complete personal-governance injection; Reviewer also receives no parent transcript, parent tool output, Notion Project Memory, session history, or personal memory.

## Retained sessions and `delegate_continue`

Lifecycle is fixed by profile:

- `Explore`, `Plan`, and `Reviewer` are one-shot and never retained.
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

`Explore`, `Plan`, and `Reviewer` never delegate. `delegate_continue` and `clarify` remain unavailable to children. The default `max_spawn_depth=2` permits one bounded `general-purpose` orchestrator layer (`parent → child → grandchild`) under the same gates; depth-2 children are leaves.

## Interrupts and durability

`/agents` (alias `/tasks`) shows active and recent subagents. `/stop` and shutdown propagate interruption to foreground/background children and continuations. Background delegation and retained sessions are process-local, not durable jobs: use cron or a managed background process for work that must survive `/new`, process exit, or Gateway restart.

## Configuration

Concurrency, depth, kill switch, per-profile model/provider and wait/run timeouts, and retained-store TTL/count/byte budgets live under `delegation` in `~/.hermes/config.yaml`. They are operator controls, not model-facing fields. See [Configuration → Delegation](/user-guide/configuration#delegation).
