---
sidebar_position: 13
title: "Delegation & Parallel Work"
description: "Practical patterns for Explore, Plan, general-purpose, batches, and retained follow-ups"
---

# Delegation & Parallel Work

Hermes can hand focused work to isolated child agents. Choose a built-in type for a predictable capability ceiling, pass a self-contained task, and let the scheduler decide whether the parent should wait.

For the full contract, see [Subagent Delegation](/user-guide/features/delegation).

## Choose the narrowest built-in

| Need | Type | Why |
|---|---|---|
| Locate code, trace a call path, gather file:line evidence | `Explore` | Read-only and foreground by default |
| Research a change before writing an implementation plan | `Plan` | Read-only, plan-oriented result contract, foreground by default |
| Edit repository files, run tests, or complete multi-step repo-local work | `general-purpose` | Closed repo-local worker policy, background by default |

Do not ask `Explore` or `Plan` to edit. `general-purpose` excludes named messaging, publishing, scheduling, Notion, cron, and memory-write tools, and the model cannot widen that allowlist with `toolsets`. It still has raw `terminal` and `process`, so shell commands can reach external systems; treat normal terminal approvals and explicit task instructions as the governing boundary, not a hard no-side-effect sandbox.

## Pattern: focused exploration before acting

Use `Explore` when the parent needs evidence without risking changes:

```python
delegate_task(
    goal="Trace how expired access tokens trigger refresh",
    context="""Repository: /home/user/webapp.
Start at src/auth/middleware.py. Return file:symbol:line evidence,
what you searched, and any unresolved call edges.""",
    subagent_type="Explore",
)
```

A single `Explore` task uses foreground scheduling under `auto`. The parent receives the result inline unless the configured wait expires; in that case, the same child continues in the background and returns one later completion.

## Pattern: planning research without implementation

Use `Plan` to gather the inputs for a later plan:

```python
delegate_task(
    goal="Research what must change to add rotating refresh tokens",
    context="""Repository: /home/user/webapp.
Identify critical files, existing tests, migration risks, security constraints,
and open questions. Do not edit files.""",
    subagent_type="Plan",
)
```

`Plan` cannot write or run shell commands. Its output should inform the parent; it is not proof that implementation happened.

## Pattern: one background implementation worker

Use `general-purpose` for scoped repository work that can run independently:

```python
delegate_task(
    goal="Fix refresh-token reuse detection and add regression tests",
    context="""Repository: /home/user/webapp.
Relevant files: src/auth/tokens.py and tests/auth/test_tokens.py.
Run: pytest tests/auth/test_tokens.py -q.
Return changed files and exact test output.""",
    subagent_type="general-purpose",
)
```

Under `auto`, Hermes returns a background handle immediately. Continue other work instead of polling; the completion is injected into the owning conversation later. Background delegation is not durable across `/new`, `/stop`, shutdown, or process restart.

## Pattern: parallel read-only research

Independent read-only questions are a good batch:

```python
delegate_task(tasks=[
    {
        "goal": "Map token creation and signing",
        "context": "Repository: /home/user/webapp. Return file:line evidence.",
        "subagent_type": "Explore",
    },
    {
        "goal": "Map token validation and revocation",
        "context": "Repository: /home/user/webapp. Return file:line evidence.",
        "subagent_type": "Explore",
    },
    {
        "goal": "Map authentication test coverage gaps",
        "context": "Repository: /home/user/webapp. Read only; do not modify tests.",
        "subagent_type": "Explore",
    },
])
```

A multi-task batch runs in the background under `auto`. The entire fan-out returns **one handle** and later produces **one consolidated result** after every child finishes. There are never per-task handles or per-task completion injections.

Use separate batches if the number of tasks exceeds `delegation.max_concurrent_children`; Hermes rejects an oversized batch rather than truncating it.

## Pattern: parallel edits with disjoint ownership

Multiple `general-purpose` children can edit the same working tree, so split work only when file ownership is disjoint:

```python
delegate_task(tasks=[
    {
        "goal": "Update server token responses",
        "context": "Repository: /home/user/webapp. Own only src/api/tokens.py and tests/api/test_tokens.py.",
        "subagent_type": "general-purpose",
    },
    {
        "goal": "Update Python SDK token parsing",
        "context": "Repository: /home/user/webapp. Own only sdk/python/ and its tests.",
        "subagent_type": "general-purpose",
    },
])
```

Avoid parallel children that may edit the same file, run destructive repository commands, or depend on each other's uncommitted output. Let the parent integrate and verify the combined diff.

## Pattern: retain one implementation thread

A successfully completed `general-purpose` child is retained by default when the parent has a nonempty session ID and capacity is available. Use its returned `agent_id` for a tightly related follow-up:

```python
delegate_continue(
    agent_id="<agent_id>",
    prompt="Address the remaining edge case from the failed parametrized test.",
    scheduling="auto",
)
```

Use `retain_session=true` if an `Explore` or `Plan` run must be continued. Retention is process-local, TTL/capacity bounded, and same-parent only. One `agent_id` cannot have two continuations in flight. A restart loses it.

A continuation keeps the original type, role, workspace hint, model/provider metadata, and capability ceiling. It cannot be used to promote `Explore` into an editor, change tools or timeouts, or move work to another parent session.

## Pattern: nested orchestration only when needed

Nested delegation is for legacy generic orchestrators, not the three built-in profiles:

```python
delegate_task(
    goal="Survey three migration approaches and synthesize a recommendation",
    context="Repository: /home/user/webapp.",
    role="orchestrator",
)
```

This requires `delegation.max_spawn_depth >= 2` and `delegation.orchestrator_enabled: true`. Nested work is synchronous/foreground; explicit background nesting fails closed. Each level can multiply cost, so prefer a top-level batch when the subtasks are already known.

## Scheduling checklist

- Single `Explore`/`Plan` + `auto` → foreground.
- Single `general-purpose` + `auto` → background.
- Model-originated legacy generic + `auto` → background.
- Multi-task batch + `auto` → background, one handle/result.
- Nested/orchestrator delegation → synchronous foreground.
- Direct Python legacy call with no type or explicit scheduling/background request → synchronous compatibility path.
- Foreground wait expiry → the same future is handed to background delivery, then one later completion.
- Foreground-started work gets the configured child run cap; pure background work does not inherit that profile cap as a blanket timeout.

## Context and verification checklist

Before delegating, include:

- repository/workspace path;
- exact files, symbols, errors, or search target;
- allowed scope and files the child owns;
- test or validation commands;
- required output language and evidence format.

After completion:

- inspect the actual diff or files;
- rerun important tests from the parent;
- treat summaries as self-reports, not independent proof;
- remember that one batch completion may contain several child results;
- use `delegate_continue` only when preserving the original capability ceiling is appropriate.

## When not to delegate

- One direct tool call: call the tool.
- Mechanical API/tool pipelines without substantial reasoning: use `execute_code`.
- Work requiring user clarification: children cannot use `clarify`.
- External side effects: perform them through an appropriately approved parent tool and verify the result.
- Durable work that must survive gateway lifecycle changes: use cron or a separately managed background process.
