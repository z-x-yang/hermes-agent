---
sidebar_position: 13
title: "Delegation & Parallel Work"
description: "Practical patterns for Explore, Plan, Reviewer, general-purpose, Batch, and retained follow-ups"
---

# Delegation & Parallel Work

Hermes delegates isolated work through `delegate_task(description=..., prompt=...)`. Choose the narrowest built-in `subagent_type`, make `prompt` self-contained, and use `run_in_background=False` only when the parent immediately depends on the result. For the complete contract, see [Subagent Delegation](/user-guide/features/delegation).

## Choose the narrowest type

| Need | Type | Lifecycle |
|---|---|---|
| Locate code, trace a call path, gather file/line evidence | `Explore` | read-only, one-shot |
| Research a change and identify critical implementation files | `Plan` | read-only, one-shot |
| Independently inspect a scoped code change and return candidate blockers | `Reviewer` | repo context, ordinary prompt/final, one-shot |
| Edit, test, use terminal/process, or perform permitted external actions | `general-purpose` | automatically retained after success |

`general-purpose` receives only the exact current-parent tool authority that survives runtime policy checks. It is not an unrestricted worker and not a no-side-effect sandbox. On the same provider/endpoint it inherits the parent's SOUL/MEMORY/USER context; cross-provider/endpoint routes and fallbacks strip that personal context. Explore/Plan remain lean and skip automatic project context; Reviewer remains personal-context isolated. Reviewer and general-purpose load repository project context and a workspace/git snapshot. Reviewer uses ordinary read/search/terminal plus authority-gated readonly web, but no named private-source, write, process, browser, or delegation tools. Because it has raw terminal, its no-edit/private-source rule is an instruction plus controller verification, not a mechanical sandbox.

## Pattern: focused exploration before acting

```python
delegate_task(
    description="trace token refresh",
    prompt="""Repository: /home/user/webapp.
Start at src/auth/middleware.py. Return absolute paths, symbols, line ranges,
and unresolved call edges. Do not modify anything.""",
    subagent_type="Explore",
    run_in_background=False,
)
```

## Pattern: planning research

```python
delegate_task(
    description="plan token rotation",
    prompt="""Repository: /home/user/webapp.
Return research inputs for the parent: critical files, existing tests, migration
risks, security constraints, and open questions. Do not choose the final plan.""",
    subagent_type="Plan",
    run_in_background=False,
)
```

## Pattern: one implementation worker

```python
delegate_task(
    description="fix token reuse",
    prompt="""Repository: /home/user/webapp.
Fix refresh-token reuse detection, add regression tests, and run
pytest tests/auth/test_tokens.py -q. Return changed files and real output.""",
    subagent_type="general-purpose",
)
```

Top-level omission defaults `run_in_background` to true except for Reviewer, whose default is foreground. Do not poll a background worker; one completion re-enters the owning conversation.

## Pattern: parallel independent work

Hermes Batch is an intentional Gateway UX extension. Use it only for independent tasks:

```python
delegate_task(
    tasks=[
        {
            "description": "inspect signing",
            "prompt": "Map token creation/signing and return path:line evidence.",
            "subagent_type": "Explore",
        },
        {
            "description": "inspect invalidation",
            "prompt": "Map session invalidation and return path:line evidence.",
            "subagent_type": "Explore",
        },
    ]
)
```

One Batch uses one handle, one async slot, and one consolidated completion. Every item contains only `description`, `prompt`, and optional `subagent_type`; the group shares top-level `run_in_background`.

## Pattern: continue retained GP work

A successful general-purpose result may return `agent_id`:

```python
delegate_continue(
    agent_id="<returned agent_id>",
    prompt="Add the missing concurrency regression test and rerun the suite.",
    run_in_background=False,
)
```

Explore and Plan are one-shot. GP retention is automatic, process-local, bounded, and restart-ephemeral. Continue only the same scoped work; use a fresh `delegate_task` for an unrelated objective.

## Pattern: runtime-derived nesting

Do not request a role. A general-purpose child receives `delegate_task` only when the parent really has that exact authority, the kill switch is enabled, and the configured depth permits another layer. Nested omission is foreground; nested `run_in_background=True` fails before child execution.

## What not to do

- Do not pass removed fields such as `goal`, `context`, per-item scheduling controls, or explicit retention controls.
- Do not ask Explore or Plan to edit or run shell commands.
- Do not put dependent tasks in one Batch.
- Do not assume a subagent self-report proves tests, file changes, or external side effects—verify from the parent.
- Do not use delegation for work that must survive `/new`, shutdown, or Gateway restart; use cron or a managed process.
