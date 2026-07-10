---
sidebar_position: 7
title: "Subagent Delegation"
description: "Built-in subagent types, scheduling, continuation, and capability boundaries"
---

# Subagent Delegation

Hermes uses `delegate_task` to run isolated child agents. A child starts with a separate conversation and terminal state, receives only the task and context supplied by the parent, and returns a structured result. The caller cannot select arbitrary child tools: Hermes applies the parent's available tools and then enforces the selected subagent type's capability ceiling.

## Built-in subagent types

`subagent_type` accepts exactly three built-ins:

| Type | Intended work | Capability ceiling | `scheduling="auto"` for one task |
|---|---|---|---|
| `Explore` | Search and understand code, files, and supporting sources | Read-only: `read_file`, `search_files`, `web_search`, and `web_extract` | Foreground |
| `Plan` | Research a codebase and prepare inputs for a later implementation plan | The same read-only ceiling as `Explore`; it cannot edit or claim implementation is complete | Foreground |
| `general-purpose` | Multi-step repository work, including edits and tests | A closed allowlist for files, raw shell/process work, task tracking, skills, and vision; named messaging/Notion/cron/memory/delegation tools are excluded, but shell access is not a no-side-effect sandbox | Background |

`Explore` and `Plan` cannot write files, run shell commands, create external side effects, or delegate. `general-purpose` may edit and test and cannot directly call the excluded named side-effect tools or delegate by default. However, it deliberately retains raw `terminal` and `process`: shell commands can access networks, invoke authenticated CLIs, and cause external effects. Those actions remain governed by normal terminal approvals and the task instructions; Hermes does not provide hard no-side-effect isolation for this profile.

Omitting `subagent_type` preserves legacy generic delegation. This has two deliberately different compatibility surfaces:

- A **model-originated** call with legacy `scheduling="auto"` runs in the background.
- A plain **direct Python** call remains synchronous when it supplies no subagent type and makes no explicit scheduling/background request.

Do not rely on the direct-Python compatibility rule to predict model-facing scheduling.

## Single tasks and batches

A focused read-only investigation:

```python
delegate_task(
    goal="Locate the authentication retry logic and explain its call path",
    context="Repository root: /home/user/webapp. Include file:line evidence.",
    subagent_type="Explore",
)
```

A repository-local implementation task:

```python
delegate_task(
    goal="Fix the failing authentication retry tests",
    context="Repository root: /home/user/webapp. Run pytest tests/auth/.",
    subagent_type="general-purpose",
)
```

A parallel batch:

```python
delegate_task(tasks=[
    {
        "goal": "Map the token refresh path",
        "context": "Repository root: /home/user/webapp.",
        "subagent_type": "Explore",
    },
    {
        "goal": "Map session invalidation",
        "context": "Repository root: /home/user/webapp.",
        "subagent_type": "Explore",
    },
])
```

The model-facing schema does not expose `toolsets`, model/provider selection, iteration budgets, or timeout controls. Those are operator-controlled policy and configuration. A stale client may still send removed fields, but it cannot use them to widen a child profile.

## Scheduling

`scheduling` accepts `auto`, `foreground`, or `background`.

- **`auto`**: one `Explore` or `Plan` task runs in the foreground; `general-purpose`, legacy generic calls, and multi-task batches run in the background.
- **`foreground`**: Hermes waits up to the resolved foreground wait timeout.
- **`background`**: Hermes immediately returns a handle and later injects the completed result into the owning conversation.
- **Nested/orchestrator work**: runs synchronously in the foreground. An explicit nested background request fails closed.

Foreground waiting and child execution have separate limits:

1. `foreground_wait_timeout_seconds` controls how long the parent waits.
2. `child_run_timeout_seconds` caps a child that **started in foreground**, using its type-specific or global configuration.

If the foreground wait expires first, Hermes hands the **same running future** to background delivery. It does not restart the child. The caller receives `backgrounded_after_foreground_timeout`, followed by exactly one later completion when that future finishes.

Pure background jobs keep the existing behavior: profile `child_run_timeout_seconds` is not applied as a blanket timeout to work that started in the background. The older opt-in `delegation.child_timeout_seconds` hard cap, if configured, still applies independently.

### Batch delivery is consolidated

One batch is one asynchronous unit:

- one returned `delegation_id` handle;
- one occupied background slot;
- one consolidated completion after all children finish;
- results remain ordered by task index.

Hermes never returns or injects a separate handle/completion for each task in the batch.

If the endpoint cannot deliver later messages (for example, a stateless HTTP request) or the background pool is at capacity, Hermes runs the already-prepared work synchronously and includes a note in the result rather than returning a handle that can never resolve.

## Context isolation

A new child does not inherit the parent's conversation transcript. Put all necessary details in `goal` and `context`: repository root, relevant files, errors, constraints, verification commands, and desired output language.

```python
# Too vague
 delegate_task(goal="Fix the error", subagent_type="general-purpose")

# Self-contained
 delegate_task(
    goal="Fix the TypeError in api/handlers.py",
    context="""Repository: /home/user/webapp.
process_request() receives None from parse_body() when Content-Type is missing.
Add a regression test and run pytest tests/api/.""",
    subagent_type="general-purpose",
)
```

Subagent summaries are self-reports. Verify important file changes, tests, and external claims from the parent before presenting them as facts.

## Retained sessions and `delegate_continue`

`delegate_task` can retain a completed child transcript for a short follow-up:

- `general-purpose` is retained by default **only after successful completion** and only when the parent has a nonempty session ID and retention capacity is available.
- `Explore` and `Plan` are one-shot by default. Set `retain_session=true` to retain a completed run explicitly.
- Set `retain_session=false` to disable retention for a call.
- Stateless/empty-session requests do not receive resumable `agent_id` values.

A retained result includes an `agent_id`. Continue it with:

```python
delegate_continue(
    agent_id="<agent_id from the completed result>",
    prompt="Now add the missing regression test and rerun the focused suite.",
    scheduling="auto",
)
```

`delegate_continue` accepts only `agent_id`, `prompt`, and `scheduling`. It preserves the original subagent type, role, workspace hint, model/provider metadata, and capability ceiling. It cannot change tools, type, role, retention policy, or timeouts.

Retention safety and lifetime:

- The store is in-process, TTL-bounded, record-count-bounded, and serialized-transcript-byte-bounded (`3600` seconds, `64` records, and `16777216` bytes by default).
- Initial records larger than the byte budget are not retained. Aggregate pruning removes only non-in-flight records; claimed continuations are never evicted.
- If a successful continuation grows beyond the byte budget, Hermes returns that successful result with `retention_dropped`, invalidates the retained handle, and rejects future continuation attempts.
- Only the same nonempty parent session may continue an `agent_id`.
- Only one continuation for a given `agent_id` may be in flight; a concurrent second call fails immediately. Different retained agents may continue concurrently.
- `/stop` and shutdown interrupt background continuations.
- Gateway/process restart loses all retained sessions; this is not durable persistence.
- Credentials and custom `base_url` values are not retained. Credentials are resolved again from current trusted configuration, so exact custom-endpoint fidelity after configuration changes is not guaranteed.

## Nested orchestration

Legacy generic children may use `role="orchestrator"` when nested delegation is enabled. Built-in `Explore`, `Plan`, and `general-purpose` profiles do not permit the orchestrator role.

- `role="leaf"` is the default and cannot delegate.
- `role="orchestrator"` keeps delegation only when `delegation.orchestrator_enabled` is true and the configured `max_spawn_depth` permits another level.
- `max_spawn_depth` defaults to `1` (flat delegation), has a floor of `1`, and has no hard upper ceiling. Each extra level can multiply cost and concurrency.

Nested work stays synchronous/foreground so a child cannot detach work from the parent that owns it.

## Interrupts, monitoring, and durability

`/agents` (alias `/tasks`) shows active and recently completed subagents in the TUI. `/stop` and shutdown propagate interruption to foreground and background children, including background continuations.

Background delegation is asynchronous, but not durable job storage. Closing the owning session with `/new`, stopping the process, or restarting the gateway can discard running work and retained transcripts. Use cron or a separately managed background process for work that must survive agent/gateway lifecycle changes.

## Configuration

Scheduling limits, per-agent model/provider overrides, retention TTL/count/byte capacity, concurrency, and nesting are configured under `delegation` in `~/.hermes/config.yaml`. These operator controls are intentionally absent from the model-facing tool schema. See [Configuration → Delegation](/user-guide/configuration#delegation).
