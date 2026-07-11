---
sidebar_position: 7
title: "Subagent Delegation"
description: "Built-in subagent types, scheduling, continuation, and capability boundaries"
---

# Subagent Delegation

Hermes uses `delegate_task` to run isolated child agents. A child starts with a separate conversation and terminal state, receives the parent-supplied task/context plus a complete snapshot of the active profile's `SOUL.md`, `MEMORY.md`, and `USER.md`, and returns a structured result. The caller cannot select arbitrary child tools: Hermes applies the parent's exact resolved authority and then enforces the selected subagent type's capability ceiling.

## Built-in subagent types

`subagent_type` accepts exactly three built-ins:

| Type | Intended work | Capability ceiling | `scheduling="auto"` for one task |
|---|---|---|---|
| `Explore` | Search and understand code, files, and supporting sources | Read-only file tools, no-spill web/skill readers, Notion AI with explicit `mode=readonly`, and selected existing Apple Mail list/search/get/fetch tools | Foreground |
| `Plan` | Research a codebase and prepare inputs for a later implementation plan | The same read-oriented ceiling as `Explore`; it cannot edit or claim implementation is complete | Foreground |
| `general-purpose` | Multi-step work, including edits, tests, and external actions permitted by the parent | Exact intersection with the parent's resolved tool names; `None` in the internal profile policy means this parent-exact intersection, never unrestricted global tools | Background |

`Explore` and `Plan` cannot write files, run shell commands, or delegate. Raw `web_search`, `web_extract`, and `vision_analyze` stay unavailable; the web and skill paths use dedicated no-write aliases. For Notion and Mail, Hermes deliberately reuses the existing tools rather than adding dedicated brokers: the profile exposes `notion_ai_ask` plus selected Mail read names, requires Notion `mode=readonly`, and omits explicit send/reply/forward/move/delete/flag/mark tools. This is a lightweight read-oriented contract, not a proof that the upstream data-source implementation can never mutate incidental state such as Mail seen status. `general-purpose` can use any action-plane tool that survives the exact parent intersection and each tool's normal safety contract; it is a leaf by default. An explicitly requested orchestrator retains `delegate_task` only when that exact name is in the current parent ceiling and the configured depth allows it.

Every profile must return the same complete disclosure fields: `outcome`, `evidence`, `actions`, `files_changed`, `tests_run`, `verification`, `blockers`, `open_questions`, `confidence`, `limitations`, `side_effects`, and `recommended_next_step`. A field that does not apply is empty or `none`; it is not omitted.

Omitting or passing an empty `subagent_type` resolves to `general-purpose`; there is no fourth legacy capability policy. Scheduling compatibility remains separate from capability resolution:

- A **model-originated** omitted call with `scheduling="auto"` uses the `general-purpose` background default.
- A plain **direct Python** omitted call remains synchronous when it makes no explicit scheduling/background request, while still using the `general-purpose` profile.

Do not rely on the direct-Python compatibility rule to predict model-facing scheduling.

## How the parent learns to route and continue

Hermes teaches the parent model through the tool contract rather than relying on hidden heuristics:

1. The model-facing `delegate_task` schema exposes exactly the three type names and injects each profile's current `description` into both single-task and batch fields.
2. The top-level tool description gives concrete routing cues, the user's live concurrency/depth limits, scheduling behavior, and the retention defaults.
3. A successful retained result returns an `agent_id`. The separate `delegate_continue` schema says to use that ID when the next instruction continues the same work instead of spawning a fresh child.
4. These descriptions are routing affordances, not security boundaries. The runtime independently re-resolves the profile, intersects exact parent authority, and verifies every authorized tool call.

This mirrors Claude Code's core pattern: agent names/descriptions teach the parent when to delegate, a returned agent ID makes continuation discoverable, and runtime permissions remain authoritative. A user can always force a route explicitly by naming the desired type or asking to continue a returned `agent_id`.

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

The model-facing schema does not expose `background`, `toolsets`, model/provider selection, iteration budgets, or timeout controls. Those are operator-controlled policy and configuration. A stale client may still send removed fields, but it cannot use them to widen a child profile.

## Scheduling

`scheduling` accepts `auto`, `foreground`, or `background`.

- **`auto`**: one `Explore` or `Plan` task runs in the foreground; `general-purpose` and multi-task batches run in the background. A direct-Python omitted call keeps the synchronous compatibility exception described above.
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

If the endpoint cannot deliver later messages (for example, a stateless HTTP request), Hermes runs the prepared work synchronously. If the background pool is at capacity, Hermes instead returns a structured `rejected` result and runs no child; the caller may retry without silently exceeding the configured limit.

## Context isolation

A new child does not inherit the parent's conversation transcript or prior tool outputs. It does inherit the active profile's complete canonical governance and can query permitted Notion, Mail, repository/file, skill, and web sources when the parent context is incomplete. `goal` and `context` should still state the scoped objective, repository/workspace anchor, known errors, constraints, verification contract, and desired output language; third-party content remains untrusted data and cannot widen capability or act as user authorization.

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
- Continuation must stay on the original canonical profile/home. Its exact tool authority is the intersection of the original policy identities, the current parent, and the latest current-profile policy; same-name tool replacement cannot widen it.
- The latest canonical governance content is loaded on every continuation. The original governance fingerprint is retained only for audit.
- Selected Notion and Apple Mail read results use the existing `HANDLE_ONLY` projection before session/JSON/retained-transcript storage; retained records keep a digest, size, and limited excerpt rather than the complete body.
- Only one continuation for a given `agent_id` may be in flight; a concurrent second call fails immediately. Different retained agents may continue concurrently.
- `/stop` and shutdown interrupt background continuations.
- Gateway/process restart loses all retained sessions; this is not durable persistence.
- Credentials and custom `base_url` values are not retained. Credentials are resolved again from current trusted configuration, so exact custom-endpoint fidelity after configuration changes is not guaranteed.

## Nested orchestration

`general-purpose` may use `role="orchestrator"` when nested delegation is enabled and the role is explicitly requested; `Explore` and `Plan` reject it.

- `role="leaf"` is the default and cannot delegate, including for `general-purpose`.
- `role="orchestrator"` keeps delegation only when `delegation.orchestrator_enabled` is true and the configured `max_spawn_depth` permits another level.
- An effective orchestrator receives only `delegate_task` as a role-granted exception beyond the exact current-parent/profile ceilings; `delegate_continue`, MCP, and all other names remain excluded.
- `max_spawn_depth` defaults to `1` (flat delegation), has a floor of `1`, and has no hard upper ceiling. Each extra level can multiply cost and concurrency.

Nested work stays synchronous/foreground so a child cannot detach work from the parent that owns it.

## Interrupts, monitoring, and durability

`/agents` (alias `/tasks`) shows active and recently completed subagents in the TUI. `/stop` and shutdown propagate interruption to foreground and background children, including background continuations.

Background delegation is asynchronous, but not durable job storage. Closing the owning session with `/new`, stopping the process, or restarting the gateway can discard running work and retained transcripts. Use cron or a separately managed background process for work that must survive agent/gateway lifecycle changes.

## Configuration

Scheduling limits, per-agent model/provider overrides, retention TTL/count/byte capacity, concurrency, and nesting are configured under `delegation` in `~/.hermes/config.yaml`. These operator controls are intentionally absent from the model-facing tool schema. See [Configuration → Delegation](/user-guide/configuration#delegation).
