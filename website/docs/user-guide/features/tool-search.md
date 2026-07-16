---
title: Tool Search
sidebar_position: 95
---

# Tool Search

When you have many MCP servers or non-core plugin tools attached to a
session, their JSON schemas can consume a substantial fraction of the
context window on every turn — even when only a few of them are relevant
to what the user actually asked for.

**Tool Search** is Hermes' progressive-disclosure layer for that
problem. When activated, MCP and plugin tools are replaced in the
model-visible tools array by three bridge tools, and the model loads each
specific tool's schema on demand.

:::info Built-in Hermes tools never defer
The tools that make up Hermes' core capability set (`terminal`,
`read_file`, `write_file`, `patch`, `search_files`, `todo`, `memory`,
`browser_*`, `web_search`, `web_extract`, `clarify`, `execute_code`,
`delegate_task`, `session_search`, and the rest of
`_HERMES_CORE_TOOLS`) are *always* loaded directly. Only MCP tools and
non-core plugin tools are eligible for deferral.
:::

## How it works

When Tool Search activates for a turn, the model sees three new tools in
place of the deferred ones:

```
tool_search(query, limit?)     — search the deferred-tool catalog
tool_describe(name | names[])  — load one or several full schemas
tool_call(name, arguments)     — invoke a deferred tool
```

The system prompt contains a compact, names-only index of every deferred tool
available to that session. If an exact name matches the requested operation, the
model can start with `tool_describe(name)` without running a keyword search
first. It can also pass an array of names to load several schemas in one round
trip. Keyword search remains the fallback when the correct name is not obvious.
Because the visible set is partial, a broader visible integration is not a
substitute for an exact operation that may be deferred. When a request names an
app or service and no already-visible dedicated tool matches, Tool Search is a
mandatory discovery gate before browser, computer, or terminal.

A typical interaction looks like:

```
System prompt index: mcp_github_create_issue, mcp_github_get_repo
Model: tool_describe(["mcp_github_create_issue", "mcp_github_get_repo"])
  → { tools: [{ name: "mcp_github_create_issue", parameters: ... }, ...], errors: [] }
Model: tool_call("mcp_github_create_issue", { title: "...", body: "..." })
  → { ok: true, issue_number: 42 }
```

When the model invokes `tool_call`, Hermes **unwraps the bridge** and
dispatches the underlying tool exactly as if the model had called it
directly. Pre-tool-call hooks, guardrails, approval prompts, and
post-tool-call hooks all run against the real tool name — not against
`tool_call`. The activity feed in the CLI and gateway also unwraps so you
see the underlying tool, not the bridge.

## When does it activate?

By default Tool Search runs in `auto` mode. It activates when any positive
threshold is met: 10 deferred tools, 10,000 deferred-schema tokens, or 10%
of the active model's context window. Set an individual threshold to `0` to
disable that gate. Below all enabled gates, the tools-array assembly is a
pure pass-through and you pay no bridge overhead.

This decision is re-evaluated every time the tools array is built, so:

- A session with fewer than 10 small MCP/plugin tools and a long context model
  never activates Tool Search.
- A session with many MCP servers attached starts activating it.
- Exact names or shell-style globs in `always_visible_tools` remain directly
  callable while the eligible long tail is deferred.
- Removing MCP servers mid-session correctly returns to direct exposure
  on the next assembly.

## Configuration

```yaml
tools:
  tool_search:
    enabled: auto
    threshold_schema_tokens: 10000
    threshold_pct: 10
    always_visible_tools:
      - "ledger_*"
      - "mcp_mail_search"
    search_default_limit: 5
    max_search_limit: 20
```

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `auto` | `auto` activates when either positive threshold is met; `on` always activates if there's at least one deferrable tool; `off` disables entirely. |
| `threshold_schema_tokens` | `10000` | Estimated deferred-schema token count at which `auto` activates. Set `0` to disable this gate. |
| `threshold_pct` | `10` | Percentage of active context at which `auto` activates. Set `0` only if another gate should decide. |
| `always_visible_tools` | `[]` | Exact names or shell-style globs kept directly visible and excluded from search/describe/call bridge scope. |
| `search_default_limit` | `5` | Hits returned when the model calls `tool_search` without a `limit`. |
| `max_search_limit` | `20` | Hard upper bound the model can request via `limit`. Range 1–50. |

You can also flip the legacy boolean shape:

```yaml
tools:
  tool_search: true   # equivalent to {enabled: auto}
```

## When NOT to use it

Tool Search trades a fixed per-turn token cost (the three bridge tool schemas
plus the compact names-only index) and extra round trips (describe → call when
the exact name is known; otherwise search → describe → call) for the savings on
the deferred schemas. Batched describe avoids repeated schema-loading turns when
several deferred tools are likely to be needed. It's a clear
win when you have many tools and use few per turn; it's overhead when
you have few tools total.

The `auto` default handles this for you. If you set `enabled: on`
unconditionally, expect a slight per-turn cost on small toolsets.

## Trade-offs that don't go away

These come from the prompt-cache integrity invariant — they are inherent
to any progressive-disclosure design, not specific to this implementation:

- **Extra round trips on cold tools.** The first time the model needs
  a deferred tool, it spends one or two extra model calls to load or find
  the schema. The token savings on the static side are real, but a
  portion is paid back at runtime.
- **No cache benefit on deferred schemas.** A loaded `tool_describe`
  result enters the conversation history (so it does get cached on
  subsequent turns) but it never benefits from the system-prompt cache
  prefix.
- **Model-quality dependence.** Tool Search assumes the model can write a
  reasonable search query for the tool it wants. Smaller models do this
  less well; the published Anthropic numbers (49% → 74% on Opus 4 with
  vs. without tool search) show the upside but also that ~26 points of
  accuracy is still retrieval failure.
- **Toolset edits invalidate cache.** The names-only index is frozen for the
  lifetime of a session so ordinary prompt rebuilds remain byte-stable. When an
  MCP refresh changes the deferred names, or a restart resumes a prompt whose
  saved index no longer matches the current pin/toolset authority, Hermes
  rebuilds and persists the system prompt once. This is the same cache trade-off
  as any real toolset edit.

## Implementation details

- **Retrieval:** BM25 over tokenized tool name + description + parameter
  names. Falls back to a literal substring match on the tool name when
  BM25 returns no positive-score hits, which protects against
  zero-IDF degenerate cases (e.g. searching `"github"` against a
  catalog where every tool name contains "github").
- **Catalog is stateless across turns.** It rebuilds from the current
  tool-defs list every assembly — no session-keyed `Map`. This avoids
  the class of bug where a stored catalog drifts out of sync with the
  live tool registry.
- **The names-only index is session-stable and authority-scoped.** Its source is
  the same deferrable slice accepted by the bridge after enabled/disabled
  toolsets and subagent policy are applied. Prompt compression reuses the frozen
  snapshot; a real MCP catalog change explicitly invalidates it.
- **The catalog is scoped to the session's toolsets.** `tool_search`,
  `tool_describe`, and `tool_call` only ever see and invoke tools the
  session was actually granted. A subagent, kanban worker, or gateway
  session restricted to a subset of toolsets cannot use the bridge to
  discover or call a tool outside that subset — the deferred catalog is
  the deferrable slice of the session's own enabled/disabled toolsets,
  not the whole process registry.
- **Configured pins are direct-only.** `always_visible_tools` entries are
  removed from the deferred catalog and cannot be described or invoked through
  the bridge, avoiding duplicate direct/bridge paths.
- **No JS sandbox.** Hermes uses the simpler "structured tools" mode
  (search / describe / call as plain functions). The JS-sandbox "code
  mode" some other implementations offer is a large surface area; we
  skip it.

## See also

- `tools/tool_search.py` — the implementation
- `tests/tools/test_tool_search.py` — the regression suite
- The `openclaw-tool-search-report` PDF in the original implementation
  PR for the research that shaped the design
