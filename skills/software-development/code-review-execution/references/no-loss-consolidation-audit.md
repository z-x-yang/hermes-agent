# No-loss consolidation audit

Use this when a delegated implementation consolidates multiple live, vendored, cached, generated, or host-specific implementations into one canonical repository/package. The acceptance question is not only “are the new adapters mutually consistent?” but also “did the consolidation preserve every required old behavior and operating instruction?”

## Why ordinary parity is insufficient

Cross-host parity can prove all new adapters agree while all of them omit the same legacy capability. Fake host CLIs can also make a suite green against an invented common envelope even when real hosts expose incompatible JSON shapes. Treat new↔new parity as one gate, never as old→new no-loss proof.

## Freeze authoritative inputs first

Before implementation or acceptance, snapshot and hash every source that may contain distinct semantics:

- current live implementation, including uncommitted but active files;
- repository source/tag/commit;
- installed targets and selected runtime caches;
- each host/machine variant;
- schemas/manifests, adapters, skills/references, tests, and installer/registry state.

Record inventories and SHA-256 hashes outside the source tree. Do not infer current external state from an old snapshot when the live source is still accessible.

## Four required matrices

### 1. Runtime/core matrix

For every engine/core file, record old source, new destination, hash, and classification. Byte-identical hashes are strong evidence for preserved core semantics. A changed core requires behavior-level disposition for every changed command/branch; “refactored” is not a classification.

### 2. Tool-contract matrix

Normalize each old and new host surface into:

```text
tool | fields | required | types | enum/const/default | cross-field invariants | root precedence | error envelope
```

Compare sets mechanically. Tightening a boundary (for example generic boolean → true-only `const:true`) may be `INTENTIONAL`, but it needs an approved rationale and negative behavioral test. Unknown/missing fields or unexplained defaults are blockers.

### 3. Behavior/test matrix

Inventory every old test name and assertion family, then map it to a new behavioral test and runtime path. Include negative/no-mutation behavior, old-format recovery/migration, completion gates, typed links, root discovery, repair limits, and lifecycle hints. A larger new test count is not proof: every old assertion needs a destination or explicit retirement rationale.

### 4. Documentation/operations matrix

Map every old skill/reference/maintainer instruction. Moved filenames need content/semantic evidence, not basename presence. Separate:

- shared operating contract that all hosts need;
- host-specific lifecycle/cron/cache/runtime guidance that must remain available but must not leak into other hosts’ runtime prompts;
- truly superseded instructions, with the superseding source and rationale.

Host-specific material should move to a clearly scoped reference, not disappear and not be injected into every adapter. A design non-goal such as “do not push Hermes cron/Notion policy into Codex or Claude” is a **routing/isolation boundary, not deletion authority**: preserve the complete useful host-only source under a marked host path unless the user explicitly approves retirement. If stale details exist inside it, update only those details and record the superseding mechanism; do not classify the whole document `INTENTIONAL` merely because other hosts must not load it.

Keep short, host-neutral **call-safety constraints** self-contained in the canonical tool/field descriptions when an agent may see the schema without the skill: exact `complete_items` semantics, active-only state limits, evidence requirements, dependent flags such as `allow_open_items + why`, typed locator shape, and old-format no-loop behavior. Move platform lifecycle policy out of descriptions, but do not use “host-neutral” as a reason to erase misuse-prevention semantics. Pin the minimum description obligations across every adapter with contract tests.

## Real-host envelope gate

Before accepting installer/readback fixtures, query each accessible real host CLI and save the minimal target entry shape. Build host-specific fixtures from those shapes. Do not force Hermes, Codex, Claude, or other hosts into a convenient shared schema when their actual fields differ.

For each host, independently verify:

- installed/present state;
- enabled/status semantics;
- version semantics (including build metadata where applicable);
- source vs selected-cache path semantics;
- content/root/engine/contract hashes;
- exactly what registry/config/cache paths registration mutates and rollback restores.

Do not infer that a `local` source or a registry entry pointing at the source tree means “no cache.” Diff the entire temporary HOME before/after a real registration: a host may still materialize a versioned selected cache. Back up and restore that plugin-specific cache subtree transactionally, and read back its version/root/engine/contract hashes independently from the source target. Fake CLIs must reproduce this filesystem side effect and failure-after-cache-creation path, not only the JSON envelope.

Also audit the **published tree boundary**. Runtime task state, review ledgers (`work/**/STATUS.md`), local evidence, temporary install state, and reviewer artifacts must never be copied merely because they live under the repository root. Test the exact staged/installed file inventory or use an explicit packaging allowlist; a source-tree copy with only `.git`/cache exclusions is not a release boundary.

A CLI exit code or directory existence is not readback proof.

When a shared manifest relies on a host-expanded placeholder (for example `${CLAUDE_PLUGIN_ROOT}`), prove support in the **exact consuming subsystem**. The token appearing in a native binary, documentation, or a hook implementation does not prove that the MCP manifest loader expands it. Use an isolated temporary HOME with the real host installer/CLI, then read back the parsed MCP command/cwd or start the MCP and list its tools from a foreign cwd. Keep live user state untouched. If only rollout can exercise that loader, retain an explicit fail-closed rollout blocker; do not downgrade it to a non-blocking residual.

Fixtures must preserve each host’s real envelope rather than normalize before parsing: for example, top-level array versus `{installed: [...]}`, `status:"enabled"` versus boolean `enabled`, and logical `source` labels versus selected-cache filesystem paths are distinct contracts. Capture the minimal real entry first, then make fake-host tests reproduce that exact shape.

## Classification and stop rule

Every row must be one of:

- `EXACT` — preserved with direct evidence;
- `INTENTIONAL` — approved host/compatibility difference with rationale and test;
- `MISSING` — no destination/evidence;
- `DIVERGENT` — behavior differs without an approved contract;
- `SUPERSEDED` — deliberately retired, with authoritative replacement and compatibility rationale.

Any `MISSING`, unexplained `DIVERGENT`, or vague “covered elsewhere” blocks commit/release/rollout. Do not retire the old live source until the final canonical hash is deployed, fresh host sessions pass temp-root workflows, and rollback evidence remains available.

## Controller acceptance sequence

1. Require the implementation owner to produce a no-loss matrix and executable gate.
2. Independently regenerate the tool/core/test/reference inventories; do not sign the owner’s matrix by inspection alone.
3. Run deterministic old→new differential tests in isolated roots.
4. Query real host envelopes and mutate only temporary HOME/registry/cache fixtures during installer tests.
5. Reconcile every discrepancy against the approved design.
6. Only then spend the independent reviewer pass; ask it explicitly to audit no-loss traceability and common-mode omissions.
7. After review fixes, rerun the full denominator on one committed hash before rollout.
