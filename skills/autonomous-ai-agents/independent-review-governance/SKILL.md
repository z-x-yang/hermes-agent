---
name: independent-review-governance
description: "Use before any independent Reviewer, Codex, Claude, human, or domain review is launched across code, research, documents, or artifacts. This is the sole owner of review authorization, global per-change pass accounting, independence boundaries, follow-up gates, and stop conditions; domain skills may execute only an authorized pass."
version: 1.0.0
author: Evelyn
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [independent-review, governance, delegation, verification, convergence]
    related_skills: [code-review-execution, receiving-code-review, verification-before-completion]
---

# Independent Review Governance

## Overview

This is the canonical control plane for independent review across domains. It decides **whether a review pass is authorized, what independence boundary is required, how many substantive passes the change has consumed, and when review must stop**.

Loading this skill does not authorize a review. If the gate says no independent pass is needed, return to the domain owner and close with its normal evidence oracle.

Domain execution stays elsewhere: `code-review-execution` handles an authorized software pass; paper, scientific, visual, release, and other artifacts stay with their domain owners.

## When to Use

Load this skill before any actor launches a fresh Reviewer, review-oriented Codex/Claude session, human review request, or domain-specific independent review.

It applies to:

- the first proposed substantive pass;
- any situation where a prior pass may already exist;
- resumed or compacted work whose pass count is uncertain;
- multiple agents, sessions, or domain owners that could invoke review;
- requests relabeled `final`, `closure`, `clearance`, `signoff`, or similar.

Do not load it for ordinary self-checks, deterministic tests, lint/build/render inspection, or processing feedback that has already been supplied. Those are verification or `receiving-code-review`, not a new independent pass.

## Ownership Boundary

| Question | Owner |
|---|---|
| Is independent review warranted? | `independent-review-governance` |
| How many substantive passes has this change used? | `independent-review-governance` |
| Is a follow-up or exceptional pass authorized? | `independent-review-governance` |
| What independence boundary is required? | `independent-review-governance` |
| How is an authorized software review packaged and executed? | `code-review-execution` |
| What proves the artifact is complete? | Domain owner + `verification-before-completion` |
| How are already-received findings processed? | Controller/domain owner; use `receiving-code-review` when applicable |

No domain execution skill may create a second pass budget or reset the global count.

## Governance Workflow

### 1. Identify the reviewed change

Freeze the durable identity of the change or artifact before counting or authorizing review:

- canonical task/ledger;
- branch, commit/range, artifact digest, or equivalent source state;
- approved contract and domain boundary;
- actors or sessions that may already have launched review.

A rebuilt artifact, repair commit, new session, new reviewer tool, or new label is still the same change unless the user/domain owner explicitly defines a new review scope.

### 2. Recover the global pass count

Before every launch, recover successful review-oriented calls from the task ledger/status and, when necessary, session/tool-call history. Count by purpose, not tool or command name.

A substantive pass requires a usable reviewer verdict or evidence-backed findings tied to the frozen package. Auth, quota, setup, transport, or empty-verdict failures do not consume a substantive pass, but record them and allow at most one corrected retry for that pass before reporting the route blocked.

If the pass count cannot be recovered, fail closed: do not assume zero and do not launch another reviewer.

Recommended durable record:

```text
review_calls:
  - pass: <1 | 2 | exceptional-N>
    domain: <software | paper | figure | release | other>
    route: <Reviewer | Codex | Claude | human | domain reviewer>
    purpose: <whole-change | targeted-closure | exceptional>
    source: <commit/range/artifact digest>
    result: <usable verdict | failed launch>
review_passes_used: N
review_next_pass_gate: <pass-1 | material-risk-reshape | explicit-exception | none>
```

### 3. Decide whether a pass is warranted

Use behavioral risk, sharedness, trust/privacy authority, concurrency, low reversibility, public-contract complexity, weak controller evidence, and explicit user/domain-owner requirements. Raw line count, generated output, mechanical renames, subagent authorship, or the word `final` are signals—not automatic gates.

If deterministic domain evidence can safely close the change, do not manufacture review for ceremony.

### 4. Authorize proportionally

Default budget: **one substantive independent pass on a stable, locally verified change**.

A targeted pass 2 is authorized only when confirmed pass-1 blockers materially reshape architecture, trust/privacy boundaries, concurrency or locking, durable state/crash recovery, irreversible side-effect ordering, or a public compatibility contract, and controller/domain-owner evidence alone cannot safely close the resulting risk.

Any later substantive pass requires explicit exceptional authorization from Zongxin or the domain owner and must name the genuinely new evidence, expertise, or independence boundary it will add.

The budget does not reset after:

- repair, rebuild, or recommit;
- compaction, restart, or session change;
- changing Reviewer/Codex/Claude/human route;
- changing model, provider, harness, or command;
- renaming a pass `targeted`, `closure`, `final`, or `signoff`;
- another agent taking ownership.

### 5. Choose the independence boundary

Classify what the risk claim actually needs:

- **Fresh implementation context:** a new one-shot reviewer sees the frozen contract, package, and evidence but not the implementer's conversation. This is the ordinary default.
- **Cross-harness independence:** a different tool loop or harness is materially useful.
- **Model/provider independence:** live configuration proves a distinct approved model/provider is required.
- **Human/domain independence:** qualified human or domain expertise is required.

Do not infer model/provider independence from a tool, executable, router, or agent name. The built-in one-shot `Reviewer` is normally sufficient for fresh-context software review; stronger boundaries require a real risk reason or explicit request.

### 6. Hand off one authorized pass

The governance handoff must state:

- pass number and purpose;
- exact frozen change/artifact identity;
- approved contract and trust/domain boundary;
- allowed reviewer route and independence claim;
- whether the pass is whole-change or targeted closure;
- the domain execution owner that now takes over.

For software, hand off to `code-review-execution`. Do not include software diff/test mechanics here.

## Follow-up Continuity

When pass 2 is authorized, use a fresh reviewer with a minimal closure packet—not retained reviewer chat and not another blind broad sweep:

```text
APPROVED CONTRACT / DOMAIN BOUNDARY
PASS 1 CONFIRMED FINDINGS AND DISPOSITIONS
EXACT REPAIR SCOPE
FRESH DETERMINISTIC EVIDENCE
TARGETED CLOSURE QUESTION
```

The reviewer role remains one-shot. `delegate_continue` is not review continuity; it carries stale source, tool output, and commitment bias.

## Findings and Stop Conditions

Reviewer findings are candidate leads, not truth. The controller or domain owner verifies each against the approved contract and real execution path, then classifies it as confirmed blocker, false positive, later scope, or user-owned decision.

After the authorized pass:

- group confirmed findings into one bounded repair;
- close ordinary repairs with deterministic evidence;
- return here only if a new pass is being considered;
- if the next-pass gate is not met, stop reviewing;
- if important uncertainty remains but no pass is authorized, mark the task blocked rather than opening a reviewer–fixer loop.

When policy calibration or external-practice comparison is explicitly needed, read `references/community-calibration.md`.

## Common Pitfalls

- Letting a domain execution skill authorize its own follow-up review.
- Counting per agent, session, commit, phase, or command instead of per change.
- Treating a failed launch as a completed pass—or retrying the same broken route indefinitely.
- Calling fresh context “independent model” without live evidence.
- Forwarding reviewer prose without controller/domain verification.
- Rebuilding an artifact and relabeling the next review as a fresh budget.
- Loading review governance for routine tests or self-review.

## Reporting

Report only the decision-relevant governance state:

- change/artifact identity;
- substantive passes used and failed launches;
- authorized route and independence boundary;
- next-pass gate or explicit stop;
- any unresolved user/domain-owner decision.
