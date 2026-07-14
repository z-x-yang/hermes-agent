---
name: subagent-driven-development
description: "Use when the user explicitly requests subagent-driven implementation, or an approved software plan has multiple independently contractible context-heavy tasks that cannot safely fit one controller session."
version: 2.0.0
author: Hermes Agent (adapted from obra/superpowers)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [delegation, subagent, implementation, workflow, review-governance]
    related_skills: [plan, requesting-code-review, test-driven-development]
---

# Subagent-Driven Development

## Overview

Execute an approved implementation plan with a fresh implementer subagent per independently contractible task. The parent/controller verifies every task's diff and test evidence; one fresh independent reviewer evaluates the integrated change at the end when the risk gate requires it.

**Core principle:** Fresh implementers + controller-owned task checks + one final independent review = quality without reviewer thrash.

## When to Use

Use this workflow only when:

- the user explicitly requests subagent-driven implementation; or
- an approved software plan has multiple independent, context-heavy tasks that would materially degrade one controller context, and fresh isolation has a concrete correctness or recovery benefit.

Do not use it merely because a plan exists, several files will change, or subagents are available. Small and medium changes should remain in the main implementation session.

## Review Ownership

The parent/controller owns all independent review for the change. Every implementer brief must state this explicitly.

- Implementers perform tests and self-review only.
- Implementers must not invoke Codex, Claude Code, reviewer subagents, or other independent reviewers on their own work.
- If an implementer sees a risk that may need early independent judgment, it reports the concrete risk to the controller instead of launching a review.
- Per-task independent reviewers are exceptional: use one only when the user/plan explicitly requires it or a concrete high-risk blocker needs isolated judgment.
- The ordinary path has one final independent review after all task diffs have landed and the controller has verified them.

This is prompt/system-contract governance, matching ordinary subagent runtimes: identity and responsibility are explicit, while tool exposure and depth continue to enforce the nesting boundary. Do not add a second review-budget state machine merely to restate this workflow.

## Process

### 1. Parse the approved plan once

Extract every task, dependency, global constraint, and completion oracle. Create todos before the first dispatch. Give each implementer a self-contained task brief rather than the whole plan or parent conversation.

### 2. Dispatch one implementer at a time

Use the current public API and the `general-purpose` profile:

```python
delegate_task(
    description="Implement Task N",
    subagent_type="general-purpose",
    run_in_background=False,
    prompt="""
    Implement Task N from the approved plan.

    REQUIREMENTS:
    [SELF-CONTAINED TASK BRIEF]

    INDEPENDENT REVIEW OWNERSHIP:
    The parent/controller owns all independent review for this change. Do not
    invoke Codex, Claude Code, reviewer subagents, or any other independent
    reviewer on your own implementation. Perform tests and self-review only.
    If independent judgment seems necessary, report the concrete risk to the
    controller instead of launching a reviewer.

    EXECUTION:
    1. Follow TDD when deterministic behavior changes.
    2. Implement only this task.
    3. Run the focused test and required regression checks.
    4. Self-review the full task diff.
    5. Make a scoped commit and report status, tests, files, and concerns.

    If required context is missing, return NEEDS_CONTEXT rather than guessing.
    """,
)
```

Do not dispatch overlapping implementers against the same files/worktree. A child may decompose genuine independent implementation subtasks only when the runtime exposes delegation and the brief permits it; it must not pass through its entire assignment or spawn reviewers.

### 3. Controller verifies each task

After the implementer returns, the controller personally:

1. checks the exact commit and changed-file scope;
2. reads the diff and any concerns;
3. runs the task's focused tests and high-signal regression checks;
4. verifies the production call path, not only helper tests;
5. sends one bounded fix brief for confirmed blockers;
6. marks the task complete only after deterministic evidence passes.

This is controller verification, not an independent reviewer call. Do not run separate spec and quality reviewer subagents after every task.

### 4. Run one final independent review when required

After all tasks are complete and controller-verified, use `requesting-code-review` for one fresh-context whole-change review. Supply the approved contract, scoped integrated diff, and fresh test/build evidence.

```python
delegate_task(
    description="Review integrated change",
    subagent_type="Reviewer",
    review_root="/absolute/path/to/local/worktree",  # optional
    run_in_background=False,
    prompt="""
    You are the assigned independent reviewer for the integrated software change.
    Do not edit, commit, or launch another reviewer.

    APPROVED CONTRACT:
    [INSERT APPROVED CONTRACT]

    ACCEPTANCE CRITERIA / INVARIANTS:
    [INSERT CRITERIA]

    SCOPED INTEGRATED DIFF OR REVIEW PACKAGE:
    [INSERT EXACT RANGE OR PACKAGE PATH]

    FRESH TEST / BUILD / RUNTIME EVIDENCE:
    [INSERT COMMANDS AND RESULTS]

    Review those supplied inputs and return only evidence-backed candidate
    blockers for controller adjudication.
    """,
)
```

For high-stakes/shared-core/auth/concurrency work, prefer Codex as the final reviewer. Reviewer findings are leads; the controller reproduces and classifies them before repair. A repair does not automatically authorize another review pass.

### 5. Close the branch

Run the final deterministic suite, lint/type/build checks that match the changed surfaces, `git diff --check`, and clean-status verification. Stage and commit only task files, then follow the branch-closeout workflow.

## Implementer Status

- **DONE:** verify the claimed diff/tests and continue.
- **DONE_WITH_CONCERNS:** inspect concerns before acceptance.
- **NEEDS_CONTEXT:** provide the missing context and re-dispatch.
- **BLOCKED:** resolve a deterministic seam, split the task, or ask the user only when the choice changes their goal/resources/done contract.

An interrupted worker that leaves coherent changes is an interrupted transaction. Audit and rescue the unresolved seam; do not restart the whole task over the dirty worktree.

## Red Flags

Never:

- dispatch an implementer without explicit controller review ownership;
- run reviewer subagents after every task by default;
- let implementer self-review replace controller verification;
- let controller verification replace the required final independent review for high-risk changes;
- let an implementer invoke Codex/Claude/reviewer agents on its own work;
- dispatch multiple implementers against overlapping files;
- accept worker self-report without diff/test readback;
- start an unbounded reviewer-fixer-reviewer loop;
- continue with open Critical/Important blockers.

## Integration

- `plan` defines the approved software contract.
- `test-driven-development` governs deterministic RED→GREEN behavior changes.
- `requesting-code-review` owns the one final independent review.
- `verification-before-completion` owns fresh completion evidence.
- `using-git-worktrees` owns isolation and safe integration.

## Further Reading

Load these only when their trigger applies:

- `references/context-budget-discipline.md` — context degradation and recovery for long multi-task runs.
- `references/gates-taxonomy.md` — pre-flight, revision, escalation, and abort gate semantics.
