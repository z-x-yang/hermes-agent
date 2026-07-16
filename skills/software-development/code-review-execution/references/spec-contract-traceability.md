# Spec Contract Traceability

Load this reference only when a software change is materially spec-driven: a large approved contract, shared protocol/schema/API, profile or config semantics, migration, or multiple subsystems whose wiring cannot be established from a short acceptance-criteria list.

## Contract map

1. Extract every normative requirement from the approved ask/spec/plan, including non-goals, negative requirements, defaults, compatibility promises, and explicitly approved scope changes. Give each a stable ID.
2. For every ID, record evidence for the runtime code path, behavioral test, and model-facing schema/docs when applicable.
3. Classify each row as `EXACT`, `PARTIAL`, `MISSING`, `DIVERGENT`, or `INTENTIONAL`.
4. A field, enum, prompt sentence, mock-only assertion, or broad passing suite is not proof that behavior is wired. Trace the production consumer and observable path; check for dead metadata, stale descriptions, config keys that never reach runtime, and tests that prove only construction.
5. `PARTIAL`, `MISSING`, or unexplained `DIVERGENT` rows block completion. Only a user-approved departure may become `INTENTIONAL`, with rationale.

If the contract has roughly more than 20 normative rows or spans multiple subsystems, do a dedicated read-only traceability pass before the adversarial bug/security pass. Any independent-review pass still counts against whatever review budget was explicitly approved for the change.

## Reviewer package

Give the reviewer the contract map together with the original authority, exact diff/range, fresh verification evidence, and relevant invariants. Require both:

- contract disposition for every normative row; and
- adversarial correctness review for bugs, security, races, data loss, and compatibility.

A review that finds no bugs but leaves normative rows undisposed is incomplete.

## Completion report

Preserve the full matrix as an artifact. Report only the disposition counts, artifact path, confirmed blockers, and deterministic verification evidence in chat.
