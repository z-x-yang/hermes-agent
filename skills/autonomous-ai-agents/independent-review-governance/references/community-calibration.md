# Community calibration for agentic independent review

Read this reference when changing the review policy, auditing whether it matches mature practice, or deciding which external mechanism to borrow. It is not an every-review checklist.

## Primary-source baseline (accessed 2026-07-11)

### Google Engineering Practices

- [What to look for in a code review](https://google.github.io/eng-practices/review/reviewer/looking-for.html): review design, functionality, complexity, tests, documentation, concurrency, and system context; tests themselves require scrutiny.
- [The Standard of Code Review](https://google.github.io/eng-practices/review/reviewer/standard.html): approve once a change clearly improves overall code health; do not hold progress hostage to perfection or non-blocking polish.

**Borrow:** broad correctness axes, evidence over preference, blocker-vs-nit distinction.

### GitHub Copilot code review

- [About GitHub Copilot code review](https://docs.github.com/en/copilot/concepts/agents/code-review): review effort should match criticality; GitHub explicitly says Copilot may miss issues or make mistakes and its feedback must be validated and supplemented with human review.

**Borrow:** risk-tiered effort and controller validation of AI findings.

### OWASP

- [Secure Code Review Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secure_Code_Review_Cheat_Sheet.html): targeted review is important for input validation, authentication/session management, authorization, deserialization, cryptography, business logic, and race conditions.

**Borrow:** mandatory adversarial focus for security and concurrency seams rather than generic style review.

### GitHub Spec Kit

- [github/spec-kit](https://github.com/github/spec-kit): separates specification (what/why), technical plan, tasks, and implementation; `/speckit.analyze` checks cross-artifact consistency and coverage, while `/speckit.converge` assesses code against spec/plan/tasks.

**Borrow:** requirements-to-implementation traceability for spec-driven work.

**Modify:** do not require the full Spec→Plan→Tasks chain for every change. Artifacts are independently triggered; one combined packet or approved chat decision is enough when it preserves the contract.

### OpenAI harness engineering

- [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering): uses local self-review plus additional targeted agent reviews; lightweight plans for small changes and durable execution plans for complex work; repository knowledge is progressively disclosed.

**Borrow:** separate implementation from final judgment, give reviewers scoped repository-legible evidence, and scale planning artifacts.

**Reject as a universal default:** looping until every agent reviewer is satisfied. Default to one stable whole-change pass; permit one targeted closure pass only after confirmed blockers materially reshape risk and controller evidence cannot safely close it. Further review requires explicit user/domain-owner escalation and a genuinely new information source. This preserves Google's progress-over-perfection principle without turning “two” into a mechanical quality proxy.

### GitHub pull-request re-review continuity

- [About pull request reviews](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews): review threads remain visible/resolvable, and re-review is requested after significant changes rather than after every edit.

**Borrow:** carry forward explicit finding threads and re-request only after material change.

**Modify for one-shot AI reviewers:** do not retain the full reviewer conversation. Give a fresh reviewer a minimal closure packet containing confirmed finding IDs, controller dispositions, exact repair diff, and fresh deterministic evidence. This preserves relevant continuity without stale tool output, context bloat, or commitment bias.

## Local adaptations that should remain explicit

1. **Risk beats raw size.** A small auth or rollback change can need review; a large generated fixture may not. Line count is only a scoping signal.
2. **Traceability is stricter for agent-generated code.** A field, enum, prompt sentence, or passing construction test may be dead. Require an actual runtime consumer and observable behavioral evidence.
3. **Fresh context is minimum independence, not full epistemic independence.** If the same model family implemented and reviewed the change, correlated blind spots remain. Controller-owned tests/readback are mandatory; highest-stakes security, privacy, public-release, or irreversible changes may also require a qualified human/domain reviewer or a distinct permitted reviewer.
4. **AI review findings are untrusted until reproduced or grounded in code/tests.** Never fix a comment merely because a reviewer labeled it critical.
5. **Domain review stays domain-owned.** This code-review policy does not replace scientific, manuscript, figure, email, Notion, cluster, or operational oracles.
