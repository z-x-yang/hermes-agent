# Stateful Database Change Acceptance Probes

Read this when delegated work changes a stateful database's open guard, schema classifier/migration, read-only diagnostics, or paired search result/snippet behavior. These are controller-owned acceptance probes; they supplement, not replace, the implementer's tests.

## 1. Prove a maintenance guard runs before the database opens

A guard helper existing or raising in isolation is insufficient. Instrument the real boundary:

1. Put the durable maintenance journal/lock into the blocking state.
2. Monkeypatch or wrap the lowest database-open primitive (`sqlite3.connect`, engine constructor, checkpoint opener).
3. Invoke every production writer/repair entry point.
4. Assert the domain-specific maintenance error occurs and the open primitive's call count remains zero.

Cover direct library construction, raw repair/doctor helpers, and CLI entry points separately. A blocked CLI must return a non-zero exit status and visible reason; read-only diagnostics must preserve any corruption evidence they obtained rather than clearing it into a false green state.

## 2. Test near-valid schemas, not only known versions

Schema classifiers often pass v1/v2 fixtures yet accept malformed hybrids. Starting from a valid schema, remove or alter one load-bearing clause at a time:

- external-content owner/table name;
- `content_rowid` ownership;
- tokenizer or tokenizer options;
- trigger ownership/body;
- required shadow/object family;
- version/marker coherence.

The classifier must reject the malformed state at startup with a clear fail-closed reason, before later SQL fails with an incidental low-level error. Assert rejection is zero-write: schema, metadata, data digest, and database bytes remain unchanged.

## 3. Verify deep immutability, not only `frozen=True`

A frozen dataclass can still expose mutable nested dictionaries/lists. Probe both mutation paths:

- mutate an object retained by the caller after construction;
- mutate a mapping/list obtained from the supposedly immutable record.

Then verify the trusted record/fingerprint is unchanged. Use defensive copies plus immutable containers (`MappingProxyType`, tuples/frozensets, or an immutable value object) at the trust boundary. Also verify a permit/token becomes stale after the backing journal is replaced or its fingerprint changes.

## 4. Keep match and snippet semantics paired

When search matching and snippet generation share a logical projection (for example `content + tool_name + tool_calls`), test them as one contract:

- rows that match only metadata/tool-call fields still return original body content;
- an OR query where only a later token matches marks that actual token for each row, not a globally chosen first token;
- case folding matches the database engine's real semantics;
- escaped `%`, `_`, and backslash stay literal;
- internal projection/helper columns never leak in returned rows;
- source, role, active/inactive, lineage, ordering, limit, and offset filters remain identical across indexed and fallback routes.

Use fixtures where content-only, metadata-only, and tool-call-only rows each match the same later token. Compare result ID sets across old/new physical schemas, then inspect row-specific markers and original content independently; set parity alone cannot catch a broken snippet contract.

## 5. Controller acceptance sequence

For each repair round:

1. Read the changed report revision once and extract commit/range, claims, and concerns.
2. Inspect the complete scoped diff and production call sites.
3. Run the implementer's focused and broad regression suites.
4. Run the smallest controller-owned hostile fixture for the exact contract dimension the suite may have missed.
5. Check compile/type/lint, `git diff --check`, changed-file scope, and clean status.
6. Close the task only when both valid behavior and fail-closed behavior match the contract.

Do not turn one discovered edge case into a one-off production special case. Require a RED regression that describes the contract class, then fix the root semantics.
