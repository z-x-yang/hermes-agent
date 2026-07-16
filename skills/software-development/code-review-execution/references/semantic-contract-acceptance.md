# Semantic Contract Acceptance Probes

Read this when a repair claims to close config wiring, typed identity, or fail-closed validation findings. These are general probes, not project-specific rules.

## 1. Typed identity must stay typed

Requirement:

```text
Copy and resolve rows by (Type, Code).
```

Insufficient substitutes:

- filter by `Code` only;
- bucket by `Code`, then reject if several Types appear;
- prove `(Type, Code)` uniqueness in one helper but drop `Type` before lookup.

Behavioral fixture:

```text
(Type=A, Code=X, Target=1)
(Type=B, Code=X, Target=2)
```

Ask for only `(A, X)`. Correct behavior ships/resolves Target 1 only. Target 2 must neither contaminate A nor make the valid A lookup fail merely because B exists.

Audit path end to end: source filter → serialized artifact → parser key → lookup key → terminal output.

## 2. A validator must be invoked at the boundary

Requirement:

```text
Runtime constants/config must fail closed on drift before materialization or execution.
```

Insufficient evidence:

- validator function exists;
- direct unit test calls validator;
- a later pipeline stage invokes it;
- one entry point invokes it while another production entry point bypasses it.

Probe by monkeypatching/drifting one runtime constant and invoking the real CLI/library boundary. Assert failure occurs before trusted output is written or external work begins.

## 3. Unknown keys must not disable validation

Dangerous pattern:

```python
allowed = allowed_by_type.get(type_name)
validate(value, allowed=allowed)  # None means "no constraint"
```

Correct boundary behavior:

```python
if type_name not in allowed_by_type:
    raise ValueError(...)
allowed = allowed_by_type[type_name]
if not isinstance(allowed, str) or not allowed.strip():
    raise ValueError(...)
```

Test unknown, blank, and missing keys, plus a valid subset/sensitivity path.

## 4. Hard gates must reject coercion

Dangerous patterns:

```python
blocked = bool(payload.get("blocked"))      # "false" -> True
count = int(payload["count"])              # 3.0 or "3" may pass unintentionally
name = str(payload.get("name", ""))        # None -> "None"
```

For a literal hard gate, require exact type/value (`value is True`, real integer but not bool, nonblank string, unique enumerated names). Test missing, false, string, integer, blank, and duplicate cases.

## 5. Cancellation and numerical edge states

When weighted vectors can cancel, distinguish:

- no selected events;
- selected events whose combined vector is zero;
- non-finite input/weight;
- near-zero numerical residue.

Do not label cancellation as “no events,” and do not normalize arbitrary near-zero noise into a unit vector. Assert the intended terminal state/error for each case.

## Acceptance note

A fail-closed implementation can still be contract-wrong. “It errors instead of producing a bad result” is not enough when the requirement says a valid input must be preserved and processed. Verify both safety **and** intended valid behavior.
