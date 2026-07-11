# Thin Continuation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不新增 broker、grant store 或持久化系统的前提下，让 retained continuation 保持同一 canonical profile 与原始 exact authority，并让 Notion/Mail 读取正文通过现有 `HANDLE_ONLY` 路径进入 retained/persistent storage。

**Architecture:** 在 `RetainedSubagentSession` 中保存 spawn 时已经存在的 governance diagnostics 与 exact authority identities；continuation 用 latest governance 重建 child，但把 current policy 与 retained identities 再取交集。数据源脱敏只修改现有 MCP descriptor retention，并在两处 retained-history choke point 调用现成 `project_messages_for_retention()`。

**Tech Stack:** Python dataclasses、Hermes tool registry/policy descriptors、pytest、ruff。

## Global Constraints

- Task 10 trusted action-grant subsystem 已取消，不创建 `action_grants.py`。
- 不恢复 Notion REST broker、Apple Mail broker、PEEK/version pin 或新 storage layer。
- continuation 继续 process-local、restart-ephemeral、TTL/count/byte bounded。
- 所有外部系统测试使用 fake MCP/local canary，不访问真实 Notion/Mail，不重启 live gateway。
- Tasks 11–12 由主会话 inline 执行。

---

### Task 1: Bind retained continuation to canonical profile and original exact authority

**Files:**
- Modify: `tools/subagent_sessions.py:11-49`
- Modify: `tools/delegate_tool.py:2605-2640`
- Modify: `tools/delegate_continue_tool.py:108-235,407-438`
- Test: `tests/tools/test_delegate_continue.py`
- Test: `tests/tools/test_delegate.py`

**Interfaces:**
- Consumes: `GovernanceSnapshot.profile_id/profile_home/fingerprint`, `ToolNamePolicy.authority_snapshot`, `build_authority_snapshot()`.
- Produces: `RetainedSubagentSession.profile_id`, `canonical_profile_home`, `original_policy_identities`, `original_governance_fingerprint`; continuation fail-closed profile gate and exact-identity intersection.

- [ ] **Step 1: Write failing metadata and behavior tests**

Add factory defaults and assertions equivalent to:

```python
record = _record(
    profile_id="default",
    canonical_profile_home=str(tmp_path / ".hermes"),
    original_policy_identities=frozenset({original_identity}),
    original_governance_fingerprint="a" * 64,
)

with patch(
    "tools.delegate_continue_tool.load_governance_snapshot",
    return_value=SimpleNamespace(
        profile_id="other",
        profile_home=tmp_path / "other-home",
        fingerprint="b" * 64,
    ),
):
    payload = json.loads(delegate_continue(...))
assert "profile" in payload["error"].lower()
assert child_backend_calls == 0
```

Add a same-name replacement test whose current child policy contains `replacement_identity` but retained metadata contains only `original_identity`; after `_build_continuation_child`, assert replacement identity is absent from `policy.authority_snapshot.policy_identities`. Add a current-parent-removal test proving names and identities both narrow.

- [ ] **Step 2: Run RED tests**

Run:

```bash
python -m pytest tests/tools/test_delegate_continue.py tests/tools/test_delegate.py -o 'addopts=' -p no:cacheprovider -q
```

Expected: new tests fail because retained profile/home/original identities do not exist and continuation only intersects names.

- [ ] **Step 3: Add retained metadata at spawn**

Extend the dataclass with required fields:

```python
profile_id: str
canonical_profile_home: str
original_policy_identities: frozenset[str]
original_governance_fingerprint: str
```

In `delegate_tool.py`, fail retention (without failing the already completed child result) unless both diagnostics and exact authority exist, then populate from:

```python
governance = child._governance_diagnostics
policy = child._subagent_tool_policy
snapshot = policy.authority_snapshot
```

Keep credentials and governance bodies out of the record.

- [ ] **Step 4: Enforce profile and identity intersections**

Before rebuilding continuation, load the latest snapshot and verify:

```python
if latest.profile_id != record.profile_id:
    raise ValueError("Retained subagent profile changed; refusing continuation.")
if str(latest.profile_home.resolve()) != record.canonical_profile_home:
    raise ValueError("Retained subagent canonical profile home changed; refusing continuation.")
```

Pass that same `latest` snapshot into `_build_child_agent`. Replace the current policy with:

```python
identity_intersection = (
    current_policy.authority_snapshot.policy_identities
    & record.original_policy_identities
)
narrowed_snapshot = build_authority_snapshot(
    identity_intersection,
    registry_generation=registry._generation,
)
replace(
    current_policy,
    allowed_names=current_allowed & retained_ceiling,
    authority_snapshot=narrowed_snapshot,
)
```

Reject missing metadata, empty identity intersection, or empty name intersection; never fall back to name-only authorization.

- [ ] **Step 5: Run GREEN and related continuation tests**

```bash
python -m pytest tests/tools/test_delegate_continue.py tests/tools/test_delegate*.py tests/tools/test_subagent_tool_policy.py -o 'addopts=' -p no:cacheprovider -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add tools/subagent_sessions.py tools/delegate_tool.py tools/delegate_continue_tool.py tests/tools/test_delegate_continue.py tests/tools/test_delegate.py
git commit -m "fix(delegation): bind retained continuation authority"
```

---

### Task 2: Reuse HANDLE_ONLY for Notion/Mail retained content

**Files:**
- Modify: `tools/mcp_tool.py:4160-4245`
- Modify: `tools/delegate_tool.py:2605-2640`
- Modify: `tools/delegate_continue_tool.py:331-340`
- Test: `tests/tools/test_mcp_dynamic_discovery.py`
- Test: `tests/tools/test_delegate_continue.py`
- Test: `tests/run_agent/test_tool_call_incremental_persistence.py`
- Modify: `website/docs/user-guide/features/delegation.md`

**Interfaces:**
- Consumes: `ResultRetention.HANDLE_ONLY`, `project_messages_for_retention(messages, retention_by_tool_call_id)` and child `_subagent_tool_result_retention_by_call_id`.
- Produces: selected existing Notion/Mail read descriptors with `HANDLE_ONLY`; projected initial and continued retained history.

- [ ] **Step 1: Write failing descriptor and retained-history tests**

Extend MCP discovery assertions:

```python
assert notion_read_descriptor.retention is ResultRetention.HANDLE_ONLY
assert mail_read_descriptor.retention is ResultRetention.HANDLE_ONLY
assert mail_send_descriptor.retention is ResultRetention.DEFAULT
```

Add initial-retention and continuation-update tests with a tool message containing `RAW_BODY_CANARY`. Seed the child retention index with `{call_id: ResultRetention.HANDLE_ONLY}` and assert the stored `RetainedSubagentSession.conversation_history` does not contain the full canary body, but includes `"retention": "handle_only"` and a `sha256:` handle.

- [ ] **Step 2: Run RED tests**

```bash
python -m pytest tests/tools/test_mcp_dynamic_discovery.py tests/tools/test_delegate_continue.py tests/run_agent/test_tool_call_incremental_persistence.py -o 'addopts=' -p no:cacheprovider -q
```

Expected: descriptor retention is `DEFAULT` and retained histories contain raw tool content. Baseline note: this combined ordering currently has one existing incremental-persistence pollution failure; the exact failing node passes isolated. New assertions must fail for the intended reason.

- [ ] **Step 3: Mark only approved read names HANDLE_ONLY**

Change `_mcp_effect_policy` to return retention alongside effects/resolver:

```python
if registered_name in APPLE_MAIL_READ_TOOL_NAMES:
    return frozenset({ToolEffect.READ_REMOTE}), None, ResultRetention.HANDLE_ONLY
if registered_name in NOTION_PROMPT_READ_TOOL_NAMES:
    return (
        frozenset({ToolEffect.UNKNOWN}),
        _notion_ai_ask_effects,
        ResultRetention.HANDLE_ONLY,
    )
return frozenset({ToolEffect.UNKNOWN}), None, ResultRetention.DEFAULT
```

Pass `retention=retention` to `mcp_policy_descriptor`. Explicit Mail write tools stay `DEFAULT` and outside Explore/Plan profiles.

- [ ] **Step 4: Project both retained-history choke points**

Before creating the initial retained record and before `update_retained_history`, call:

```python
retained_messages = project_messages_for_retention(
    list(messages),
    getattr(child, "_subagent_tool_result_retention_by_call_id", None),
)
```

Store `retained_messages`, not the live raw list. Do not mutate the live in-memory conversation.

- [ ] **Step 5: Run GREEN, persistence, and regression tests**

```bash
python -m pytest tests/tools/test_mcp_dynamic_discovery.py tests/tools/test_delegate_continue.py tests/run_agent/test_tool_call_incremental_persistence.py tests/tools/test_subagent_effect_enforcement.py -o 'addopts=' -p no:cacheprovider -q
python -m pytest tests/run_agent/test_tool_call_incremental_persistence.py::test_run_conversation_flushes_assistant_tool_call_before_execution -o 'addopts=' -p no:cacheprovider -q
```

Expected: all new tests pass; if the known order-polluted node still fails only in the combined run but passes isolated, record it as pre-existing evidence rather than modifying unrelated runtime code.

- [ ] **Step 6: Update docs and commit**

Document that Notion/Mail retained/persistent results use `HANDLE_ONLY`, without claiming broker/PEEK/version-pin guarantees.

```bash
git add tools/mcp_tool.py tools/delegate_tool.py tools/delegate_continue_tool.py tests/tools/test_mcp_dynamic_discovery.py tests/tools/test_delegate_continue.py tests/run_agent/test_tool_call_incremental_persistence.py website/docs/user-guide/features/delegation.md
git commit -m "fix(delegation): project retained data-source results"
```

---

### Task 3: Hand off to final integration and traceability

**Files:**
- Modify: `.hermes/plans/2026-07-10-subagent-data-source-governance-implementation.md` (local plan annotation only; do not force-add unrelated ignored artifacts)
- Modify/Create under Task 12: controller evidence/report artifacts already named by the parent plan.

**Interfaces:**
- Consumes: committed Task 1/2 behavior and all prior Tasks 1–8 evidence.
- Produces: Task 12 input with Task 9/10 cancellations and H06/H07/H13 `INTENTIONAL` classifications.

- [ ] **Step 1: Reconcile scope**

Mark Task 10 cancelled and Task 11 replaced by this thin plan. Keep original requirements visible for traceability; do not delete them.

- [ ] **Step 2: Run cheap plan verification**

```bash
git diff --check
python -m ruff check tools/subagent_sessions.py tools/delegate_tool.py tools/delegate_continue_tool.py tools/mcp_tool.py tests/tools/test_delegate_continue.py tests/tools/test_mcp_dynamic_discovery.py
python -m py_compile tools/subagent_sessions.py tools/delegate_tool.py tools/delegate_continue_tool.py tools/mcp_tool.py
```

Expected: all commands exit 0.

- [ ] **Step 3: Continue directly to Task 12**

Do not request another execution choice. Run final targeted/broad/race/FD/schema evidence, the 73-row and H01–H18 adjudications, and one Codex adversarial correctness + design-contract traceability review. Stop before merge, gateway reload/restart, or live rollout.
