# Claude-like Subagent 简化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Hermes subagent 的 model-facing API、prompt 与 lifecycle 收敛到 Claude Code 的简单形态，同时保留 Zongxin 明确要求的 Batch API、完整 Evelyn governance、Notion/Mail 数据源、provider fallback、独立 timeout 和既有 runtime safety hardening。

**Architecture:** `delegate_task` 保留 single/batch 两种入口，但统一使用 `description + prompt`，并以一个 top-level `run_in_background` 控制整次调用；Batch 继续提供并发组、单 handle 和 consolidated completion。Explore/Plan 固定 one-shot、read-only 且跳过 repo context；general-purpose 自动 retained，并复用现有 project-context 与 coding-workspace helpers 加载真实 repo rules/git snapshot。删除 model-facing role/retention/三态 scheduling、统一 12 字段 result checklist、动态 delegation schema 和无真实 consumer 的 context capsule/profile booleans；resolved-call security、continuation race、HANDLE_ONLY、governance preflight、provider fallback 与 timeout 不改弱。

**Tech Stack:** Python 3.11、pytest、Hermes Tool Registry、Docusaurus、Markdown、Git worktree。

## Global Constraints

- 只暴露 `Explore`、`Plan`、`general-purpose`；省略/空 `subagent_type` 仍解析为 `general-purpose`。
- **保留 Batch API**：`tasks[]` 是 intentional Hermes divergence；一个 batch 只有一个 handle 和一个 consolidated completion。
- Single 使用 `description + prompt`；Batch item 使用 `description + prompt + optional subagent_type`。删除 `goal/context/role/retain_session/scheduling/background` model/internal compatibility 参数。
- Top-level `run_in_background?: bool` 控制整次 single/batch；顶层省略默认 background，nested 省略默认 foreground，nested 显式 `true` fail closed。
- Explore/Plan 固定 one-shot；general-purpose 成功后在 parent session/容量允许时自动 retained。
- Explore/Plan 继续完整接收最新 `SOUL.md/MEMORY.md/USER.md`，但不加载 repo context/git snapshot；general-purpose 除完整 governance 外，加载真实 project context 与 workspace snapshot。
- 删除统一 12 字段 result contract。Explore 返回 concise evidence report；Plan 以 `### Critical Files for Implementation` + 3–5 个文件结尾；general-purpose 按 parent prompt 指定的 deliverable 返回一条 concise final message。
- 删除 model-facing role；nested delegation 由 profile、parent exact authority、operator kill switch 和 `max_spawn_depth` 自动决定。默认 `max_spawn_depth=1` 仍是 flat delegation。
- 不改弱 `ToolPolicyDescriptor` / exact resolved identity / frozen args / middleware reauthorization / registry TOCTOU / backend-zero rejection。
- 不改弱 claim generation、interrupt cancellation、late-commit rejection、capacity rejection、workspace safety notice、retention failure surfacing、Notion/Mail `HANDLE_ONLY`。
- 不改 provider fallback、所有 provider 的完整 governance snapshot、preflight/final-request context-fit 和 profile-specific wait/run timeout。
- 不新增 cross-restart persistence、custom profile registry、per-agent worktree、mid-run steering、per-call model override。
- 实现、测试、文档、active skill、traceability 全部完成后仍停在 merge/restart 前；Gateway 不在本计划中重启。
- 当前 worktree baseline：`314 passed in 14.63s`，命令见 Task 1 Step 2。

---

### Task 1: 精简 profile prompt，并补齐 general-purpose 真实项目上下文

**Files:**
- Modify: `tools/subagent_profiles.py`
- Modify: `tools/delegate_tool.py:824-959,1273-1646`
- Delete: `agent/subagent_context_policy.py`
- Test: `tests/tools/test_subagent_profiles.py`
- Test: `tests/tools/test_delegate_prompt_layers.py`
- Test: `tests/tools/test_delegate_context_cwd.py`

**Interfaces:**
- Consumes: existing `agent.prompt_builder.build_context_files_prompt(cwd, skip_soul, context_length)` and `agent.coding_context.build_coding_workspace_block(cwd)`.
- Produces: lean `SubagentProfile` without `result_contract`, `context_policy`, `can_write_files`, `can_external_side_effects`, or `can_delegate`; `_build_child_system_prompt(..., allow_delegation: bool, ...)` with profile-specific final guidance and real GP project context.

- [ ] **Step 1: Write RED tests for profile-specific final guidance and real context loading**

Replace `test_all_profiles_share_the_complete_result_contract` with:

```python
def test_profiles_use_claude_like_type_specific_final_guidance():
    explore = get_subagent_profile("Explore").system_instructions
    plan = get_subagent_profile("Plan").system_instructions
    gp = get_subagent_profile("general-purpose").system_instructions

    for prompt in (explore, plan, gp):
        assert "recommended_next_step" not in prompt
        assert "files_changed" not in prompt
        assert "side_effects" not in prompt
    assert "clearly and concisely" in explore
    assert "absolute file paths" in explore
    assert "### Critical Files for Implementation" in plan
    assert "3-5" in plan
    assert "exact return requirements in the task prompt" in gp
```

Add to `tests/tools/test_delegate_prompt_layers.py`:

```python
def test_general_purpose_loads_real_project_context_but_readonly_profiles_skip_it(
    tmp_path,
):
    (tmp_path / "AGENTS.md").write_text("PROJECT_SENTINEL", encoding="utf-8")
    gp = _build_child_system_prompt(
        profile=get_subagent_profile("general-purpose"),
        allow_delegation=False,
        workspace_path=str(tmp_path),
        child_depth=1,
        max_spawn_depth=1,
        governance_snapshot=None,
    )
    explore = _build_child_system_prompt(
        profile=get_subagent_profile("Explore"),
        allow_delegation=False,
        workspace_path=str(tmp_path),
        child_depth=1,
        max_spawn_depth=1,
        governance_snapshot=None,
    )
    assert "PROJECT_SENTINEL" in gp
    assert "Workspace (snapshot at session start" in gp
    assert "PROJECT_SENTINEL" not in explore
    assert "Workspace (snapshot at session start" not in explore
```

Update prompt-layer callers to pass `allow_delegation=False` and remove `role/context_policy_capsule` assertions.

- [ ] **Step 2: Run the focused baseline/RED gate**

Run:

```bash
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/tools/test_delegate.py \
  tests/tools/test_delegate_continue.py \
  tests/tools/test_delegate_context_cwd.py \
  tests/tools/test_delegate_prompt_layers.py \
  tests/tools/test_delegate_toolset_scope.py \
  tests/tools/test_subagent_profiles.py \
  tests/tools/test_subagent_effect_enforcement.py \
  tests/agent/test_subagent_governance.py \
  tests/agent/test_subagent_governance_preflight.py \
  -o 'addopts=' -p no:cacheprovider -q
```

Expected before implementation: new profile/context tests FAIL while the recorded baseline remains `314 passed in 14.63s` on `00cd0985b`.

- [ ] **Step 3: Replace profile metadata and final guidance**

Refactor `SubagentProfile` to the exact shape:

```python
@dataclass(frozen=True)
class SubagentProfile:
    name: str
    description: str
    system_instructions: str
    model: str
    provider: Optional[str]
    allowed_tool_names: Optional[FrozenSet[str]]
    default_wait_timeout_seconds: int
    default_run_timeout_seconds: int
```

Keep existing read-only tool-name sets and argument-sensitive effect policy. Remove `_COMPLETE_RESULT_FIELDS`, `_result_contract`, `default_scheduling`, `context_policy`, and the four capability booleans.

Use these final-guidance clauses:

```python
EXPLORE_FINAL = (
    "Report findings clearly and concisely in one final message. Include absolute "
    "file paths and relevant symbols or line ranges for claims the parent must verify."
)
PLAN_FINAL = (
    "Return an actionable implementation plan without making changes. End with "
    "`### Critical Files for Implementation` and list 3-5 files that are central "
    "to the plan, each with a one-line reason."
)
GENERAL_FINAL = (
    "Return one concise final message that follows the exact return requirements in "
    "the task prompt. The parent will verify claimed changes and side effects."
)
```

- [ ] **Step 4: Delete the capsule and load real project context for GP**

Delete `agent/subagent_context_policy.py`. In `tools/delegate_tool.py`, import:

```python
from agent.coding_context import build_coding_workspace_block
from agent.prompt_builder import build_context_files_prompt
```

Change `_build_child_system_prompt` to accept `allow_delegation: bool` instead of `role/context_policy_capsule`. Append real project context only for GP:

```python
if profile.name == "general-purpose" and workspace_path:
    project_context = build_context_files_prompt(
        cwd=workspace_path,
        skip_soul=True,
    )
    workspace_snapshot = build_coding_workspace_block(workspace_path)
    if project_context:
        sections.append(project_context)
    if workspace_snapshot:
        sections.append(workspace_snapshot)
```

Keep governance snapshot injection separate and unchanged. Remove `_trusted_project_routes` reads and `<SUBAGENT_CONTEXT_POLICY>` output. Keep workspace path validation already performed by `_resolve_workspace_hint` / continuation workspace resolution.

- [ ] **Step 5: Run GREEN tests**

Run:

```bash
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/tools/test_subagent_profiles.py \
  tests/tools/test_delegate_prompt_layers.py \
  tests/tools/test_delegate_context_cwd.py \
  tests/agent/test_subagent_governance.py \
  tests/agent/test_subagent_governance_preflight.py \
  -o 'addopts=' -p no:cacheprovider -q
```

Expected: PASS; GP contains real repo context/workspace snapshot, Explore/Plan do not, and all profiles still receive the governance snapshot through the existing governance tests.

- [ ] **Step 6: Commit**

```bash
git add tools/subagent_profiles.py tools/delegate_tool.py \
  agent/subagent_context_policy.py \
  tests/tools/test_subagent_profiles.py \
  tests/tools/test_delegate_prompt_layers.py \
  tests/tools/test_delegate_context_cwd.py
git commit -m "refactor(delegation): use Claude-like profile prompts"
```

---

### Task 2: 收敛 static single/batch schema，保留 Batch API

**Files:**
- Modify: `tools/delegate_tool.py:519-583,2776-3050,3880-4261`
- Modify: `tests/tools/test_delegate.py`
- Modify: `tests/tools/test_delegate_context_cwd.py`
- Modify: `tests/tools/test_subagent_effect_enforcement.py`

**Interfaces:**
- Consumes: profile resolution and existing background batch registry/consolidated completion implementation.
- Produces: static `DELEGATE_TASK_SCHEMA`; `delegate_task(description=None, prompt=None, tasks=None, *, subagent_type=None, run_in_background=None, parent_agent=None)` model-facing contract; batch items `{description,prompt,subagent_type?}`.

- [ ] **Step 1: Write RED schema tests**

Replace the old schema assertions in `tests/tools/test_delegate.py` with:

```python
def test_schema_is_static_claude_like_and_keeps_batch():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert set(props) == {
        "description",
        "prompt",
        "tasks",
        "subagent_type",
        "run_in_background",
    }
    assert props["subagent_type"]["enum"] == [
        "Explore",
        "Plan",
        "general-purpose",
    ]
    item_props = props["tasks"]["items"]["properties"]
    assert set(item_props) == {"description", "prompt", "subagent_type"}
    assert props["tasks"]["items"]["required"] == ["description", "prompt"]
    for removed in ("goal", "context", "role", "retain_session", "scheduling", "background"):
        assert removed not in props
        assert removed not in item_props
```

Add:

```python
def test_batch_keeps_one_handle_and_one_consolidated_completion_contract():
    description = DELEGATE_TASK_SCHEMA["description"]
    assert "one batch handle" in description
    assert "one consolidated completion" in description
    assert "multiple independent tasks" in description
```

Change the registry test to assert `delegate_task` has no `dynamic_schema_overrides`, while image/video dynamic-schema tests remain unchanged.

- [ ] **Step 2: Run RED tests**

Run:

```bash
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/tools/test_delegate.py \
  tests/tools/test_subagent_effect_enforcement.py \
  -o 'addopts=' -p no:cacheprovider -q
```

Expected: FAIL on legacy fields and dynamic delegation schema.

- [ ] **Step 3: Define the static schema**

Replace the delegation schema with this stable shape:

```python
_SUBAGENT_TYPE_SCHEMA = {
    "type": "string",
    "enum": ["Explore", "Plan", "general-purpose"],
    "description": (
        "Explore searches and explains read-only evidence; Plan produces a read-only "
        "implementation plan; general-purpose executes multi-step work. Omission "
        "resolves to general-purpose."
    ),
}

DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    "description": (
        "Delegate one self-contained task or a batch of multiple independent tasks. "
        "A batch runs concurrently and produces one batch handle and one consolidated "
        "completion. Children have fresh context, so prompts must include all required "
        "paths, constraints, and exact return requirements. Work runs in the background "
        "by default; set run_in_background=false only when the result is needed before "
        "continuing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "3-5 word progress label."},
            "prompt": {"type": "string", "description": "Self-contained delegated task."},
            "tasks": {
                "type": "array",
                "description": "Independent tasks executed concurrently as one batch.",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                        "subagent_type": _SUBAGENT_TYPE_SCHEMA,
                    },
                    "required": ["description", "prompt"],
                },
            },
            "subagent_type": _SUBAGENT_TYPE_SCHEMA,
            "run_in_background": {"type": "boolean"},
        },
    },
}
```

Do not add config values, max concurrency, max depth, timeout, retention budget, role, or operator kill-switch prose to the schema.

- [ ] **Step 4: Normalize single and batch calls**

Change `_build_child_agent` and all progress/run helpers from `goal/context` to `description/prompt`, then change the public handler signature to:

```python
def delegate_task(
    description: Optional[str] = None,
    prompt: Optional[str] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    *,
    subagent_type: Optional[str] = None,
    run_in_background: Optional[bool] = None,
    parent_agent=None,
) -> str:
```

Runtime rules:

```python
if tasks is not None:
    # reject simultaneous description/prompt; enforce config max; copy each dict
elif description and prompt:
    task_list = [{
        "description": description,
        "prompt": prompt,
        "subagent_type": resolve_subagent_type(subagent_type),
    }]
else:
    return tool_error("Provide description+prompt for one task, or tasks for a batch.")
```

Validate each label is nonempty and each prompt is nonempty. Keep `_recover_tasks_from_json_string()` only if stale providers still stringify the `tasks` array; update its error text to the new contract. Remove direct-Python `goal/context`, `_dispatch_origin`, legacy `background`, `max_iterations`, ACP override, and auto/sync compatibility branches.

Resolve the whole call once:

```python
def _resolve_run_in_background(
    requested: Optional[bool], *, is_subagent: bool
) -> bool:
    if is_subagent:
        if requested is True:
            raise ValueError("Nested delegation cannot run in the background.")
        return False
    return True if requested is None else bool(requested)
```

A batch never mixes foreground/background per item. Keep current ThreadPoolExecutor fan-out, batch registry, one-handle response, consolidated queue message, capacity rejection, interrupt behavior, and per-profile run timeout selection.

Rename progress labels from `goal` to `description`; pass only `prompt` into `_build_child_task_payload()` as an untrusted user/task message:

```python
def _build_child_task_payload(prompt: str) -> str:
    return (
        "<DELEGATED_TASK_DATA trust=\"untrusted\">\n"
        + json.dumps({"prompt": prompt.strip()}, ensure_ascii=False)
        + "\n</DELEGATED_TASK_DATA>"
    )
```

- [ ] **Step 5: Register a static delegation entry**

Remove `_build_dynamic_schema_overrides()` and register `delegate_task` without `dynamic_schema_overrides`. Do not modify registry support for dynamic image/video schemas or their resolved-identity tests.

- [ ] **Step 6: Run GREEN tests**

Run:

```bash
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/tools/test_delegate.py \
  tests/tools/test_delegate_context_cwd.py \
  tests/tools/test_subagent_effect_enforcement.py \
  -o 'addopts=' -p no:cacheprovider -q
```

Expected: PASS, including one batch handle/consolidated completion and stable delegation policy identity.

- [ ] **Step 7: Commit**

```bash
git add tools/delegate_tool.py \
  tests/tools/test_delegate.py \
  tests/tools/test_delegate_context_cwd.py \
  tests/tools/test_subagent_effect_enforcement.py
git commit -m "refactor(delegation): simplify the model-facing task schema"
```

---

### Task 3: 删除 role protocol，按 profile/authority/depth 自动派生 nesting

**Files:**
- Modify: `tools/delegate_tool.py:388-402,637-649,824-944,1273-1646`
- Modify: `agent/subagent_tool_policy.py`
- Modify: `tools/subagent_sessions.py:15-47`
- Modify: `tools/delegate_continue_tool.py:25-52,112-230`
- Test: `tests/tools/test_delegate_toolset_scope.py`
- Test: `tests/tools/test_delegate.py`
- Test: `tests/tools/test_delegate_continue.py`
- Test: `tests/agent/test_subagent_governance.py`

**Interfaces:**
- Consumes: existing parent exact authority snapshot, `delegation.orchestrator_enabled`, `max_spawn_depth`, and exact-name deny policy.
- Produces: `_child_can_delegate(profile_name, parent_agent, child_depth, max_spawn_depth) -> bool`; retained records without `role`; continuation recomputes current delegation availability under the original/current authority intersection.

- [ ] **Step 1: Write RED tests for automatic nesting**

Replace role-based cases with:

```python
def test_general_purpose_automatically_gets_delegate_task_only_when_all_gates_allow(
    monkeypatch,
):
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.valid_tool_names = {"read_file", "delegate_task"}
    parent.enabled_toolsets = {"file", "delegation"}
    parent._active_children = []
    parent._active_children_lock = None
    child = MagicMock()
    child.valid_tool_names = {"read_file", "delegate_task"}
    _set_authority(parent, parent.valid_tool_names)
    _set_authority(child, child.valid_tool_names)
    monkeypatch.setattr(dt, "_get_orchestrator_enabled", lambda: True)
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 2)
    with patch("run_agent.AIAgent", return_value=child):
        built = dt._build_child_agent(
            task_index=0,
            description="decompose work",
            prompt="split this task",
            toolsets=None,
            model=None,
            max_iterations=5,
            task_count=1,
            parent_agent=parent,
            profile=get_subagent_profile("general-purpose"),
        )
    assert "delegate_task" in built.valid_tool_names
```

Parametrize fail-closed cases: Explore, Plan, kill switch false, `child_depth >= max_spawn_depth`, and parent exact authority missing `delegate_task`; every case must exclude `delegate_task`, `delegate_continue`, and `clarify`.

Add retained-session assertions that `RetainedSubagentSession` has no `role` field and continuation cannot gain delegation when current parent/depth/kill-switch no longer allow it.

- [ ] **Step 2: Run RED tests**

Run:

```bash
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/tools/test_delegate_toolset_scope.py \
  tests/tools/test_delegate_continue.py \
  tests/agent/test_subagent_governance.py \
  -o 'addopts=' -p no:cacheprovider -q
```

Expected: FAIL because role is still caller-controlled and retained.

- [ ] **Step 3: Replace role with a derived boolean**

Delete `_normalize_role` and all schema/task/result `_child_role` fields. Introduce:

```python
def _parent_exposes_tool_name(parent_agent, name: str) -> bool:
    marker = object()
    try:
        static_names = inspect.getattr_static(parent_agent, "valid_tool_names", marker)
    except Exception:
        return False
    if static_names is marker:
        return False
    return name in frozenset(getattr(parent_agent, "valid_tool_names", set()) or set())


def _child_can_delegate(
    *,
    profile_name: str,
    parent_agent,
    child_depth: int,
    max_spawn_depth: int,
) -> bool:
    if profile_name != "general-purpose":
        return False
    if not _get_orchestrator_enabled() or child_depth >= max_spawn_depth:
        return False
    return _parent_exposes_tool_name(parent_agent, "delegate_task")
```

This name-level availability check only decides whether to assemble the delegation toolset; `build_child_tool_policy()` must still perform the existing exact resolved-identity intersection before dispatch. Never derive availability from composite toolset labels. In `_build_child_agent`, add the delegation toolset only when this returns true. Build `denied_names` as:

```python
denied_names = (
    frozenset({"delegate_continue", "clarify"})
    if allow_delegation
    else frozenset({"delegate_task", "delegate_continue", "clarify"})
)
```

The child prompt should state whether delegation is available and the current depth, but must not mention leaf/orchestrator roles or instruct the model to pass a role.

- [ ] **Step 4: Remove role from retained state and continuation**

Delete `role` from `RetainedSubagentSession`, constructors, size accounting, result metadata, and continuation reconstruction. Continuation must rebuild a GP child from the retained exact authority ceiling intersected with current parent/current registry authority; it may retain `delegate_task` only if current depth/kill-switch also allow it.

Keep claim generation, cancellation, handle ownership, workspace validation, transcript projection, retention status, files-written warning, and late-commit behavior unchanged.

- [ ] **Step 5: Run GREEN tests**

Run:

```bash
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/tools/test_delegate_toolset_scope.py \
  tests/tools/test_delegate.py \
  tests/tools/test_delegate_continue.py \
  tests/agent/test_subagent_governance.py \
  -o 'addopts=' -p no:cacheprovider -q
```

Expected: PASS; nesting is automatic only for GP under all gates, and role is absent from schema/state/results.

- [ ] **Step 6: Commit**

```bash
git add tools/delegate_tool.py agent/subagent_tool_policy.py \
  tools/subagent_sessions.py tools/delegate_continue_tool.py \
  tests/tools/test_delegate_toolset_scope.py tests/tools/test_delegate.py \
  tests/tools/test_delegate_continue.py tests/agent/test_subagent_governance.py
git commit -m "refactor(delegation): derive nesting from runtime authority"
```

---

### Task 4: 固定 lifecycle，并简化 continuation scheduling

**Files:**
- Modify: `tools/delegate_tool.py:519-523,3030-3450`
- Modify: `tools/delegate_continue_tool.py:25-52,430-680`
- Modify: `tools/subagent_sessions.py`
- Test: `tests/tools/test_delegate.py`
- Test: `tests/tools/test_delegate_continue.py`
- Test: `tests/tools/test_subagent_data_source_integration.py`

**Interfaces:**
- Consumes: existing retained store, TTL/count/byte budgets, exact authority capture, claim generation, async registry, and profile timeout config.
- Produces: `_should_retain_session(subagent_type: str) -> bool`; `delegate_continue(agent_id, prompt, run_in_background=None)`; GP-only successful retention.

- [ ] **Step 1: Write RED lifecycle tests**

Add:

```python
@pytest.mark.parametrize(
    ("profile", "expected"),
    [("Explore", False), ("Plan", False), ("general-purpose", True)],
)
def test_retention_is_fixed_by_profile(profile, expected):
    assert dt._should_retain_session(profile) is expected
```

Update the existing `test_run_single_child_retains_completed_general_purpose_session` to remove its `retain_session=True` and `role="leaf"` arguments; it must still assert `retention_status == "retained"` and a real `agent_id`. Add a parametrized `_run_single_child` test for Explore/Plan using the same `FakeChild` and `_parent()` helpers already defined in `test_delegate_continue.py`, asserting no `agent_id`, `retained_until`, or retained store record.

Update continuation schema test:

```python
def test_continue_schema_uses_background_boolean():
    props = DELEGATE_CONTINUE_SCHEMA["parameters"]["properties"]
    assert set(props) == {"agent_id", "prompt", "run_in_background"}
    assert "scheduling" not in props
```

Keep sensitive result tests proving Notion/Mail retained history is `HANDLE_ONLY` while the in-memory live result is not mutated.

- [ ] **Step 2: Run RED tests**

Run:

```bash
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/tools/test_delegate_continue.py \
  tests/tools/test_subagent_data_source_integration.py \
  -o 'addopts=' -p no:cacheprovider -q
```

Expected: FAIL on explicit retention/scheduling legacy behavior.

- [ ] **Step 3: Make retention profile-fixed**

Replace:

```python
def _should_retain_session(subagent_type: str) -> bool:
    return subagent_type == "general-purpose"
```

Remove per-task/top-level retention resolution. Retain only successful GP results with nonempty parent session ID and available capacity. Retention failure must continue returning `retention_status="failed"` plus `retention_error`, without fake `agent_id` or `retained_until`.

- [ ] **Step 4: Simplify continuation input**

Change schema and handler to:

```python
def delegate_continue(
    agent_id: str,
    prompt: str,
    run_in_background: Optional[bool] = None,
    *,
    parent_agent=None,
) -> str:
```

Top-level omission defaults background; nested omission defaults foreground; nested explicit true rejects before child execution. Keep foreground wait timeout transition to the same future, no duplicate queue, structured capacity rejection, claim release, and interrupt semantics.

Delete context-capsule construction/import. Rebuilt GP child obtains the real project context/workspace snapshot through Task 1’s `_build_child_agent` path.

- [ ] **Step 5: Run GREEN and race tests**

Run:

```bash
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/tools/test_delegate_continue.py \
  tests/tools/test_subagent_data_source_integration.py \
  tests/tools/test_subagent_sessions.py \
  -o 'addopts=' -p no:cacheprovider -q
```

Expected: PASS, including interrupted late-worker history protection and no-child capacity rejection.

- [ ] **Step 6: Commit**

```bash
git add tools/delegate_tool.py tools/delegate_continue_tool.py \
  tools/subagent_sessions.py tests/tools/test_delegate.py \
  tests/tools/test_delegate_continue.py \
  tests/tools/test_subagent_data_source_integration.py \
  tests/tools/test_subagent_sessions.py
git commit -m "refactor(delegation): fix profile lifecycle and continuation input"
```

---

### Task 5: 同步 docs、active skill 与 traceability

**Files:**
- Modify: `website/docs/user-guide/features/delegation.md`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/features/delegation.md`
- Modify: `website/docs/guides/delegation-patterns.md`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/delegation-patterns.md`
- Modify: `website/docs/user-guide/configuration.md`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/configuration.md`
- Modify: `website/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md`
- Modify: `.hermes/reviews/2026-07-11-claude-subagent-rollout-readiness-zh.md`
- Create: `.hermes/reviews/2026-07-11-claude-subagent-simplification-traceability.md`
- Modify outside repo after repo commit: `/Users/zongxin/.hermes/skills/autonomous-ai-agents/hermes-agent/SKILL.md`
- Test: `tests/tools/test_subagent_profiles.py`

**Interfaces:**
- Consumes: final schema/prompt/lifecycle behavior from Tasks 1–4 and the approved Batch intentional divergence.
- Produces: EN/ZH user docs, active skill readback, and a supersession/trace artifact that distinguishes Claude parity from Hermes-specific decisions.

- [ ] **Step 1: Extend docs contract tests before editing docs**

Replace old 12-field checks with assertions that all delegation docs:

```python
required_claims = (
    "description",
    "prompt",
    "run_in_background",
    "Explore",
    "Plan",
    "general-purpose",
)
stale_claims = (
    "retain_session",
    'scheduling="auto"',
    'role="orchestrator"',
    "recommended_next_step",
)
```

Also assert feature docs state:

- Batch is retained as one concurrent group/handle/consolidated completion;
- Explore/Plan are one-shot;
- GP is automatically retained;
- GP loads project context/workspace snapshot, while Explore/Plan skip project context but retain complete governance;
- nested GP delegation is runtime-derived, not caller role-selected.

Run the new docs test and expect FAIL before docs edits.

- [ ] **Step 2: Rewrite EN/ZH delegation docs**

Document the exact single/batch examples:

```python
delegate_task(
    description="inspect auth flow",
    prompt="Find the auth middleware. Return absolute paths and line ranges.",
    subagent_type="Explore",
    run_in_background=False,
)

delegate_task(
    tasks=[
        {
            "description": "inspect backend",
            "prompt": "Find the backend auth path and report evidence.",
            "subagent_type": "Explore",
        },
        {
            "description": "inspect frontend",
            "prompt": "Find the frontend auth path and report evidence.",
            "subagent_type": "Explore",
        },
    ]
)
```

State that Batch is intentional Hermes divergence for Gateway UX. Remove 12-field, role, explicit retention, auto scheduling, dynamic-limit prose, and direct-Python legacy behavior.

- [ ] **Step 3: Update configuration and bundled-skill docs**

Retain operator config for concurrency, depth, kill switch, profile wait/run timeout, and retained-store budgets. Rewrite semantics:

- top-level omitted `run_in_background` → background;
- nested omitted → foreground;
- batch shares one scheduling choice;
- GP retention automatic;
- Explore/Plan never retained;
- `max_spawn_depth=1` disables nesting; higher depth allows GP only when parent exact authority and kill switch permit.

Do not claim result fields are mandatory.

- [ ] **Step 4: Supersede old rollout artifact and create new trace**

At the top of `.hermes/reviews/2026-07-11-claude-subagent-rollout-readiness-zh.md`, add a clear supersession note pointing to the new trace. The new trace must list every approved simplification and classify Batch, full governance, Notion/Mail, no-terminal readonly profiles, process-local continuation, provider fallback, and timeouts as `INTENTIONAL` rather than falsely `EXACT` Claude parity.

Record fresh code/test/doc anchors; do not reuse old 73+18 counts as current truth without remapping.

- [ ] **Step 5: Patch and read back the active Hermes skill**

After repo docs are correct, use `skill_manage(action="patch", name="hermes-agent", ...)` to update the current profile’s `Durable & Background Systems > Delegation` paragraph. Read it back with `skill_view(name="hermes-agent")` and verify:

- Batch remains;
- no 12-field contract;
- no role/retain_session/auto scheduling;
- single/batch schema and lifecycle are accurate;
- GP project context and complete governance distinction is explicit.

- [ ] **Step 6: Run docs tests and build**

Run:

```bash
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/tools/test_subagent_profiles.py \
  -o 'addopts=' -p no:cacheprovider -q
cd website
npm ci
npm run build
```

Expected: pytest PASS; Docusaurus EN + zh-Hans production build exits 0. Existing unrelated broken-link warnings may remain, but delegation pages must emit no new warning.

- [ ] **Step 7: Commit repo artifacts**

```bash
git add website/docs/user-guide/features/delegation.md \
  website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/features/delegation.md \
  website/docs/guides/delegation-patterns.md \
  website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/delegation-patterns.md \
  website/docs/user-guide/configuration.md \
  website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/configuration.md \
  website/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md \
  website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md \
  tests/tools/test_subagent_profiles.py \
  .hermes/reviews/2026-07-11-claude-subagent-rollout-readiness-zh.md
git add -f .hermes/reviews/2026-07-11-claude-subagent-simplification-traceability.md
git commit -m "docs(delegation): document the simplified Claude-like contract"
```

---

### Task 6: 最终 verification、独立 review 和上线前收口

**Files:**
- Modify only if review finds a reproduced defect: files already in Tasks 1–5
- Evidence: `.hermes/reviews/2026-07-11-claude-subagent-simplification-traceability.md`

**Interfaces:**
- Consumes: committed Tasks 1–5.
- Produces: clean candidate branch, fresh deterministic evidence, baseline differential, independent correctness/design review, and a merge/restart decision packet. No Gateway restart.

- [ ] **Step 1: Run the high-signal delegation gate**

Run:

```bash
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/run_agent/test_tool_call_incremental_persistence.py \
  tests/run_agent/test_provider_fallback.py \
  tests/agent/test_subagent_governance.py \
  tests/agent/test_subagent_governance_preflight.py \
  tests/tools/test_delegate*.py \
  tests/tools/test_subagent*.py \
  tests/tools/test_tool_effects.py \
  tests/tools/test_registry.py \
  tests/tools/test_mcp_dynamic_discovery.py \
  tests/tools/test_refresh_agent_mcp_tools.py \
  tests/tools/test_tool_search.py \
  tests/tools/test_web_extract_robustness.py \
  tests/tools/test_skills_tool.py \
  tests/tools/test_image_generation.py \
  tests/tools/test_video_generation_dispatch.py \
  tests/run_agent/test_run_agent.py::TestConcurrentToolExecution::test_agent_runtime_post_hook_ownership_predicate_covers_agent_tools \
  -o 'addopts=' -p no:cacheprovider -q
```

Expected: PASS with zero new failure.

- [ ] **Step 2: Run static checks on baseline→HEAD Python changes**

Run:

```bash
BASE=00cd0985b6e8de1d398080755972fd293175e0e5
PY_FILES=$(git diff --diff-filter=ACMR --name-only "$BASE"...HEAD -- '*.py')
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m ruff check $PY_FILES
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m py_compile $PY_FILES
git diff --check "$BASE"...HEAD
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Run current and detached-baseline full `tests/tools`**

Create a temporary detached worktree at `00cd0985b` and run the same command in both:

```bash
git worktree add --detach /tmp/hermes-subagent-simplification-baseline 00cd0985b
set +e
/Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
  tests/tools -o 'addopts=' -p no:cacheprovider -q \
  > /tmp/hermes-subagent-simplification-current-tools.log 2>&1
CURRENT_RC=$?
(
  cd /tmp/hermes-subagent-simplification-baseline
  /Users/zongxin/.hermes/hermes-agent/.venv/bin/python -m pytest \
    tests/tools -o 'addopts=' -p no:cacheprovider -q \
    > /tmp/hermes-subagent-simplification-baseline-tools.log 2>&1
)
BASELINE_RC=$?
set -e
printf 'current_rc=%s baseline_rc=%s\n' "$CURRENT_RC" "$BASELINE_RC"
python - <<'PY'
from pathlib import Path
import re
pat = re.compile(r'^FAILED\s+(\S+)', re.M)
current = set(pat.findall(Path('/tmp/hermes-subagent-simplification-current-tools.log').read_text()))
baseline = set(pat.findall(Path('/tmp/hermes-subagent-simplification-baseline-tools.log').read_text()))
print('current_failed', len(current), 'baseline_failed', len(baseline))
print('branch_only', sorted(current - baseline))
print('baseline_only', sorted(baseline - current))
PY
```

Parse `FAILED <node-id>` sets and report exact `current - baseline` / `baseline - current`; totals alone are insufficient. Investigate every branch-only node. Remove the temporary worktree only after the comparison is recorded:

```bash
git worktree remove /tmp/hermes-subagent-simplification-baseline
```

- [ ] **Step 4: Run independent two-axis review without polling**

Load `superpowers:requesting-code-review` and `autonomous-ai-agents:codex`. Use the default Codex/Noema route, maximum two passes total:

1. correctness/security review: effect/identity, Batch concurrency/notification, retention race, provider fallback, FD/thread cleanup;
2. design-contract review: current Claude 2.1.207 schema/prompt/lifecycle, approved Batch divergence, GP project context, docs/skill truthfulness.

Start Codex with completion notification. **Do not call `poll`, `wait`, `log`, `ps`, or `lsof` while review runs.** Only if no completion arrives beyond a reasonable timeout may one diagnostic be performed. Reproduce every finding locally before changing code.

- [ ] **Step 5: Fix only reproduced blockers with RED→GREEN tests**

For each valid finding, add a deterministic failing test in the exact owning test file, run it RED, implement the root-cause fix, run focused GREEN, then rerun the high-signal gate. Commit each logical remediation with scoped staging. Do not add new brokers, action grants, persistence layers, or profile types.

- [ ] **Step 6: Final readback and candidate commit state**

Run:

```bash
git status --short --branch
git log --oneline 00cd0985b..HEAD
git diff --stat 00cd0985b...HEAD
git diff --check 00cd0985b...HEAD
```

Read back:

- `DELEGATE_TASK_SCHEMA` and `DELEGATE_CONTINUE_SCHEMA`;
- all three profile prompts;
- GP project-context/workspace inclusion tests;
- Batch handle/consolidated completion tests;
- active `hermes-agent` skill;
- final traceability artifact.

Expected: feature worktree clean; every source/doc/review change committed; no Gateway restart or live rollout performed.

- [ ] **Step 7: Present merge/restart decision packet**

Report:

- old→new schema and lifecycle;
- why Batch remains as intentional divergence;
- what was removed versus what security hardening remains;
- GP project-context parity fix;
- targeted/full/baseline/docs/static evidence;
- independent review verdict and unresolved limitations;
- exact merge, external Gateway restart, live smoke, and revert plan.

Stop and ask Zongxin before merge/restart. Do not execute rollout from this plan.
