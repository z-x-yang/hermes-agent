# Hermes Reasoning `max` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hermes 全链路接受真正的 `max` reasoning effort，并让 Desktop 将 `xhigh` 显示为 XHigh、`max` 显示为 Max；不接 `ultra`。

**Architecture:** 扩展 `hermes_constants.VALID_REASONING_EFFORTS` 作为 Core SSOT，让复用该常量的 cron/delegation 路径自动获得 `max`；逐一修正仍硬编码 effort 列表的 CLI/tool schema。Desktop 保留现有通用 effort 控件结构，但将 `xhigh` 与 `max` 拆成两个独立值和标签。Codex transport 不做映射，只以回归测试确认 `max` 原样进入 Responses payload。

**Tech Stack:** Python 3.11+、pytest、TypeScript/React、Vitest、Electron/Vite。

## Global Constraints

- `max` 是独立于 `xhigh` 的真实值，禁止 alias、降级或迁移。
- `ultra` 本次不接，仍需被 Core/CLI 拒绝。
- 现有 `xhigh` 配置保持行为不变，只改 UI 标签。
- 所有代码修改遵循 RED → GREEN → REFACTOR。
- 只提交本任务文件，不混入生成物、缓存或其他 worktree 状态。

---

### Task 1: Core parser 与 Codex payload

**Files:**
- Modify: `hermes_constants.py:794-820`
- Test: `tests/test_hermes_constants.py:427-490`
- Test: `tests/agent/transports/test_codex_transport.py`

**Interfaces:**
- Consumes: `parse_reasoning_effort(effort) -> dict | None`
- Produces: `VALID_REASONING_EFFORTS` 包含 `max`；Codex `ResponsesApiTransport.build_kwargs(...)` 对 `reasoning_config={"enabled": True, "effort": "max"}` 产生 `reasoning.effort == "max"`。

- [ ] **Step 1: 写 Core parser failing test**

在 `TestParseReasoningEffort` 增加断言：

```python
def test_max_is_a_distinct_supported_level(self):
    assert parse_reasoning_effort("max") == {"enabled": True, "effort": "max"}
    assert parse_reasoning_effort("ultra") is None
```

并把 `max` 从 unknown 参数表移除、加入 documented set。

- [ ] **Step 2: 运行 RED**

Run:

```bash
uv run pytest tests/test_hermes_constants.py::TestParseReasoningEffort -q
```

Expected: `test_max_is_a_distinct_supported_level` FAIL，因为当前 `max` 返回 `None`。

- [ ] **Step 3: 最小 Core 实现**

将：

```python
VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
```

改为：

```python
VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
```

并同步 docstring。

- [ ] **Step 4: 写并运行 Codex payload 回归测试**

在 `tests/agent/transports/test_codex_transport.py` 使用现有 `ResponsesApiTransport` 测试模式，断言：

```python
kwargs = transport.build_kwargs(
    model="gpt-5.6-sol",
    messages=[{"role": "user", "content": "hi"}],
    reasoning_config={"enabled": True, "effort": "max"},
    is_codex_backend=True,
)
assert kwargs["reasoning"]["effort"] == "max"
```

Run:

```bash
uv run pytest tests/test_hermes_constants.py tests/agent/transports/test_codex_transport.py -q
```

Expected: PASS；`ultra` 仍返回 `None`。

- [ ] **Step 5: 提交 Core change**

```bash
git add hermes_constants.py tests/test_hermes_constants.py tests/agent/transports/test_codex_transport.py
git commit -m "feat: support max reasoning effort in core"
```

---

### Task 2: CLI、cron 与 tool schema

**Files:**
- Modify: `cron/scheduler.py:223-229`
- Modify: `cron/jobs.py:913-918`
- Modify: `tools/cronjob_tools.py:1041-1044`
- Modify: `batch_runner.py:1261-1267`
- Modify: `hermes_cli/subcommands/cron.py:72-76,141-145`
- Modify: `hermes_cli/cli_commands_mixin.py:2471-2475,2534-2538`
- Test: `tests/cron/test_cron_reasoning_effort.py`
- Test: `tests/cron/test_cronjob_schema.py`
- Test: `tests/cron/test_jobs_reasoning_effort.py` or nearest existing cron jobs test module
- Test: `tests/cli/test_reasoning_command.py`

**Interfaces:**
- Consumes: `VALID_REASONING_EFFORTS` from Task 1。
- Produces: `/reasoning max`、cron config/per-job、cron tool schema、cron CLI 和 batch runner 接受 `max`；帮助/错误文案包含 `max`。

- [ ] **Step 1: 写 cron/schema failing tests**

更新/增加：

```python
assert _resolve_cron_reasoning_config({"reasoning_effort": "max"}, {}) == {
    "enabled": True,
    "effort": "max",
}
assert "max" in CRONJOB_SCHEMA["parameters"]["properties"]["reasoning_effort"]["enum"]
```

并验证 `ultra` 仍抛 `ValueError`。

- [ ] **Step 2: 运行 RED**

```bash
uv run pytest tests/cron/test_cron_reasoning_effort.py tests/cron/test_cronjob_schema.py -q
```

Expected: schema enum 和显式文案测试 FAIL。

- [ ] **Step 3: 更新所有硬编码入口**

把每处合法列表扩展为：

```text
none, minimal, low, medium, high, xhigh, max
```

`ultra` 不加入 choices/schema。

- [ ] **Step 4: 运行 GREEN**

```bash
uv run pytest tests/cron/test_cron_reasoning_effort.py tests/cron/test_cronjob_schema.py tests/cli/test_reasoning_command.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交入口一致性 change**

```bash
git add cron/scheduler.py cron/jobs.py tools/cronjob_tools.py batch_runner.py hermes_cli/subcommands/cron.py hermes_cli/cli_commands_mixin.py tests/cron tests/cli/test_reasoning_command.py
git commit -m "feat: expose max reasoning across Hermes controls"
```

---

### Task 3: Desktop XHigh/Max 拆分

**Files:**
- Modify: `apps/desktop/src/app/shell/model-edit-submenu.tsx`
- Modify: `apps/desktop/src/app/settings/model-settings.tsx`
- Modify: `apps/desktop/src/lib/model-status-label.ts`
- Modify: `apps/desktop/src/lib/model-status-label.test.ts`
- Modify: `apps/desktop/src/app/shell/model-menu-panel.test.tsx`
- Modify: `apps/desktop/src/i18n/types.ts`
- Modify: every locale module under `apps/desktop/src/i18n/*.ts` that defines `shell.modelOptions`

**Interfaces:**
- Consumes: Core accepts `max` from Tasks 1–2。
- Produces: Desktop effort controls emit literal `xhigh` or `max`；compact status labels are `XHigh` and `Max` respectively。

- [ ] **Step 1: 写 Desktop failing tests**

在 `model-status-label.test.ts` 更新为：

```ts
expect(reasoningEffortLabel('xhigh')).toBe('XHigh')
expect(reasoningEffortLabel('max')).toBe('Max')
```

在 model menu test 增加两个独立选项，并验证点击 Max 向 gateway 发送：

```ts
{ key: 'reasoning', value: 'max', session_id: expect.any(String) }
```

- [ ] **Step 2: 运行 RED**

```bash
npm run test:ui --workspace apps/desktop -- --run \
  src/lib/model-status-label.test.ts \
  src/app/shell/model-menu-panel.test.tsx
```

Expected: `xhigh` 当前返回 `Max`，且菜单没有 literal `max`，因此 FAIL。

- [ ] **Step 3: 最小 Desktop 实现**

菜单 options 改为：

```ts
{ value: 'xhigh', labelKey: 'xhigh' },
{ value: 'max', labelKey: 'max' }
```

Settings values 改为：

```ts
['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']
```

状态标签改为：

```ts
xhigh: 'XHigh',
max: 'Max'
```

i18n 类型增加 `xhigh: string`；各 locale 使用稳定产品词 `XHigh`，保留本地化 `max` 文案。

- [ ] **Step 4: 运行 GREEN + typecheck**

```bash
npm run test:ui --workspace apps/desktop -- --run \
  src/lib/model-status-label.test.ts \
  src/app/shell/model-menu-panel.test.tsx
npm run typecheck --workspace apps/desktop
```

Expected: tests 与 typecheck 都 PASS。

- [ ] **Step 5: 提交 Desktop change**

```bash
git add apps/desktop/src/app/shell/model-edit-submenu.tsx \
  apps/desktop/src/app/settings/model-settings.tsx \
  apps/desktop/src/lib/model-status-label.ts \
  apps/desktop/src/lib/model-status-label.test.ts \
  apps/desktop/src/app/shell/model-menu-panel.test.tsx \
  apps/desktop/src/i18n
git commit -m "fix: distinguish XHigh from Max in desktop"
```

---

### Task 4: 集成验证、review 与本地部署

**Files:**
- Modify: `.hermes/specs/2026-07-09-reasoning-max-design.md`
- Modify: `.hermes/plans/2026-07-09-reasoning-max.md`
- No additional production files unless verification exposes a blocker。

**Interfaces:**
- Consumes: Tasks 1–3 的 commits。
- Produces: 可审查、可构建、可在本机 Hermes Desktop/Gateway 重新加载的完整 change。

- [ ] **Step 1: 完整 targeted Python gate**

```bash
uv run pytest \
  tests/test_hermes_constants.py \
  tests/cron/test_cron_reasoning_effort.py \
  tests/cron/test_cronjob_schema.py \
  tests/cli/test_reasoning_command.py \
  tests/agent/transports/test_codex_transport.py -q
```

Expected: 0 failures。

- [ ] **Step 2: Desktop gate**

```bash
npm run test:ui --workspace apps/desktop -- --run \
  src/lib/model-status-label.test.ts \
  src/app/shell/model-menu-panel.test.tsx
npm run typecheck --workspace apps/desktop
npm run build --workspace apps/desktop
```

Expected: tests、typecheck、build 全部 exit 0；构建后检查并清理不属于任务的 generated diff。

- [ ] **Step 3: scoped Codex review**

```bash
codex review --uncommitted
```

如果 worktree 已全部 committed，则对 feature branch commit range 做 prepared scoped diff + `codex exec --sandbox read-only` review。验证每条 finding，localized finding 修复后只跑 deterministic tests，不无意义重复 review。

- [ ] **Step 4: 合入本机 main 并读回**

从 feature worktree 检查 main dirty paths 与 feature diff 不重叠，必要时 rebase，随后：

```bash
git -C /Users/zongxin/.hermes/hermes-agent merge --ff-only fix/reasoning-max
```

在 main checkout 重跑 targeted Python 与 Desktop tests。

- [ ] **Step 5: 本地运行时验证**

由于当前会话运行在 Gateway 内，不从子进程自杀式重启 Gateway。完成 source/build 后：

- 重新构建/重启 Desktop participant。
- 用外部 `/restart` 或正常 service restart 重新加载 Gateway。
- 新 session 选择 `Max`，读取 live session/status 与 request evidence，确认 effort 为 `max`。
- 切回 `XHigh`，确认状态栏显示 `XHigh` 且 payload 为 `xhigh`。

- [ ] **Step 6: 最终 scoped commit/status**

```bash
git status --short --branch
git log --oneline -5
```

Expected: feature/main 工作树没有本任务未提交文件；报告测试、build、review 和 live reload 的真实状态。
