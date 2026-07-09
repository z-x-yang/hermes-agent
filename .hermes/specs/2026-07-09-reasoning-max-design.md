# Hermes Reasoning `max` 支持设计

## 目标

让 Hermes Core、CLI、cron、gateway 和 Desktop 原生支持独立的 `max` reasoning effort；Desktop 不再把 `xhigh` 误标成 “Max”。本次明确不接 `ultra`。

## 已验证现状

- OpenAI Codex live catalog：GPT-5.6 Sol/Terra 支持 `low, medium, high, xhigh, max, ultra`；Luna 支持到 `max`。
- Hermes `VALID_REASONING_EFFORTS` 只接受到 `xhigh`，`parse_reasoning_effort("max")` 返回 `None`。
- Codex Responses transport 会把 `reasoning_config.effort` 原样放进 `reasoning.effort`，因此传输层无需为 `max` 做别名转换。
- Desktop 当前把 `{ value: "xhigh", labelKey: "max" }`，状态栏也把 `xhigh` 显示成 `Max`，语义错误。

## 设计

### 1. Core 语义

全局合法 reasoning effort 改为：

`minimal, low, medium, high, xhigh, max`

`none` 继续表示关闭 reasoning。`ultra` 继续非法并显式拒绝，不做 alias 或静默降级。

`parse_reasoning_effort("max")` 必须返回：

```python
{"enabled": True, "effort": "max"}
```

Codex Responses transport 应原样发出：

```json
{"reasoning": {"effort": "max", "summary": "auto"}}
```

### 2. 所有配置入口一致

以下入口都接受并公开 `max`：

- `agent.reasoning_effort`
- `/reasoning max`
- TUI gateway `config.set(key="reasoning", value="max")`
- cron 全局与 per-job override
- `cronjob` tool schema
- `hermes cron create/edit --reasoning-effort max`
- batch runner
- delegation/auxiliary 中复用 Core parser 的路径

所有错误文案与帮助文案同步增加 `max`，避免 schema 支持但 CLI 文案仍说不支持。

### 3. Desktop 语义

Desktop 的 effort 顺序改为：

`Minimal, Low, Medium, High, XHigh, Max`

必须满足：

- `xhigh` 写入值仍为 `xhigh`，显示为 `XHigh`。
- `max` 写入值为 `max`，显示为 `Max`。
- 状态栏 `xhigh → XHigh`，`max → Max`。
- Settings 和模型菜单使用同一语义，不再把二者合并。

本次沿用 Hermes 现有“全局 effort 列表”架构，不新增完整的 per-model effort capability 管线；后端仍负责对不支持 `max` 的模型显式报错。这样能修复 GPT-5.6，而不把本次工作扩成模型能力系统重构。

### 4. 兼容性

- 现有配置中的 `xhigh` 保持原值和实际行为，只修正 UI 标签。
- 不迁移任何 `xhigh` 到 `max`，因为两者现在是不同强度。
- 不接受 `ultra`；它包含自动任务委派语义，需独立设计。
- provider-specific adapter 若已有 `max` 映射继续保持；本次只扩展 Hermes 通用枚举。

## 验证标准

1. Core parser、cron、tool schema、CLI choices 均覆盖 `max`，仍拒绝 `ultra`。
2. Codex transport 构造出的请求保留 `max`。
3. Desktop 单测证明 `xhigh ≠ max`，状态栏和菜单标签正确。
4. Python targeted tests、Desktop Vitest、Desktop typecheck/build 通过。
5. 用本地 Hermes/OpenAI Codex 请求读取 request payload 或 live startup evidence，确认选择 `max` 后实际发送 `max`，不是 `xhigh`。
