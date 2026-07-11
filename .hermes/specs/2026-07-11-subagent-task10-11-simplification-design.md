# Hermes Subagent Task 10/11 简化设计

> 日期：2026-07-11
> 状态：Zongxin 已批准方向，待书面复核
> 作用：覆盖原 Data-Source Parity amendment 中 Task 10 与 Task 11 的实现范围。

## 1. 结论

取消独立的 `trusted action grant` 子系统，不新增 `action_grants.py`，也不把 gateway `/approve`、CLI approval、外部工具确认流程改造成统一 capability token 系统。

Task 11 仅保留两个可复用现有机制完成的薄修复：

1. continuation 绑定同一 canonical profile/home，并把原始 exact tool policy identities 与 current parent/latest policy 再取交集；
2. Notion/Apple Mail 读取结果使用现有 `HANDLE_ONLY` retention 投影，避免完整正文进入 retained transcript 和普通持久化。

## 2. 为什么取消 Task 10

现有 Hermes 已有三层约束：

- `general-purpose` 继承完整 `SOUL.md`、`MEMORY.md`、`USER.md` 与 trusted parent task scope；
- 各工具已有自身的发送、删除、外部写入确认合同；
- terminal dangerous commands 已有 session-scoped gateway/CLI approval。

原 Task 10 要再建立 action/tool/target/critical args/TTL/one-use/replay 的通用 grant store，并改 gateway、CLI、dispatcher 与测试。这会扩大安全关键代码面，同时与现有 approval 和工具合同重叠。按 Zongxin 的最新决定，外部动作继续服从现有 Evelyn/SOUL/工具合同，不声称新增 runtime grant 证明。

最终 traceability 中，H13 与原 grant acceptance rows 标记为 `INTENTIONAL`，理由是明确的用户范围覆盖，而不是伪报 `EXACT`。

## 3. Continuation 薄绑定

### 3.1 Retained metadata

`RetainedSubagentSession` 只新增最小非秘密 metadata：

- `profile_id`
- `canonical_profile_home`
- `original_policy_identities`
- `original_governance_fingerprint`（仅审计，不冻结旧正文）

现有 `subagent_type`、`workspace_path`、`parent_session_id`、`effective_allowed_tool_names` 保留。

### 3.2 Continuation gate

每次 continuation：

1. 加载当前一致性 governance snapshot；
2. current `profile_id` 或 canonical profile home 与 retained record 不同则 fail closed；
3. 使用最新 governance 正文重建 child，不回放旧 governance；
4. 最终 authority 为：

```text
retained original exact policy identities
∩ current parent exact policy identities
∩ latest current-profile policy identities
```

5. 最终 names 仍与 retained original names 和 current available names 取交集；
6. workspace 继续使用既有 strict absolute-directory 验证；
7. 不新增跨进程 durability，retained session 继续保持 process-local、TTL/数量/字节有界。

任何 metadata 缺失、profile 漂移或交集为空都结构化拒绝，不退回 name-only policy。

## 4. Notion/Mail transcript 投影

不新增 broker、存储层或专用正文数据库。

对已批准暴露给 Explore/Plan 的现有 Notion AI readonly call 与 Apple Mail read tool names，将 descriptor retention 设为 `HANDLE_ONLY`。复用 Task 6 已实现的统一路径：

- child live result 只获得有界 head/tail 内容；
- hook/observer/session DB/JSON snapshot/retained transcript 只保留 handle、SHA-256、size 与有限 excerpt；
- 超长正文不做 generic disk spill；
- continuation 需要更多内容时重新读取数据源。

不承诺 Apple Mail `PEEK`、server version pin 或 Notion REST hard broker；这些已被用户明确取消。显式 send/reply/forward/move/delete/flag/mark 工具仍不进入 Explore/Plan profile。

## 5. 非目标

- 不新增第四种 subagent profile；
- 不新增通用 action grant/token service；
- 不修改现有 gateway `/approve` 或 terminal dangerous-command approval；
- 不持久化 retained session 到重启后；
- 不恢复 Notion REST broker、Apple Mail broker、PEEK/version pin；
- 不触碰 live gateway 或真实 Notion/Mail 写入。

## 6. 验证合同

使用 fake/local tests 证明：

1. 同 parent session、同 profile/home、未变 authority 的 continuation 成功；
2. 换 profile/home 后 backend invocation 为零；
3. same-name/new-policy-identity replacement 不能借 continuation 扩权；
4. current parent 移除工具后 continuation 自动降权；
5. continuation 读取最新 governance canary，而不是旧正文；
6. Notion/Mail read descriptor 为 `HANDLE_ONLY`；
7. retained transcript、SessionDB 与 JSON snapshot 不出现测试正文 canary，只出现 handle/digest/excerpt；
8. 原有 provider fallback、foreground/background、race/FD、web/skills read adapters 不回归。

完成后进入 Task 12：对原始 73 条和 H01–H18 分开 adjudicate；被用户明确取消的 H06/H07/H13 等要求标记 `INTENTIONAL` 并引用范围变更，不得隐藏差异。
