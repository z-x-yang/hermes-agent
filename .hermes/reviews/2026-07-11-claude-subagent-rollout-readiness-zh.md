# Hermes Claude-like Subagent 上线前说明

> **SUPERSEDED（2026-07-11）**：本文描述的是已被进一步简化的旧审计候选，包含 dynamic schema、统一 12 字段结果、caller role、显式 retention 与三态 scheduling 等现已删除的设计。当前候选及新鲜证据以 [`2026-07-11-claude-subagent-simplification-traceability.md`](./2026-07-11-claude-subagent-simplification-traceability.md) 为准；本文仅保留为历史审计记录，不得用于 rollout 决策。

## 结论

当前分支已完成 source、behavioral tests、stress、docs、73+18 traceability 和两轮 Codex 审查后的修复，**停在 merge main / Gateway restart 之前**。Main 尚未合并，Gateway 尚未重启，所以当前 Discord 会话仍运行旧版 subagent runtime。

这是一个可进入 controlled rollout 的本地候选，但有一条必须明说的审查 caveat：Codex pass 1/2 和 pass 2/2 都返回 `BLOCKED`；两轮指出的 concrete blockers 均已用 RED→GREEN 行为测试修复，review budget 按硬规则耗尽，因此没有第三轮独立 `PASS`。最终正确性依据是 post-fix deterministic tests、baseline differential、docs build、Ruff/compile 和逐行 traceability，不是假装存在的 reviewer PASS。

## 旧版和新版相比，真正改了什么

### 1. 从模糊 generic child 变成三种 Claude-like profile

模型侧只暴露：

- `Explore`：focused lookup / exploration，只读。
- `Plan`：planning research，只读，不实施。
- `general-purpose`：复杂多步执行，可编辑、测试、使用 terminal/process，也可使用 parent 明确拥有且工具合同允许的命名外部工具。

省略或空 `subagent_type` 一律解析为 `general-purpose`，不存在第四种 legacy capability policy。Direct-Python 的同步兼容只影响 scheduling，不形成另一套权限。

### 2. Harness 会主动教 parent 怎么路由

`delegate_task` 的 model-facing schema 动态注入：

- 三个 profile 的 name + description；
- 哪种任务对应 Explore / Plan / general-purpose；
- 当前真实 concurrency / depth 上限；
- foreground / background / auto 行为；
- retention 默认值；
- general-purpose 的 exact parent-authorized action plane；
- leaf / orchestrator 真正不能使用的 control-plane tools。

Description 只负责 routing affordance，runtime policy 才是安全边界。用户显式点名 profile 时会强制选择该路线。

### 3. Parent 会续聊同一个 child，而不是重复开新 child

成功 retained 的 general-purpose result 返回 `agent_id`。`delegate_continue` 是独立窄 schema，只接受 `agent_id / prompt / scheduling`。Tool description 明确教 parent：如果是同一工作，优先 `delegate_continue`，不要重新 spawn。

Continuation 会：

- 固定原 profile 和 canonical profile home；
- 加载最新完整 governance；
- 使用 original exact policy identities ∩ current exact parent/profile identities；
- 保留 workspace；
- 对有文件写入历史的 child 注入“workspace 可能变化，先核对当前 diff/state”的可信提示；
- 用 claim generation 阻止 interrupted worker 晚到覆盖 retained history。

### 4. 所有 provider / fallback attempt 都获得完整最新 governance

每个 child 都收到 active profile 最新、byte-preserving 的：

- `SOUL.md`
- `MEMORY.md`
- `USER.md`

不是摘要，不是旧 snapshot，不是只给官方 provider。每次 primary/fallback provider attempt 都在 backend 前对最终 payload 做 context-fit；未知或超预算 fail closed，不破坏原 provider fallback accounting。

Child 不继承 parent transcript 或旧 tool outputs；task objective、anchors、known errors、constraints、verification contract 仍要放进 task/context。

### 5. 能力不再只靠 prompt：runtime exact identity + resolved effect enforcement

新 runtime 使用：

- `ToolEffect`
- `ResultRetention`
- frozen `ToolPolicyDescriptor`
- schema/callable/argument-resolver/entry-generation identity
- exact parent authority snapshot
- frozen args + digest
- final dispatch recheck

覆盖 direct、sequential、concurrent、Tool Search bridge、MCP refresh。Dynamic-schema tools（`delegate_task`、`image_generate`、`video_generate`）现在 definition、authorization、final dispatch 共用同一 resolved schema identity；同名 replacement 不能拿旧 authorization 执行新 handler。

Role 限制按 exact tool name 强制，不依赖 composite toolset 表示：

- leaf 禁 `delegate_task / delegate_continue / clarify`；
- approved orchestrator 可保留 `delegate_task`，仍禁 `delegate_continue / clarify`；
- memory、cron、execute_code、send_message 等只在它们通过 current parent exact ceiling 和各自工具安全合同时可用。

### 6. Explore / Plan 的 Notion、Mail、web、skills 数据源

按 Zongxin 后续简化决定，没有造专用 broker：

- web / skills 使用 exact-source-derived readonly alias，`NO_SPILL`，不写 debug/cache/telemetry；
- Notion AI 只接受显式 `mode=readonly`；`write` 或缺 mode 在 backend 前拒绝；
- Apple Mail 只暴露选定 list/search/get/fetch/contact/status 等 read names；send/reply/forward/move/delete/flag/mark/save 不进 profile。

Fake-backend integration 已证明：Notion write/missing mode 和 8 类 Mail write surface 均 backend `call_tool=0`。

边界：这是“existing read tools + prompt/effect gate”的轻量方案，不声称 upstream Notion AI 具备硬 DOM no-mutation，也不声称 Mail read 绝无 seen-state incidental mutation；这是用户明确批准的 `INTENTIONAL` 范围覆盖。

### 7. 敏感正文不会原样落 retained/persistent transcript

Notion/Mail selected read tools 使用 `HANDLE_ONLY`。初始 retention、continuation update、SessionDB/JSON incremental persistence 都投影为：

- handle
- SHA-256 digest
- size
- bounded excerpt

Live in-memory result仍可供当前 child 使用，但持久/retained transcript 不保存完整敏感正文或附件。

### 8. Scheduling、并行、timeout 与 capacity

仍支持 parallel fan-out：batch 最多按 `delegation.max_concurrent_children` 并行；一个 batch 返回一个 handle 和一次 consolidated completion。

`auto`：

- 单个 Explore / Plan → foreground；
- general-purpose 和 batch → background；
- nested → foreground。

独立默认 wait/run timeout：

- Explore：900 / 1800 秒；
- Plan：1800 / 3600 秒；
- general-purpose：1800 / 7200 秒。

Foreground wait timeout 会把**同一个 future**移交 background，不能重复执行或重复投递。Background pool capacity 满时返回 structured `rejected`，不再偷偷同步执行绕过并发上限。

### 9. Result contract 统一且 profile-specific

三种 profile 都必须返回 12 个披露字段：

`outcome / evidence / actions / files_changed / tests_run / verification / blockers / open_questions / confidence / limitations / side_effects / recommended_next_step`

同时保留 profile-specific guidance：

- Explore：source/file/symbol/line evidence + searches/lookups；
- Plan：implementation-plan shape + feasibility；
- general-purpose：executed actions、real commands/outputs、external handles/status。

## 两轮 Codex 审查的真实结果

### Pass 1/2：BLOCKED

确认并修复：

- mixed composite toolset 造成 role capability leakage；
- registry final dispatch TOCTOU；
- interrupted continuation late history commit；
- capacity rejection 后偷偷同步执行；
- retention lifecycle / workspace warning / retention failure 不可见；
- result/dead metadata/docs/skill 漂移；
- H17 缺 source-specific backend-zero 行为证据。

### Pass 2/2：BLOCKED

确认并修复：

- dynamic-schema authorization 在 final dispatch 与 static identity 比较；
- canonical result contract 漏 `actions / verification / blockers / side_effects`；
- model-facing GP 文案仍误称禁 memory/send_message/execute_code；
- traceability 的 commit handles 和 D07 evidence 不够精确。

按 review budget 硬停止，没有第三轮 Codex。报告和聊天都不得把它包装成“独立审查 PASS”。

## 最终验证证据

- Post-fix high-signal gate：`720 passed in 35.50s`。
- Final full `tests/tools`：`22 failed, 7599 passed, 64 skipped`。
  - 其中 21 个 failure node-ID 与 detached baseline 完全相同；
  - 额外 atomic-snapshot concurrency node 在 current 与 baseline 都独立 `3/3` 失败，因此是已复现的 baseline issue，不是本分支新增回归。
- Docusaurus production build：en + zh-Hans 成功。
  - 全站仍有既有 broken-link / broken-anchor warnings；
  - delegation 页面没有 warning。
- Baseline→HEAD 共 49 个 changed Python files：Ruff PASS、`py_compile` PASS。
- Baseline range 与 worktree `git diff --check` PASS。
- Capability stress：100 次 parallel authorize/dispatch + same-name replacement；200 retained cycles + bounded bytes / FD probe。
- Traceability：
  - 原始 73：`64 EXACT / 9 INTENTIONAL`；
  - H01–H18：`15 EXACT / 3 INTENTIONAL`；
  - `0 PARTIAL / 0 MISSING / 0 DIVERGENT`。
- 完整 matrix：`.hermes/reviews/2026-07-11-claude-subagent-final-traceability.md`。

## 已知边界

1. Main 未合并、Gateway 未重启，live runtime 仍是旧版。
2. Notion/Mail 的轻量读约束不是专用 hard broker；H06/H07 按用户决定为 `INTENTIONAL`。
3. Trusted action-grant 子系统取消；H13 为 `INTENTIONAL`，继续复用现有 tool confirmation / terminal approval / SOUL scope。
4. Retained session 是 process-local、TTL/count/byte-bounded、restart-ephemeral；不是 durable job system。
5. Full tools suite 本身不是全绿；已做 baseline node-ID attribution，不能对外说“全仓零失败”。
6. Codex pass 2 后的修复只有 deterministic local verification，没有第三轮独立 PASS。

## 建议的 controlled rollout（尚未执行）

需要 Zongxin 明确批准后才做：

1. 在 main checkout 读回 `git status`，确认没有其他任务的未提交改动。
2. 将 audit branch losslessly squash 到 main，生成一个本地 logical commit；不要 push。
3. 从 Gateway 外部 shell 或 Discord `/restart` 重启，不能在 Gateway child terminal 中自杀式 restart。
4. 读回新 PID / start time，证明进程晚于 merge commit。
5. Live safe smoke：
   - schema 只暴露 Explore / Plan / general-purpose；
   - omit→general-purpose；
   - Explore read-only lookup；
   - GP repo-local no-op/temporary-file task；
   - retained `agent_id` → `delegate_continue`；
   - parallel batch one handle / one consolidated completion；
   - Notion/Mail 只做 fake/safe read-boundary smoke，不执行真实写入。
6. 检查 Gateway logs 无 policy identity、retention、provider fallback 新错误。

## Rollback

若 live smoke 失败：

1. 立即停止继续 spawn 新版 child；
2. 在 main `git revert <squash-commit>`，不要丢失其他 main 改动；
3. 从外部 shell 或 Discord 再次重启 Gateway；
4. 读回 PID/start time；
5. 跑旧版最小 delegation smoke，确认恢复；
6. 保留 audit branch/worktree 和失败日志用于 root-cause，不静默 fallback。
