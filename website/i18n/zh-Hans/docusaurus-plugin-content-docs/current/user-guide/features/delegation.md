---
sidebar_position: 7
title: "子代理委派"
description: "Claude 风格的子代理类型、Batch 投递、续聊与运行时安全"
---

# 子代理委派

Hermes 通过 `delegate_task` 启动隔离的 child agent。模型可见合同刻意保持很小：`description`、`prompt`、可选 `subagent_type`、可选 `run_in_background`、仅限单个 Reviewer 的 `review_root`，以及 Hermes 有意保留的 Batch 扩展。工具、嵌套、保留、超时和 provider fallback 都由运行时策略控制，不由调用者传权限字段决定。

## 内置 subagent 类型

`subagent_type` 只接受四种值：

| 类型 | 适合任务 | 生命周期与上下文 |
|---|---|---|
| `Explore` | 只读代码、文件和资料调查 | 一次性；lean Core Contract + task capsule；不注入个人与项目上下文 |
| `Plan` | 只读实现调研和规划 | 一次性；Core Contract + controller 明确选择的 task/project summary；不注入完整个人 governance |
| `Reviewer` | 对 scoped code change 做 fresh-context 独立审查 | 一次性；默认前台；repo context + 普通自包含 prompt；普通 final response |
| `general-purpose` | 多步骤执行、编辑、测试和获准的外部动作 | 成功完成后自动保留；Core Contract 加项目上下文和 workspace snapshot |

与 Claude Code 一样，每个 canonical profile 自己持有 routing `description`，告诉 parent 何时选择它。`Reviewer` 与其他 profile 使用同一个普通 `prompt` 字段，没有隐藏 JSON grammar 或 profile-specific completion tool。`review_root` 是 Hermes 保留的顶层适配：当前 workspace 不是目标时，用它绑定一个本机 worktree。

省略 `subagent_type` 时解析为 `general-purpose`。child 不继承 parent transcript、parent tool result，也不自动注入 active profile 完整的 `SOUL.md`、`MEMORY.md`、`USER.md`。`general-purpose` 和 `Reviewer` 会从各自绑定的 root 加载 repo rules 与 workspace snapshot；这种规范性项目上下文与个人记忆、实现 session rationale 分开。

`Explore` 和 `Plan` 使用运行时强制的只读工具上限：可使用获准的 repo/file reader、no-spill web/skill reader、显式 `mode=readonly` 的 Notion AI，以及选定的 Apple Mail 读取工具；不给 raw terminal 和文件写入。`Reviewer` 必须有 `read_file`、`search_files`、`terminal`；只有 parent 当前拥有 web authority 时才附加 readonly/no-spill web。它不暴露 named Notion、Mail、session/memory、MCP、browser、write/patch、process/execute-code 或 delegation tools。Raw terminal 是刻意的 Claude-like 选择，所以 Reviewer **不是**机械 no-write/no-external-side-effect sandbox：system contract 禁止编辑、安装、改 Git、发布和访问私有来源，controller 必须在审查后检查 worktree。`general-purpose` 同样只获得通过 current parent 权限和正常安全合同筛选的工具。

所有 profile 都使用 prompt-defined return contract 与普通 final response。Reviewer 只应报告 newly introduced、evidence-backed 的 Critical/Important candidate blocker，附 file/line 证据和具体 failure path；简短 no-findings final 也算有效完成。Parent/controller 必须逐条复现或证伪。

## 单任务

```python
delegate_task(
    description="inspect auth flow",
    prompt="Find the auth middleware. Return absolute paths and line ranges.",
    subagent_type="Explore",
    run_in_background=False,
)
```

单任务必须同时提供 `description` 和 `prompt`：

- `description` 是简短进度标签；
- `prompt` 是自包含任务，写清路径、约束、证据与交付要求；
- 顶层省略 `run_in_background` 时采用所选 profile 默认值：`Reviewer` 默认前台，其余 profile 默认后台；显式布尔值始终优先。

### Reviewer

Reviewer 接受普通自包含任务，与 Claude Code custom agent 一样：

```python
delegate_task(
    description="independent auth review",
    prompt="""审查 HEAD 已提交的 auth-race 修复。
检查 src/auth 和 tests/auth 是否满足：同一个 token 不得并发 refresh 两次，
且 public API 不变。Fresh verification：`pytest tests/auth -q` 返回 `18 passed`。

只报告 newly introduced Critical/Important candidate blocker，附 file:line
证据和具体 failure path。不要编辑 checkout。""",
    subagent_type="Reviewer",
    review_root="/absolute/path/to/local/worktree",  # 可选
)
```

`review_root` 是 controller 绑定的 tool argument，不是 prompt syntax。它只允许用于顶层单个 Reviewer，且必须精确解析为现存的本机 absolute Git worktree root；relative path、repo 子目录、Batch/nested、非 Reviewer 和 remote/cluster root 都 fail closed。省略时审查当前 workspace。Runtime 会绑定 child workspace 并加载该 root 的 repo guidance；需要审查的 diff/range/files 直接用普通 prose 写进 prompt。

## Batch API：Hermes 的有意差异

Claude Code 用同一条 assistant message 内的多个 Agent call 表达并行。Hermes 保留 Batch API，因为 Gateway 和聊天平台更适合一个分组生命周期：

```python
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

一个 Batch 是一个并发组：一个 batch handle、占用一个 async slot，并在全部 child 结束后发出一次合并完成通知。结果按 task index 排序。Batch item 只有 `description`、`prompt` 和可选 `subagent_type`；整个 Batch 共用顶层 `run_in_background`。

活跃 child runner 同时受两层原子 ceiling 约束：`max_concurrent_children` 限制一个 root session（包括嵌套后代与 continuation），同时也限制单个 Batch width；`max_global_concurrent_children` 限制整个 Hermes 进程。默认每个 root session 最多 5 个、整个进程最多 20 个。一次 reservation 若会超过任一 ceiling，会在任何 child 启动前整批拒绝。

后台池满时，Hermes 返回结构化 `rejected`，不会偷偷同步执行 child。若 endpoint 无法稍后投递，已准备好的工作会同步执行，并带显式说明。

## 前台、后台与超时

调度只使用 `run_in_background`：

- 顶层省略 → 所选 profile 默认值（`Reviewer` 前台，其余后台）；
- 顶层 `False` → 前台等待；
- 嵌套省略或 `False` → 前台；
- 嵌套 `True` → child 执行前 fail closed。

前台等待和 child 运行使用独立的 operator timeout。默认 wait/run 为 Explore `900/1800` 秒、Plan `1800/3600`、Reviewer `1800/3600`、general-purpose `1800/7200`。Reviewer 仍使用 operator 配置的 `delegation.max_iterations`，没有隐藏的更低轮次上限。profile run limit 作用于每个 child，包括直接在后台启动的工作。前台等待超时时，Hermes 把同一个 future 转到后台，之后只投递一次完成结果；不会重排或重启 child。

## 上下文隔离

child 从全新 conversation state 开始，所以 `prompt` 必须自包含：

```python
delegate_task(
    description="fix body parser",
    prompt="""Repository: /home/user/webapp.
修复 api/handlers.py 的 TypeError：缺少 Content-Type 时 parse_body()
返回 None。增加回归测试并运行 pytest tests/api/，返回改动文件和真实测试输出。""",
    subagent_type="general-purpose",
    run_in_background=False,
)
```

`general-purpose` 和 `Reviewer` 会加载真实 repo rules（按正常发现规则选择 `.hermes.md`、`AGENTS.md`、`CLAUDE.md` 或 `.cursorrules`）以及 workspace/git snapshot。`Explore` 和 `Plan` 跳过自动 project context。所有 profile 都不注入完整个人 governance；Reviewer 也不会得到 parent transcript/tool output、Notion Project Memory、session history 或 personal memory。

## 保留会话与 `delegate_continue`

生命周期按 profile 固定：

- `Explore`、`Plan` 和 `Reviewer` 是一次性任务，永不保留；
- 成功的 `general-purpose` 在 parent session ID 非空且容量允许时自动保留；
- 保留失败会显式返回 `retention_status="failed"` 和 `retention_error`，不会伪造 `agent_id`。

拿到 retained result 的 `agent_id` 后可续聊：

```python
delegate_continue(
    agent_id="<completed result 中的 agent_id>",
    prompt="Now add the missing regression test and rerun the focused suite.",
    run_in_background=False,
)
```

`delegate_continue` 只接受 `agent_id`、`prompt` 和可选 `run_in_background`。它固定原 profile/workspace，并取原始权限与当前权限的精确交集。store 只在当前进程内存在，受 TTL、数量和 transcript bytes 限制，重启即失效。Notion 与 Apple Mail 敏感读取结果在 retained history 中继续使用 `HANDLE_ONLY`。claim generation/cancellation 会阻止 interrupt 或 timeout 后的 late worker 写回旧 history。

## 运行时派生的嵌套委派

嵌套资格是运行时派生，不由调用者选 role。child 只有同时满足以下条件才获得 `delegate_task`：

1. profile 是 `general-purpose`；
2. current parent 精确工具权限中真实存在 `delegate_task`；
3. `delegation.orchestrator_enabled` 为 true；
4. `child_depth < max_spawn_depth`。

`Explore`、`Plan` 和 `Reviewer` 永远不能继续委派；`delegate_continue` 与 `clarify` 也不提供给 child。默认 `max_spawn_depth=2`，允许一层有界的 `general-purpose` orchestrator（`parent → child → grandchild`）；depth-2 child 是 leaf，且 nesting 仍须满足上述所有 gates。

## 中断与持久性

`/agents`（别名 `/tasks`）显示 active/recent subagent。`/stop` 和 shutdown 会向前台、后台 child 与 continuation 传播中断。后台 delegation 和 retained session 都不是 durable job；需要跨 `/new`、进程退出或 Gateway 重启的任务应使用 cron 或受管理的后台进程。

## 配置

并发、depth、kill switch、各 profile 的 model/provider 与 wait/run timeout、retained store 的 TTL/count/byte budget 都在 `~/.hermes/config.yaml` 的 `delegation` 下配置。这些是 operator controls，不是 model-facing fields。参见[配置 → Delegation](/user-guide/configuration#delegation)。
