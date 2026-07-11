---
sidebar_position: 7
title: "子代理委派"
description: "Claude 风格的子代理类型、Batch 投递、续聊与运行时安全"
---

# 子代理委派

Hermes 通过 `delegate_task` 启动隔离的 child agent。模型可见合同刻意保持很小：`description`、`prompt`、可选 `subagent_type`、可选 `run_in_background`，以及 Hermes 有意保留的 Batch 扩展。工具、嵌套、保留、超时和 provider fallback 都由运行时策略控制，不由调用者传权限字段决定。

## 内置 subagent 类型

`subagent_type` 只接受三种值：

| 类型 | 适合任务 | 生命周期与上下文 |
|---|---|---|
| `Explore` | 只读代码、文件和资料调查 | 一次性（one-shot）；完整 governance；跳过项目上下文与 workspace snapshot |
| `Plan` | 只读实现调研和规划 | 一次性（one-shot）；完整 governance；跳过项目上下文与 workspace snapshot |
| `general-purpose` | 多步骤执行、编辑、测试和获准的外部动作 | 成功完成后自动保留；完整 governance 加项目上下文和 workspace snapshot |

省略 `subagent_type` 时解析为 `general-purpose`。所有 profile 都获得当前 active profile 完整的 `SOUL.md`、`MEMORY.md` 和 `USER.md`，但不会继承 parent transcript 或旧 tool result。

`Explore` 和 `Plan` 使用运行时强制的只读工具上限：可使用获准的 repo/file reader、no-spill web/skill reader、显式 `mode=readonly` 的 Notion AI，以及选定的 Apple Mail 读取工具；不给 raw terminal 和文件写入。`general-purpose` 只能获得通过 current parent 精确工具权限与正常安全合同共同筛选的工具；它不是“无外部副作用”的 sandbox。

Hermes 不再要求所有 profile 填同一套通用结果字段。`Explore` 清楚简洁地报告发现，`Plan` 以 `### Critical Files for Implementation` 结尾；若 parent 需要固定输出，应在 `prompt` 中明确。重要改动和事实仍由 parent 验证。

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
- 顶层调用省略 `run_in_background` 时默认后台运行；只有 parent 立即依赖结果时才设为 `False`。

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

后台池满时，Hermes 返回结构化 `rejected`，不会偷偷同步执行 child。若 endpoint 无法稍后投递，已准备好的工作会同步执行，并带显式说明。

## 前台、后台与超时

调度只使用 `run_in_background`：

- 顶层省略 → 后台；
- 顶层 `False` → 前台等待；
- 嵌套省略或 `False` → 前台；
- 嵌套 `True` → child 执行前 fail closed。

前台等待和 child 运行使用独立的 operator timeout。默认 wait/run 为 Explore `900/1800` 秒、Plan `1800/3600`、general-purpose `1800/7200`。等待超时时，Hermes 把同一个 future 转到后台，之后只投递一次完成结果；不会重排或重启 child。纯后台工作不套用 profile 的 foreground run cap。

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

`general-purpose` 会加载真实 repo rules（按正常发现规则选择 `.hermes.md`、`AGENTS.md`、`CLAUDE.md` 或 `.cursorrules`）以及 workspace/git snapshot。`Explore` 和 `Plan` 刻意跳过项目上下文，但保留完整 governance。

## 保留会话与 `delegate_continue`

生命周期按 profile 固定：

- `Explore` 和 `Plan` 是一次性任务，永不保留；
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

`Explore` 和 `Plan` 永远不能继续委派；`delegate_continue` 与 `clarify` 也不提供给 child。默认 `max_spawn_depth=1`，因此默认仍是 flat delegation；调高后也只在上述 gates 全部满足时允许下一层 GP。

## 中断与持久性

`/agents`（别名 `/tasks`）显示 active/recent subagent。`/stop` 和 shutdown 会向前台、后台 child 与 continuation 传播中断。后台 delegation 和 retained session 都不是 durable job；需要跨 `/new`、进程退出或 Gateway 重启的任务应使用 cron 或受管理的后台进程。

## 配置

并发、depth、kill switch、各 profile 的 model/provider 与 wait/run timeout、retained store 的 TTL/count/byte budget 都在 `~/.hermes/config.yaml` 的 `delegation` 下配置。这些是 operator controls，不是 model-facing fields。参见[配置 → Delegation](/user-guide/configuration#delegation)。
