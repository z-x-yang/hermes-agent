---
sidebar_position: 7
title: "子智能体委派"
description: "内置子智能体类型、调度、续接与能力边界"
---

# 子智能体委派

Hermes 通过 `delegate_task` 运行隔离的子智能体。每个子智能体都有独立的对话与终端状态，只会收到父智能体传入的任务和上下文，最后返回结构化结果。调用方不能随意指定子智能体工具：Hermes 会先考虑父智能体当前可用的工具，再强制套用所选子智能体类型的能力上限。

## 内置子智能体类型

`subagent_type` 只接受以下三个内置值：

| 类型 | 适用工作 | 能力上限 | 单任务使用 `scheduling="auto"` 时 |
|---|---|---|---|
| `Explore` | 搜索并理解代码、文件和辅助资料 | 只读：`read_file`、`search_files`、`web_search`、`web_extract` | 前台 |
| `Plan` | 调研代码库，为后续实施计划准备输入 | 与 `Explore` 相同的只读上限；不能编辑，也不能声称实现已完成 | 前台 |
| `general-purpose` | 需要编辑、测试的多步骤仓库内工作 | 使用封闭的仓库内工具白名单，可操作文件、shell/进程、任务、技能和视觉工具；不能产生外部副作用或继续委派 | 后台 |

`Explore` 和 `Plan` 不能写文件、运行 shell、产生外部副作用或委派。`general-purpose` 可以在仓库内编辑和测试，但默认不能发消息、创建定时任务、写共享记忆、执行外部副作用或继续委派。

省略 `subagent_type` 会保留旧版通用委派行为。这里有两个刻意不同的兼容入口：

- **模型发起**且使用旧版 `scheduling="auto"` 的调用会在后台运行。
- 直接调用 Python API 时，如果既未指定子智能体类型，也没有显式指定调度或后台运行，则仍保持同步执行。

不要用直接 Python API 的兼容规则推断模型侧的调度行为。

## 单任务与批次

进行一次只读调查：

```python
delegate_task(
    goal="Locate the authentication retry logic and explain its call path",
    context="Repository root: /home/user/webapp. Include file:line evidence.",
    subagent_type="Explore",
)
```

执行一个仓库内实现任务：

```python
delegate_task(
    goal="Fix the failing authentication retry tests",
    context="Repository root: /home/user/webapp. Run pytest tests/auth/.",
    subagent_type="general-purpose",
)
```

并行批次：

```python
delegate_task(tasks=[
    {
        "goal": "Map the token refresh path",
        "context": "Repository root: /home/user/webapp.",
        "subagent_type": "Explore",
    },
    {
        "goal": "Map session invalidation",
        "context": "Repository root: /home/user/webapp.",
        "subagent_type": "Explore",
    },
])
```

模型可见的 schema 不包含 `toolsets`、模型/provider 选择、迭代预算或超时控制。这些都由用户配置和运行策略决定。即使旧客户端仍传入已移除字段，也不能借此放宽子智能体的能力上限。

## 调度

`scheduling` 可取 `auto`、`foreground` 或 `background`。

- **`auto`**：单个 `Explore` 或 `Plan` 任务在前台运行；`general-purpose`、旧版通用调用以及多任务批次在后台运行。
- **`foreground`**：Hermes 等待到已解析的前台等待时限。
- **`background`**：立即返回一个 handle，完成后再把结果注入原来的会话。
- **嵌套/编排任务**：始终同步前台执行；显式要求嵌套后台运行会直接失败，不会悄悄放行。

前台等待和子智能体执行使用两个不同的时限：

1. `foreground_wait_timeout_seconds` 决定父智能体等待多久。
2. `child_run_timeout_seconds` 限制**从前台启动**的子智能体最多运行多久，值来自全局或对应类型的配置。

如果先到达前台等待时限，Hermes 会把**同一个仍在运行的 future**交给后台投递，不会重启子智能体。调用方先收到 `backgrounded_after_foreground_timeout`，等该 future 完成后只会再收到一次结果。

纯后台启动的任务保持既有行为：profile 的 `child_run_timeout_seconds` 不会变成覆盖所有后台任务的统一超时。如果用户另行配置了旧版 `delegation.child_timeout_seconds` 硬上限，它仍会独立生效。

### 批次结果统一投递

一个批次就是一个异步单元：

- 只返回一个 `delegation_id`；
- 只占用一个后台槽位；
- 所有子任务完成后只投递一次汇总结果；
- 结果仍按任务索引排序。

Hermes 不会为批次中的每个任务分别返回 handle，也不会分别注入完成消息。

如果当前端点无法投递后续消息（例如无状态 HTTP 请求），或者后台池已经满了，Hermes 会同步执行已准备好的任务，并在结果中解释原因，而不是返回一个永远无法完成投递的 handle。

## 上下文隔离

新子智能体不会继承父智能体的对话记录。请在 `goal` 和 `context` 中写全仓库根目录、相关文件、错误信息、约束、验证命令以及输出语言。

```python
# 过于含糊
delegate_task(goal="Fix the error", subagent_type="general-purpose")

# 信息完整
delegate_task(
    goal="Fix the TypeError in api/handlers.py",
    context="""Repository: /home/user/webapp.
process_request() receives None from parse_body() when Content-Type is missing.
Add a regression test and run pytest tests/api/.""",
    subagent_type="general-purpose",
)
```

子智能体摘要只是自我报告。父智能体应当重新核验关键文件改动、测试结果和外部事实，再把它们当作已确认结论交给用户。

## 保留会话与 `delegate_continue`

`delegate_task` 可以暂时保留已完成子智能体的对话，供后续继续：

- `general-purpose` 默认只在任务**成功完成**后保留，而且父会话 ID 必须非空，存储容量也必须足够。
- `Explore` 和 `Plan` 默认一次性使用；如需续接，显式设置 `retain_session=true`。
- 设置 `retain_session=false` 可禁止本次保留。
- 无状态请求或父会话 ID 为空时，不会返回可续接的 `agent_id`。

保留成功的结果中会包含 `agent_id`。继续同一个子智能体：

```python
delegate_continue(
    agent_id="<agent_id from the completed result>",
    prompt="Now add the missing regression test and rerun the focused suite.",
    scheduling="auto",
)
```

`delegate_continue` 只接受 `agent_id`、`prompt` 和 `scheduling`。续接时会保留原来的子智能体类型、角色、工作区提示、模型/provider 元数据和能力上限，不能借续接修改工具、类型、角色、保留策略或超时。

保留机制的安全边界与生命周期：

- 存储仅存在于当前进程，受 TTL 和容量限制（默认 `3600` 秒、`64` 条）。
- 只有同一个且非空的父会话才能续接对应 `agent_id`。
- 同一个 `agent_id` 同时只能有一个续接任务；第二个并发请求会立即失败。不同的保留子智能体可以并发续接。
- `/stop` 和进程关闭会中断后台续接。
- Gateway 或进程重启会丢失全部保留会话；这不是持久化存储。
- 记录中不保存凭据或自定义 `base_url`。续接时会从当前可信配置重新解析凭据，因此配置变化后不能保证精确复现旧的自定义端点。

## 嵌套编排

启用嵌套委派后，旧版通用子智能体可以使用 `role="orchestrator"`。三个内置 profile（`Explore`、`Plan`、`general-purpose`）都不允许使用编排者角色。

- `role="leaf"` 是默认值，不能继续委派。
- 只有 `delegation.orchestrator_enabled` 为 true 且 `max_spawn_depth` 允许下一层时，`role="orchestrator"` 才会保留委派能力。
- `max_spawn_depth` 默认 `1`，表示扁平委派；最小值为 `1`，没有硬性上限。每增加一层都可能成倍增加费用和并发量。

嵌套工作始终同步前台执行，子智能体不能把任务从拥有它的父会话中脱离出去。

## 中断、监控与持久性

TUI 中可用 `/agents`（别名 `/tasks`）查看正在运行和最近完成的子智能体。`/stop` 与进程关闭会把中断传播到前台、后台子智能体以及后台续接。

后台委派是异步执行，但不是持久任务存储。通过 `/new` 关闭所属会话、停止进程或重启 Gateway，都可能丢弃正在运行的工作和保留记录。必须跨越 agent/Gateway 生命周期的任务，应使用 cron 或独立管理的后台进程。

## 配置

调度时限、各类型模型/provider 覆盖、保留 TTL/容量、并发和嵌套深度都在 `~/.hermes/config.yaml` 的 `delegation` 下配置。这些由用户掌控的设置不会暴露在模型可见的工具 schema 中。详见[配置 → 委派](/user-guide/configuration#委托)。
