---
sidebar_position: 7
title: "子智能体委派"
description: "内置子智能体类型、调度、续接与能力边界"
---

# 子智能体委派

Hermes 通过 `delegate_task` 运行隔离的子智能体。每个子智能体都有独立的对话与终端状态，会收到父智能体传入的任务/上下文，以及 active profile 最新完整的 `SOUL.md`、`MEMORY.md`、`USER.md` 快照，最后返回结构化结果。调用方不能随意指定子智能体工具：Hermes 先应用父智能体当前的 exact resolved authority，再强制套用所选类型的能力上限。

## 内置子智能体类型

`subagent_type` 只接受以下三个内置值：

| 类型 | 适用工作 | 能力上限 | 单任务使用 `scheduling="auto"` 时 |
|---|---|---|---|
| `Explore` | 搜索并理解代码、文件和辅助资料 | 只读文件工具、no-spill web/skill readers、显式 `mode=readonly` 的 Notion AI，以及现有 Apple Mail list/search/get/fetch 工具 | 前台 |
| `Plan` | 调研代码库，为后续实施计划准备输入 | 与 `Explore` 相同的 read-oriented 上限；不能编辑，也不能声称实现已完成 | 前台 |
| `general-purpose` | 编辑、测试及父级允许的多步骤执行 | 与父智能体 exact resolved tool identities 取交集；内部 policy 的 `None` 表示 parent-exact intersection，不表示无限制全局工具 | 后台 |

`Explore` 和 `Plan` 不能写文件、运行 shell 或委派。raw `web_search`、`web_extract`、`vision_analyze` 不可用，web/skills 走专用 no-write alias。Notion/Mail 不再新造 broker：只暴露 `notion_ai_ask` 和选定 Mail 读取工具，要求 Notion `mode=readonly`，并隐藏 send/reply/forward/move/delete/flag/mark。这里是轻量 read-oriented 合同，不声称上游绝无诸如 Mail seen 状态之类的 incidental mutation。`general-purpose` 可使用通过 parent exact intersection 和工具安全合同的 action-plane 工具，默认仍是 leaf；只有显式要求且深度允许时才保留 `delegate_task`。

三种 profile 都必须返回同一套完整披露字段：`outcome`、`evidence`、`actions`、`files_changed`、`tests_run`、`verification`、`blockers`、`open_questions`、`confidence`、`limitations`、`side_effects`、`recommended_next_step`。不适用的字段填空或 `none`，不能省略。

省略或传空 `subagent_type` 一律解析为 `general-purpose`，不存在第四种 legacy capability policy。调度兼容与能力解析分开：

- **模型发起**且 `scheduling="auto"` 的省略调用采用 `general-purpose` 后台默认值。
- 直接 Python API 的省略调用在没有显式调度/后台请求时仍同步执行，但能力仍是 `general-purpose`。

不要用直接 Python API 的兼容规则推断模型侧的调度行为。

## Parent 如何学会路由和续聊

Hermes 通过 tool contract 教 parent model，而不是依赖隐藏 heuristic：

1. model-facing `delegate_task` schema 只暴露三种 type，并把每个 profile 当前的 `description` 同时注入单任务与 batch 字段。
2. 顶层 tool description 给出具体 routing cue、用户当前 concurrency/depth 上限、调度行为与 retention 默认值。
3. 成功保留的结果会返回 `agent_id`；独立的 `delegate_continue` schema 明确说明：下一条指令若延续同一工作，应使用该 ID，而不是重新 spawn child。
4. 这些 description 只负责 routing affordance，不是安全边界。Runtime 会独立重新解析 profile、与 parent exact authority 取交集，并验证每个 authorized tool call。

这与 Claude Code 的核心模式一致：agent name/description 教 parent 何时委派，返回的 agent ID 让续聊可发现，runtime permission 才是最终权威。用户也可以直接点名 type，或要求继续某个返回的 `agent_id`，强制选择路线。

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

- **`auto`**：单个 `Explore` 或 `Plan` 任务在前台运行；`general-purpose` 和多任务批次在后台运行。direct-Python 省略类型且未指定调度的调用保留同步兼容路径。
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

如果当前端点无法投递后续消息（例如无状态 HTTP 请求），Hermes 会同步执行已准备好的任务。如果后台池已满，Hermes 会返回结构化的 `rejected` 结果且不启动 child；调用方可以重试，不会静默越过并发上限。

## 上下文隔离

新子智能体不会继承父智能体的对话记录或历史 tool outputs，但会继承 active profile 的完整 canonical governance；父级上下文不足时，可按权限主动查询 Notion、Mail、repo/files、skills 和 web。`goal` 与 `context` 仍应写清 scoped objective、workspace/repo anchor、已知错误、约束、验证合同和输出语言。第三方内容始终只是 untrusted data，不能提升权限或充当用户授权。

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

- 存储仅存在于当前进程，同时受 TTL、记录数和序列化 transcript 字节预算限制（默认 `3600` 秒、`64` 条、`16777216` 字节）。
- 单条初始记录超过字节预算时不会保留；聚合裁剪只删除未在续接中的记录，绝不驱逐已 claim 的记录。
- 如果一次成功续接使 transcript 超过预算，Hermes 仍返回成功结果并附上 `retention_dropped`，同时使该 handle 失效，后续续接会稳定失败。
- 只有同一个且非空的父会话才能续接对应 `agent_id`。
- 续接必须保持原 canonical profile/home；exact authority 为原始 policy identities、当前 parent 与最新 profile policy 的交集，同名工具替换不能借续接扩权。
- 每次续接重新读取最新 canonical governance；原 fingerprint 仅用于审计，不冻结旧正文。
- 选定的 Notion/Apple Mail 读取结果在 SessionDB、JSON 与 retained transcript 前复用 `HANDLE_ONLY` 投影，只保留 digest、size 与有限 excerpt，不保留完整正文。
- 同一个 `agent_id` 同时只能有一个续接任务；第二个并发请求会立即失败。不同的保留子智能体可以并发续接。
- `/stop` 和进程关闭会中断后台续接。
- Gateway 或进程重启会丢失全部保留会话；这不是持久化存储。
- 记录中不保存凭据或自定义 `base_url`。续接时会从当前可信配置重新解析凭据，因此配置变化后不能保证精确复现旧的自定义端点。

## 嵌套编排

启用嵌套委派后，只有显式指定 `role="orchestrator"` 的 `general-purpose` 具备编排资格。`Explore` 和 `Plan` 会拒绝该角色。

- `role="leaf"` 是默认值，包括 `general-purpose` 在内都不能继续委派。
- 只有 `delegation.orchestrator_enabled` 为 true 且 `max_spawn_depth` 允许下一层时，`role="orchestrator"` 才会保留委派能力。
- 最终有效的编排者只会在父级精确工具上限和 profile 上限之外额外获得 `delegate_task`；`delegate_continue`、MCP 和其他名称仍被排除。
- `max_spawn_depth` 默认 `1`，表示扁平委派；最小值为 `1`，没有硬性上限。每增加一层都可能成倍增加费用和并发量。

嵌套工作始终同步前台执行，子智能体不能把任务从拥有它的父会话中脱离出去。

## 中断、监控与持久性

TUI 中可用 `/agents`（别名 `/tasks`）查看正在运行和最近完成的子智能体。`/stop` 与进程关闭会把中断传播到前台、后台子智能体以及后台续接。

后台委派是异步执行，但不是持久任务存储。通过 `/new` 关闭所属会话、停止进程或重启 Gateway，都可能丢弃正在运行的工作和保留记录。必须跨越 agent/Gateway 生命周期的任务，应使用 cron 或独立管理的后台进程。

## 配置

调度时限、各类型模型/provider 覆盖、保留 TTL/记录数/字节容量、并发和嵌套深度都在 `~/.hermes/config.yaml` 的 `delegation` 下配置。这些由用户掌控的设置不会暴露在模型可见的工具 schema 中。详见[配置 → 委派](/user-guide/configuration#委托)。
