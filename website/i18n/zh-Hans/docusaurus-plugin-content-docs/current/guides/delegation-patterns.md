---
sidebar_position: 13
title: "委派与并行工作"
description: "Explore、Plan、general-purpose、批次与保留续接的实用模式"
---

# 委派与并行工作

Hermes 可以把边界清晰的工作交给隔离的子智能体。选择内置类型可获得稳定的能力上限；把任务上下文写完整后，再由调度器决定父智能体是否需要等待。

完整契约请参阅[子智能体委派](/user-guide/features/delegation)。

## 选择能力最窄的内置类型

| 需求 | 类型 | 原因 |
|---|---|---|
| 定位代码、追踪调用路径、收集 file:line 证据 | `Explore` | 只读，默认前台运行 |
| 在编写实施计划前调研改动范围 | `Plan` | 只读，按计划调研格式返回，默认前台运行 |
| 编辑仓库文件、运行测试或完成多步骤仓库内任务 | `general-purpose` | 使用封闭的仓库内工作策略，默认后台运行 |

不要让 `Explore` 或 `Plan` 编辑文件。`general-purpose` 会排除命名的消息、发布、调度、Notion、cron 和记忆写入工具，模型也不能通过 `toolsets` 放宽该白名单。但它仍有原始 `terminal` 和 `process`，shell 命令可以访问外部系统；应以正常 terminal 审批和明确任务指令作为治理边界，而不是把它当作硬性的无副作用沙箱。

## 模式：行动前先做定向调查

需要证据、又不希望产生改动时，使用 `Explore`：

```python
delegate_task(
    goal="Trace how expired access tokens trigger refresh",
    context="""Repository: /home/user/webapp.
Start at src/auth/middleware.py. Return file:symbol:line evidence,
what you searched, and any unresolved call edges.""",
    subagent_type="Explore",
)
```

单个 `Explore` 在 `auto` 下前台运行。通常结果会直接返回；如果达到已配置的等待时限，同一个子智能体会转交后台继续执行，稍后只投递一次结果。

## 模式：只调研计划输入，不实施

需要为后续计划收集资料时，使用 `Plan`：

```python
delegate_task(
    goal="Research what must change to add rotating refresh tokens",
    context="""Repository: /home/user/webapp.
Identify critical files, existing tests, migration risks, security constraints,
and open questions. Do not edit files.""",
    subagent_type="Plan",
)
```

`Plan` 不能写文件，也不能运行 shell。它的输出供父智能体制定计划使用，不能视为已经完成实现的证据。

## 模式：单个后台实现任务

边界清晰、可以独立推进的仓库内任务适合 `general-purpose`：

```python
delegate_task(
    goal="Fix refresh-token reuse detection and add regression tests",
    context="""Repository: /home/user/webapp.
Relevant files: src/auth/tokens.py and tests/auth/test_tokens.py.
Run: pytest tests/auth/test_tokens.py -q.
Return changed files and exact test output.""",
    subagent_type="general-purpose",
)
```

在 `auto` 下，Hermes 会立即返回后台 handle。父智能体应继续其他工作，不要轮询；完成结果稍后会注入所属会话。后台委派无法跨越 `/new`、`/stop`、进程关闭或重启持久存在。

## 模式：并行只读调研

互不依赖的只读问题很适合组成批次：

```python
delegate_task(tasks=[
    {
        "goal": "Map token creation and signing",
        "context": "Repository: /home/user/webapp. Return file:line evidence.",
        "subagent_type": "Explore",
    },
    {
        "goal": "Map token validation and revocation",
        "context": "Repository: /home/user/webapp. Return file:line evidence.",
        "subagent_type": "Explore",
    },
    {
        "goal": "Map authentication test coverage gaps",
        "context": "Repository: /home/user/webapp. Read only; do not modify tests.",
        "subagent_type": "Explore",
    },
])
```

多任务批次在 `auto` 下后台运行。整个 fan-out 只返回**一个 handle**，等所有子任务完成后只产生**一次汇总结果**；不会为每个任务分别生成 handle 或完成消息。

如果任务数超过 `delegation.max_concurrent_children`，请拆成多个批次。Hermes 会拒绝过大的批次，不会静默截断。

## 模式：并行编辑时明确文件所有权

多个 `general-purpose` 子智能体可能操作同一个工作树，因此只有文件范围互不重叠时才适合并行：

```python
delegate_task(tasks=[
    {
        "goal": "Update server token responses",
        "context": "Repository: /home/user/webapp. Own only src/api/tokens.py and tests/api/test_tokens.py.",
        "subagent_type": "general-purpose",
    },
    {
        "goal": "Update Python SDK token parsing",
        "context": "Repository: /home/user/webapp. Own only sdk/python/ and its tests.",
        "subagent_type": "general-purpose",
    },
])
```

如果多个子智能体可能修改同一个文件、运行破坏性仓库命令，或依赖彼此尚未提交的输出，就不要并行。最终应由父智能体整合并验证完整 diff。

## 模式：保留一条实现线程继续追问

`general-purpose` 成功完成后，只要父会话 ID 非空且容量足够，默认会保留其会话。使用返回的 `agent_id` 继续紧密相关的工作：

```python
delegate_continue(
    agent_id="<agent_id>",
    prompt="Address the remaining edge case from the failed parametrized test.",
    scheduling="auto",
)
```

如果需要继续 `Explore` 或 `Plan`，初次调用时设置 `retain_session=true`。保留记录只存在于当前进程，受 TTL 和容量限制，而且只能由同一个父会话使用。同一个 `agent_id` 不能同时运行两个续接；重启后记录会丢失。

续接会保留原来的类型、角色、工作区提示、模型/provider 元数据和能力上限。不能借此把 `Explore` 提升为编辑器，也不能修改工具、超时或所属父会话。

## 模式：确有必要时再用嵌套编排

嵌套委派只适用于旧版通用编排者，不适用于三个内置 profile：

```python
delegate_task(
    goal="Survey three migration approaches and synthesize a recommendation",
    context="Repository: /home/user/webapp.",
    role="orchestrator",
)
```

这要求 `delegation.max_spawn_depth >= 2` 且 `delegation.orchestrator_enabled: true`。嵌套工作同步前台执行；显式要求嵌套后台运行会直接失败。每增加一层都可能成倍增加费用，因此在子任务已经明确时，应优先使用顶层批次。

## 调度检查表

- 单个 `Explore`/`Plan` + `auto` → 前台。
- 单个 `general-purpose` + `auto` → 后台。
- 模型发起的旧版通用调用 + `auto` → 后台。
- 多任务批次 + `auto` → 后台，一个 handle、一次结果。
- 嵌套/编排委派 → 同步前台。
- 直接 Python 旧版调用，且未指定类型、调度或后台请求 → 走同步兼容路径。
- 前台等待超时 → 同一个 future 转交后台，之后只投递一次完成结果。
- 前台启动的任务使用配置的 child run cap；纯后台任务不会自动套用这项 profile 超时。

## 上下文与验证检查表

委派前请写明：

- 仓库或工作区路径；
- 精确的文件、symbol、错误信息或搜索目标；
- 允许的范围，以及子智能体拥有的文件；
- 测试或验证命令；
- 输出语言和证据格式。

完成后：

- 查看真实 diff 或文件内容；
- 由父智能体重新运行关键测试；
- 把摘要视为自我报告，而不是独立证据；
- 记住一个批次完成消息中可能包含多个子任务结果；
- 只有在原能力上限仍然合适时才使用 `delegate_continue`。

## 不适合委派的情况

- 只需一次工具调用：直接调用工具。
- 不需要复杂推理的机械式 API/工具流水线：使用 `execute_code`。
- 需要用户澄清：子智能体不能使用 `clarify`。
- 需要外部副作用：由获得明确授权的父智能体工具执行并核验结果。
- 必须跨越 Gateway 生命周期长期运行：使用 cron 或独立管理的后台进程。
