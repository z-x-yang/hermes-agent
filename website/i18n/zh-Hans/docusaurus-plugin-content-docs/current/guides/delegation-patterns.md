---
sidebar_position: 13
title: "委派与并行工作"
description: "Explore、Plan、Reviewer、general-purpose、Batch 与 retained follow-up 的实用模式"
---

# 委派与并行工作

Hermes 通过 `delegate_task(description=..., prompt=...)` 委派隔离任务。选择最窄的内置 `subagent_type`，让 `prompt` 自包含；只有 parent 立刻依赖结果时才设 `run_in_background=False`。完整合同见[子代理委派](/user-guide/features/delegation)。

## 选择最窄的类型

| 需求 | 类型 | 生命周期 |
|---|---|---|
| 定位代码、追踪调用链、收集文件/行证据 | `Explore` | 只读，一次性 |
| 调研变更并识别关键实现文件 | `Plan` | 只读，一次性 |
| 独立检查 scoped code change 并返回 candidate blocker | `Reviewer` | repo context、普通 prompt/final、一次性 |
| 编辑、测试、terminal/process 或获准外部动作 | `general-purpose` | 成功后自动保留 |

`general-purpose` 只能获得通过 current parent 精确工具权限和运行时 policy 检查的工具，不是 unrestricted worker，也不是“无外部副作用”的 sandbox。它始终继承 parent 的 SOUL/MEMORY/USER context，包括跨 provider、endpoint、ACP transport 和 fallback route。Explore/Plan 保持 lean 并跳过自动项目上下文；Reviewer 保持 personal-context 隔离。Reviewer 和 general-purpose 加载 repo project context 与 workspace/git snapshot。Reviewer 使用普通 read/search/terminal 与按 parent authority 提供的 readonly web，不暴露 named private-source、write、process、browser 或 delegation tools。由于存在 raw terminal，no-edit/private-source 是 profile instruction 并由 controller 检查，不是机械 sandbox。

## 模式：行动前先调查

```python
delegate_task(
    description="trace token refresh",
    prompt="""Repository: /home/user/webapp.
从 src/auth/middleware.py 开始。返回 absolute path、symbol、line range
和仍未解决的 call edge。不要修改任何内容。""",
    subagent_type="Explore",
    run_in_background=False,
)
```

## 模式：实现前规划调研

```python
delegate_task(
    description="plan token rotation",
    prompt="""Repository: /home/user/webapp.
为 parent 返回调研输入：critical files、现有 tests、migration risk、security
constraint 和 open questions；不要选择最终 plan。""",
    subagent_type="Plan",
    run_in_background=False,
)
```

## 模式：一个实现 worker

```python
delegate_task(
    description="fix token reuse",
    prompt="""Repository: /home/user/webapp.
修复 refresh-token reuse detection，增加 regression tests，并运行
pytest tests/auth/test_tokens.py -q。返回改动文件和真实输出。""",
    subagent_type="general-purpose",
)
```

顶层省略 `run_in_background` 时除 Reviewer 默认前台外，其余 profile 默认后台。不要轮询后台 worker；完成后只向 owning conversation 回流一次结果。

## 模式：并行独立任务

Hermes Batch 是为 Gateway UX 有意保留的扩展，只放彼此独立的任务：

```python
delegate_task(
    tasks=[
        {
            "description": "inspect signing",
            "prompt": "Map token creation/signing and return path:line evidence.",
            "subagent_type": "Explore",
        },
        {
            "description": "inspect invalidation",
            "prompt": "Map session invalidation and return path:line evidence.",
            "subagent_type": "Explore",
        },
    ]
)
```

一个 Batch 只有一个 handle、一个 async slot 和一次 consolidated completion。每个 item 只有 `description`、`prompt` 与可选 `subagent_type`；整个 group 共用顶层 `run_in_background`。

## 模式：继续 retained GP 工作

成功的 general-purpose result 可能返回 `agent_id`：

```python
delegate_continue(
    agent_id="<returned agent_id>",
    prompt="Add the missing concurrency regression test and rerun the suite.",
    run_in_background=False,
)
```

Explore 和 Plan 是一次性任务。GP 自动保留，且 store 仅在当前进程内存在、受 budget 限制、重启即失效。只有同一 scope 的 follow-up 才用 continuation；新目标重新 `delegate_task`。

## 模式：运行时派生嵌套

不要请求 role。general-purpose child 只有在 parent 真实拥有该 exact authority、kill switch 开启且 depth 允许时才获得 `delegate_task`。嵌套省略时前台执行；嵌套 `run_in_background=True` 会在 child 启动前失败。

## 不要这样做

- 不传已删除的 `goal`、`context`、per-item scheduling 或 explicit retention fields；
- 不让 Explore/Plan 编辑或运行 shell；
- 不把互相依赖的任务放进同一 Batch；
- 不把 subagent self-report 当作 tests、文件改动或外部动作的最终证据，parent 必须验证；
- 需要跨 `/new`、shutdown 或 Gateway restart 的工作不要用 delegation，应使用 cron 或受管理进程。
