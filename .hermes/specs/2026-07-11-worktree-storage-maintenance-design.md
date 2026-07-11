# Worktree 安全归档与 Hermes 空间哨兵设计

日期：2026-07-11
状态：待 Zongxin 审阅
范围：Hermes 本地维护脚本与 no-agent Cron；不改变 Hermes 公共 API

## 1. 结论

采用两层维护：

1. **Worktree 安全归档**：14 天无活动先提醒；30 天无活动后，仅自动移除可无损恢复的 checkout，保留未合并 branch 与 pinned HEAD。dirty、locked、仍被进程使用或状态无法确认的 worktree 永不自动删除。
2. **空间哨兵**：每周汇总 Hermes 主要磁盘占用；仅自动删除已确认结束且超过 30 天的后台进程日志。普通轮转日志、Cron 输出、session 文件、`state.db` 与 Chrome profile 只监测和报警，不自动删。

这套设计解决“僵尸 checkout 长期占空间”，但不拿未合并提交和排错历史冒险。

## 2. 现场基线

2026-07-11 只读检查：

- 当前 worktree doctor：14 天后将 clean、未合并 worktree 标为 `STALE`；`--reap` 只删除已并入 `main` 的 clean worktree 与 branch。
- 当前告警项 `fix/explicit-compress-failure-20260625`：约 16 天无提交、clean、相对 `main` ahead 41；按 30 天门槛今天不会归档。
- `~/.hermes/logs`：50.5 MiB。
- `~/.hermes/process_logs`：8.3 MiB。
- `~/.hermes/cron/output`：6.5 MiB；配置已按每 job 50 份输出保留。
- 常规日志已有轮转：`agent.log` 5 MiB × 3、`errors.log` 2 MiB × 2、`gateway.log` 5 MiB × 3；每个后台进程日志封顶 200 KiB。
- 真正的空间大头不是普通日志：`state.db` 约 5.84 GiB，`chrome-debug` 约 5.65 GiB。两者含会话历史或浏览器状态，不能按日志直接删除。

## 3. Worktree 安全归档

### 3.1 “无活动”定义

每个 linked worktree 的最近活动时间为：

- branch tip 的 commit 时间；以及
- `git status --porcelain -z` 列出的 tracked/untracked 变更路径中，最新的文件 mtime。

两者取较新值。ignored cache 不作为“工作活动”；它也不能让 dirty worktree 获得自动删除资格。所有自动删除路径（`ARCHIVABLE` 与 `MERGED`）在移除 checkout 前都另枚举 ignored 内容：只允许明确可重建的 cache（如 Python/pytest/ruff/mypy cache、venv、node_modules、egg-info、test duration cache）；`.hermes/`、`work/` 与其他未知 ignored artifact 一律 fail closed，防止 `git worktree remove` 静默删除它们；live cwd 检查同样覆盖两条路径。

如果状态解析、文件 stat 或进程占用检查失败，自动归档必须 **fail closed**：保留 worktree，并把原因放进 `skipped`。

### 3.2 分类

- `SKIP`：主 worktree、detached、`main`、Git 标记 `prunable`、checkout 已损坏，或 worktree 被 `git worktree lock` 保护。
- `ACTIVE`：最近活动不足 14 天。
- `STALE`：无活动至少 14 天但不足 30 天；只提醒。
- `STALE_DIRTY`：无活动至少 14 天且有未提交/未跟踪改动；只提醒，永不自动删除。
- `ARCHIVABLE`：无活动至少 30 天、clean、未合并、未锁定、无进程 cwd 位于该 worktree 内；可安全移除 checkout。
- `MERGED`：clean，且分支改动内容已被 `main` 吸收；沿用现有无损回收规则。

### 3.3 删除语义

- `MERGED`：维持现状，删除 checkout，并以 pinned OID 原子删除 branch。
- `ARCHIVABLE`：只执行非强制 `git worktree remove <path>`；**保留 branch 与 commit**。
- `STALE` / `STALE_DIRTY` / `ACTIVE` / `SKIP`：不删除。

归档前必须再次验证：

1. branch OID 未变化；
2. path 对应的 current worktree record 仍是原 branch 与 pinned HEAD，不能被同路径其他 checkout 替换；
3. worktree 仍 clean；
4. 最近活动仍超过 30 天；
5. 未 locked；
6. 没有进程 cwd 位于 worktree；
7. checkout 仍是有效 worktree；
8. `.hermes/`、`work/` 等 protected ignored root 必须先于 cache allowlist 判定，不能因子路径含 `node_modules`/`.venv` 等而放行。

任一条件变化即跳过，不使用 `--force`。

### 3.4 恢复记录

每次归档记录：时间、旧 path、branch、pinned HEAD、活动年龄、动作结果与恢复命令：

```bash
git worktree add <old-path> <branch>
```

记录写入有界 action manifest；最多保留最近 1000 条，采用临时文件 + 文件 `fsync` + 原子替换 + parent directory `fsync`。doctor 是 manifest 唯一写 owner：每次删除 checkout **之前**先 crash-durable 写入含 time/path/branch/HEAD/age/restore 的 `pending` 记录；动作成功后更新为 `archived`，安全门跳过时更新为 `skipped` + reason。若 post-action 写失败，`pending` recovery record 仍保留，避免 checkout 已移除却完全无恢复记录。

另用单个原子覆盖的状态文件记录已报告的 `(branch, HEAD, class)`。同一 HEAD、同一状态不重复提醒；回到 active、HEAD/dirty 状态变化、进入新阶段或被归档时更新状态。

### 3.5 Cron UX

现有每日 05:00 ET no-agent job 保持：

- 健康且无动作：stdout 空，静默。
- 首次进入 `STALE` / `STALE_DIRTY`，或 HEAD/dirty 状态变化：发精简提醒，明确“只提醒，不会自动删”。
- 同一 HEAD、同一 stale 状态连续多天：stdout 空，不重复刷屏。
- 有 `ARCHIVABLE` 被归档：报告 checkout、branch、HEAD 与恢复命令。
- race/占用/检查失败：仅在首次出现或原因变化时报告 `skipped`。
- doctor 或 JSON 解析失败：非零退出，保留 fail-fast。

## 4. Hermes 空间哨兵

### 4.1 运行方式

新增每周 no-agent Cron，建议周日 05:40 ET，投递 `#ops-alerts`。它与现有 `dev-cache-weekly-prune` 分工：后者清开发缓存；空间哨兵只负责 Hermes 自身目录。

每次测量：

- `~/.hermes/logs`
- `~/.hermes/process_logs`
- `~/.hermes/cron/output`
- `~/.hermes/sessions`
- `state.db` + WAL/SHM
- `~/.hermes/chrome-debug`
- 文件系统可用空间

状态写入单个原子覆盖的 JSON 文件，用于判断阈值首次越界和周增长；不追加无限 JSONL。

### 4.2 唯一自动清理项

只删除 `~/.hermes/process_logs/proc_*` 中同时满足以下条件的目录：

- 真实目录且 resolve 后仍位于 `process_logs` 下；
- 存在完成 sidecar `exit_code`；
- 目录内最新 mtime 超过 30 天；
- 不在当前 `processes.json` 的活跃集合中。

缺 sidecar、仍活跃、路径异常、symlink 或检查失败都保留并报警。删除必须通过已打开的 `process_logs`/candidate directory fd 锚定，candidate 在 `stat → open` 后立即比对 inode，`exit_code` 必须是 non-symlink regular file；candidate 内容只能通过 pinned candidate fd 清理，顶层只能在名称仍指向原 inode 时用非递归 `rmdir` 删除。ancestor/path replacement 后只能作用于原 pinned root，同名 real-directory replacement 也必须保留，不得跟随新路径或递归删除 replacement。单次回收至少 50 MiB 才主动通知；警告无论大小都通知，但以 warning SHA-256 signature 去重，同一问题持续存在时保持静默，恢复后再次出现才重报。单次最多递送 20 条新 warning；超出的只标记 deferred，后续运行继续分批递送，不能把未展示 warning 的 hash 提前记成已报告。

### 4.3 报警条件

以下任一项触发精简报告：

- 可用磁盘低于 50 GiB 或低于总容量 10%；
- 普通 logs 超过 250 MiB；
- process logs 清理后仍超过 250 MiB；
- Cron output 超过 250 MiB；
- sessions 目录超过 1 GiB；
- `state.db` 超过 5 GiB；
- `chrome-debug` 超过 5 GiB；
- 任一受监测项单周增长至少 1 GiB，或在增长至少 250 MiB 的前提下增长 25%。

阈值首次越界时通知一次；之后只有继续显著增长、恢复后再次越界或出现新错误才再次通知，避免每周重复刷屏。

### 4.4 明确不做

空间哨兵不得：

- 删除/截断当前普通日志或 compression audit sidecar；
- 绕过已有日志轮转；
- 删除 Cron 输出（已有每 job 50 份保留）；
- 删除 sessions、运行 `hermes sessions prune`、`VACUUM` 或改 retention；
- 清除 Chrome profile、cookies、IndexedDB、Service Worker 或 extensions；
- 删除 worktree、branch、项目、模型、数据集、环境或备份。

`state.db` 与 Chrome profile 的治理另开只读审计与用户确认，不能夹带进“日志清理”。

## 5. 文件与配置边界

预计改动：

- repo：`scripts/worktree_doctor.py`
- repo：`tests/scripts/test_worktree_doctor.py`
- repo：新增空间哨兵的可测试逻辑与测试
- live wrapper：`~/.hermes/scripts/worktree_doctor_cron.py`
- live wrapper/script：`~/.hermes/scripts/hermes_storage_sentinel.py`
- live Cron：更新现有 worktree job；新增 weekly storage sentinel job

不修改 Gateway scheduler 核心，不需要新增 model/tool schema，不需要 LLM Cron。

## 6. 测试契约

严格 TDD，至少覆盖：

### Worktree doctor

- clean、未合并、29 天：只提醒，不归档；
- clean、未合并、31 天：移除 checkout，但 branch 与 pinned HEAD 保留；
- dirty、31 天：不归档；
- 最近未提交改动 mtime 会刷新活动时间；
- ignored unknown artifact 阻止归档，明确 allowlist 的可重建 cache 可随 checkout 删除；
- locked worktree 不归档；
- 有进程 cwd 位于 worktree 时不归档；
- 进程检查失败时 fail closed；
- 分类后 branch OID 变化时跳过；
- 原有 merged squash-safe 回收、prunable skip 回归测试继续通过；
- action manifest 有界且恢复命令正确；
- 同一 HEAD/同一 stale 状态不重复提醒，状态变化会重新提醒；
- Cron wrapper stdout/stderr/no-agent 语义正确。

### 空间哨兵

- 仅删除超过 30 天且有 `exit_code` 的 finished process log；
- 新目录、活跃 PID、缺 sidecar、symlink、越界路径不删；
- 阈值首次越界、持续越界去重、显著增长、恢复后再次越界行为正确；
- 状态文件原子更新；
- dry-run 不产生删除；
- 空 stdout 表示健康/无动作。

## 7. 上线与验收

1. 在独立 worktree 完成 RED → GREEN → REFACTOR。
2. 跑 focused tests、脚本语法检查、`git diff --check`。
3. 以 dry-run 对真实 worktree 与目录执行；证明今天不会归档任何 30 天候选，也不会删除 process logs。
4. scoped commit，squash 到 active `main`。
5. 更新 live wrappers 与 Cron，读回 job ID、schedule、script、deliver、`no_agent=true`。
6. 手动运行两个脚本；验证 worktree job 的真实分类，以及空间哨兵只报告当前已知大项、不做 destructive cleanup。
7. 观察一次 Cron 输出/状态，确认 `last_status=ok`；无需 Gateway 源码重启，因为只改脚本与 job 配置。

## 8. 完成标准

- 30 天无活动的 clean、未合并 worktree checkout 会被安全归档，branch/HEAD 可恢复。
- dirty、locked、活跃或不确定 worktree 不会自动删除。
- 已合并 worktree 继续无损回收。
- 仅 finished process logs 按 30 天保留自动清理。
- 普通日志、Cron 输出、sessions、`state.db`、Chrome profile 不被误删。
- 告警只在有动作、错误、首次越界或显著增长时出现。
- 代码、测试、live script、Cron readback 与真实 dry-run 均有可核验证据。
