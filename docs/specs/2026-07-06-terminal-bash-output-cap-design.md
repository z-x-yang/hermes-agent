# Terminal/Bash 大输出阈值对齐 Claude Code 设计

## 背景

Claude Code 官方文档明确：Bash 输出默认超过 `30,000` characters 时，完整输出保存到 session 目录，模型上下文只收到路径和短 preview；`BASH_MAX_OUTPUT_LENGTH` 可提高到 `150,000` characters。

Hermes 当前 live 行为：`terminal` 注册的 `max_result_size_chars` 是 `100,000`，超过后走现有 `tools/tool_result_storage.py` 的 per-result persistence：完整输出写到 `/tmp/hermes-results/<tool_call_id>.txt`，上下文里放 `<persisted-output>` + `1,500` chars preview + 路径。单 turn aggregate budget 仍是 `200,000` chars。

Zongxin 已批准只对齐第一项：降低 Bash/terminal 大输出进入模型前的阈值。**不做** durable session storage 重构，不做 generic budgeter，不改 auto-compact。

## 目标

把 Hermes `terminal` tool 的 model-visible 大输出阈值从 `100,000` chars 降到 `30,000` chars，对齐 Claude Code Bash 默认值，减少测试日志/命令输出对上下文和 prompt cache 的污染。

## 非目标

- 不修改 `read_file`：继续用自身 `80,000` chars guard、分页和 dedup。
- 不修改 `execute_code`：继续用当前 stdout `50KB` head/tail cap。
- 不修改 `web_extract` / browser / MCP 动态工具。
- 不把 `/tmp/hermes-results` 改成长期 session archive。
- 不新增 LLM 摘要、复杂 preview 策略、或自创统一 budgeter。
- 不修改 compression / cheap cleanup / auto-compact 阈值。

## 设计

### 1. 阈值

在 `tools/terminal_tool.py` 的 `registry.register(name="terminal", ...)` 中设置：

```python
max_result_size_chars=30_000
```

当前通用 persistence 逻辑保持不变：

```text
terminal output <= 30K chars: 原样进入上下文
terminal output > 30K chars: full output 写入 /tmp/hermes-results，模型看到 persisted-output + preview + path
```

### 2. Preview 大小

把通用 persisted-output preview 从 `1,500` chars 调到 `2,000` chars：

```python
DEFAULT_PREVIEW_SIZE_CHARS = 2_000
```

理由：Claude Code 本机 `2.1.195` binary 在 persisted-output 附近能看到 `2000` 常量；公开 issue 也出现过 `Preview (first 2KB)`。这不是官方稳定 API，但证据比其它预算参数更硬，且每个 persisted output 只多 `500` chars，上下文成本很小。

其它预算不动：`DEFAULT_RESULT_SIZE_CHARS=100_000`、`DEFAULT_TURN_BUDGET_CHARS=200_000`、small-model scaling、`read_file` guard、`execute_code` head/tail cap 都保持当前行为。

### 3. 磁盘策略

这次不做 durable storage。原因：长期保存 full terminal logs 会引入硬盘增长、secret 生命周期和清理策略问题。

继续使用现有 `/tmp/hermes-results` 临时保存：

- 优点：不长期占硬盘；机制已有测试覆盖；改动窄。
- 缺点：reboot / temp cleanup 后可能不可恢复；这是本轮接受的 trade-off。

新增磁盘量只来自 `30K < output <= 100K` 的 terminal 输出；超过 `100K` 的输出当前本来也会落盘。

### 4. 测试

新增/更新测试覆盖：

- registry 层：`terminal` 的 `max_result_size_chars` 应为 `30_000`。
- budget config：`DEFAULT_PREVIEW_SIZE_CHARS` 应为 `2_000`；其它默认 budget 不变。
- persistence 集成：terminal 输出超过 `30_000` chars 时会产生 `<persisted-output>` 或 fallback truncation，不会完整进入消息上下文。
- 不影响 `read_file` / `search_files` / generic result/turn budget defaults。

### 5. 验证

实现后运行：

```bash
uv run python -m pytest tests/tools/test_tool_result_storage.py -q -o 'addopts='
uv run python -m pytest tests/run_agent/test_run_agent.py::<relevant-test> -q -o 'addopts='
```

如果测试通过，再 squash 到 live main，重启 gateway，跑一个 local smoke：让 `terminal` 输出约 `40K` chars，确认模型上下文里是 `<persisted-output>`，不是完整输出。

## 风险与缓解

- **风险：模型需要完整中段输出但只看到 preview。** 缓解：persisted-output 给 `read_file` 路径，可按需读取。
- **风险：更多 `/tmp/hermes-results` 文件。** 缓解：仅新增 30K–100K 区间的临时文件，不做长期保存；当前 `/tmp/hermes-results` 为空。
- **风险：preview first-only 可能错过失败尾部。** 本轮不改，因为 Claude Code Bash 文档也是 short preview from start；如需改 preview 策略，必须另起证据调研。

## Done Contract

- `terminal` output persistence threshold 为 `30,000` chars。
- persisted-output preview size 为 `2,000` chars，其它 result/turn budget 不变。
- 相关测试通过。
- live gateway 加载新代码。
- smoke 证明约 `40K` terminal 输出不再完整进入模型上下文。
