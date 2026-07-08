# Compression Append-Cached Summary Call + Audit Spec

**日期：** 2026-07-08
**状态：** Draft for Zongxin review
**范围：** Hermes Agent context compression summary-generation path；不实现代码，只定义设计与审查标准。
**推荐方案：** 新增 `compression.summary_call_mode: append_cached`，让压缩摘要生成请求复用当前主会话的 provider-visible prefix，并把压缩指令作为最后一条 user message 追加；保留现有 summary 规则、tail policy、in-place/rotation 持久化语义。

---

## 1. 背景与问题

当前 Hermes 压缩系统的摘要生成路径在 `agent/context_compressor.py::_generate_summary()` 中：

1. `compress()` 先选定 `summarize_start` / `compress_end` / retained tail。
2. `_generate_summary()` 调用 `_serialize_for_summary(turns_to_summarize)`，把要压缩的历史窗口序列化成一大段文本。
3. 再把现有九段 summary 规则、minimal sufficient state 规则、All User Messages 规则、temporal anchoring 等内容拼成一个巨大 prompt。
4. 通过 `call_llm(..., messages=[{"role": "user", "content": prompt}])` 调 auxiliary compression LLM。

这个模式质量稳定，但有一个性能问题：**summary 生成请求的 provider-visible prefix 和主会话请求完全不同**。对 prompt cache 来说，它不是“当前会话 + 最后一条压缩指令”，而是一条全新的 user prompt，所以会打破缓存命中。

Claude Code 官方文档描述的 `/compact` 行为更 cache-friendly：生成 summary 的一次性请求使用同样的 system prompt、tools、history，再追加一个 final user summarization instruction。这样 summary request 自身能复用已有会话 prefix；不过 `/compact` 之后下一轮仍会因为消息历史被 summary 替换而重建较短的 conversation cache。

**本 Spec 解决的不是“压缩后永远不 cache miss”，而是：压缩摘要生成那次不要白白 miss 已经 warmed 的 prefix。**

---

## 2. 目标 / Done Contract

实现完成后应满足：

1. **不改变现有压缩摘要规则的语义。** 九段 summary 结构、minimal sufficient state、All User Messages、secret redaction、focus topic、temporal anchoring、summary prefix/end marker、retained tail policy 都保留。
2. **summary 生成请求可选择 cache-friendly append 模式。** 在兼容 provider/model/cache key 的情况下，summary request 的前缀应与当前主会话请求的 provider-visible prefix 对齐，只追加一条压缩指令。
3. **272K threshold 不误当 1M 模型上限。** Hermes 的 `compression.threshold_tokens` 是触发压缩的工作窗口，不是 summary request 的真实 API ceiling；summary request fitting 必须按实际 summary runtime/provider/model context limit 判断。
4. **可审查。** 每次压缩 audit 能回答：用了哪种 summary call mode、是否预期能命中 cache、实际 cache read/write tokens、summary 规则版本、source window、tail、fallback 原因、摘要质量抽查句柄。
5. **可抽查质量。** 能从安全 sidecar 拿到 summary 文本/结构质量信号、真实用户消息 ground truth、section coverage、retained tail 行号/DB row ids；不需要把完整 tool output 或 secrets 写进 audit log。
6. **失败 fail-closed。** 新模式不支持、上下文超限、返回 tool call、cache 统计缺失、summary validation 失败时，要显式记录 fallback/abort 原因；不得静默降级成“看似成功但不可审计”的压缩。

---

## 3. 非目标

- 不在本次设计里重写 summary prompt 质量规则。
- 不把 retained tail 改成可压缩；protected tail 的 raw provider-visible evidence 仍不可后置缩短。
- 不默认启用 Claude-like old tool cleanup 的新行为；已有 `cheap_tool_result_cleanup` 保持现状。
- 不依赖 provider 自动 dereference `<persisted-output>` 或 archived DB handles。
- 不把压缩健康报告默认发到日常聊天；如果后续做周期性/主动审查推送，默认目标应是 Zongxin 偏好的 `#ops-alerts`。

---

## 4. 当前代码锚点

主要实现相关文件：

- `agent/context_compressor.py`
  - `ContextCompressor.compress()`：选择 `compress_start` / `compress_end` / retained tail，调用 `_generate_summary()`，组装 compacted transcript，写 `compression_audit.jsonl`。
  - `ContextCompressor._generate_summary()`：当前 serialized-prompt summary call 的核心。
  - `_build_compression_audit_record()` / `write_compression_persist_audit()` / `_write_user_message_ground_truth_audit()`：当前 metadata audit 与 user-message sidecar。
  - `_summary_source_token_budget()`：当前 source budget 用 `min(context_length, threshold_tokens)` 约束 serialized source；新模式需要分离 “触发 threshold” 与 “summary request runtime limit”。
- `agent/conversation_compression.py`
  - `compress_context()`：压缩锁、in-place/rotation 持久化、post-persist audit row ids、repeated compression warning。
- `agent/chat_completion_helpers.py`
  - `build_api_kwargs()`：主会话 provider-visible request kwargs 构造入口；新模式应复用它，避免重新实现 provider quirks。
  - `call_chat_completion_non_streaming()` / streaming 路径：已有 cache stats 提取与使用统计处理可复用或扩展。
- `agent/transports/chat_completions.py`
  - `ChatCompletionsTransport.build_kwargs()`：OpenAI-compatible request shape。
  - `extract_cache_stats()`：当前从 `prompt_tokens_details.cached_tokens` / `cache_write_tokens` 提取 cache stats。
- `agent/anthropic_adapter.py`
  - Anthropic Messages path 的 cache-control、usage fields 与 system/messages conversion。
- Tests：`tests/agent/test_context_compressor.py`、`tests/run_agent/test_*compress*.py`、`tests/gateway/test_compress_command.py` 等。

---

## 5. 方案选择

### 方案 A：继续 serialized prompt，只加 audit

**做法：** 不改 summary request shape，只记录 cache miss 与质量抽查字段。
**优点：** 风险最小。
**缺点：** 不解决核心 cache miss；长会话压缩时仍重复 prefill。
**结论：** 不推荐，只适合作为 fallback。

### 方案 B：append_cached 到 `compress_end`（推荐）

**做法：** summary request 使用当前 provider-visible 消息前缀到 `compress_end`，最后追加一条 compression instruction。retained tail 不进入 summary request，因为它会原样保留在 compacted transcript 中。

**优点：**
- 与当前 Hermes 语义最一致：summary 只吸收将被压掉的窗口。
- 不把 retained tail 重复写进 summary，降低 stale-current-work 风险。
- 能复用主会话从 system/tools/history 到 `compress_end` 的 prefix cache。
- 容易做 deterministic tests：检查 summary request messages prefix 与主会话 provider payload prefix 一致。

**缺点：**
- 如果当前主会话最后几条消息在 retained tail 中，summary request prefix 只复用到 `compress_end`；这仍是大量 cache hit，但不是完整当前请求 prefix。
- 需要在 instruction 里明确“总结上方到此为止的 compacted prefix/window，retained tail 会另行原样保留”。

**结论：** 推荐 v1 实现。

### 方案 C：append_cached full-history + instruction

**做法：** summary request 使用完整当前 provider-visible history，再追加 instruction；instruction 指定只总结将被压缩的 window，tail 会保留。

**优点：** 最接近 Claude Code 文档描述，理论上复用完整当前会话 prefix。
**缺点：** summary 模型会看到 retained tail，容易把 tail 也纳入 summary；还需要处理 full-history + output budget 的更大 context pressure。
**结论：** v1 不做；等方案 B 稳定后再评估。

---

## 6. 推荐架构

### 6.1 配置

新增 typed config：

```yaml
compression:
  summary_call_mode: serialized_prompt  # 默认保持旧行为；可设 append_cached
  append_cached_summary:
    source_scope: compacted_prefix       # v1 唯一支持：messages up to compress_end
    require_main_runtime: true           # 默认 true：只有 main runtime 才声称 cache-friendly
    allow_tool_choice_none: true         # 尝试禁止 summary request 调工具
    fallback_to_serialized_prompt: true  # 新模式不支持时显式 fallback
    audit_sample_summary_chars: 12000    # 仅写入安全 sidecar，不进 content-free audit
```

默认保持 `serialized_prompt`，避免直接改变生产行为。Zongxin profile 可以后续单独开启。

### 6.2 Summary rules 抽离

把 `_generate_summary()` 里的规则构建拆成两个概念：

1. `build_summary_rules(...) -> SummaryRules`
   - 负责九段 template、minimal sufficient state、All User Messages、temporal anchoring、focus topic。
   - 输出不包含 serialized conversation source。
2. `build_serialized_summary_prompt(rules, content_to_summarize, previous_summary)`
   - 保持旧模式。
3. `build_append_cached_summary_instruction(rules, previous_summary, window, tail_policy)`
   - 新模式：只把规则和边界说明作为 final user instruction。
   - 不再内联 `{content_to_summarize}`，避免历史重复。

`SummaryRules` 需要带 `rules_hash`，用于 audit 和回归测试：只要规则文本变了，hash 就变；如果只是 transport shape 变，hash 应保持稳定。

### 6.3 Summary request builder

新增内部接口：

```python
@dataclass
class SummaryCallResult:
    summary_text: str | None
    mode: str
    source_binding: str
    request_tokens_estimate: int | None
    runtime_context_limit: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cache_eligible: bool
    fallback_reason: str | None
    tool_call_violation: bool
```

新路径：

```python
def _generate_summary_append_cached(
    self,
    messages: list[dict[str, Any]],
    *,
    summarize_start: int,
    compress_end: int,
    previous_summary: str | None,
    focus_topic: str | None,
    agent_runtime: SummaryRuntime,
) -> SummaryCallResult:
    # 返回统一的 summary call 结果；具体实现由 implementation plan 展开。
    return SummaryCallResult(
        summary_text=None,
        mode="append_cached",
        source_binding="provider_payload_prefix_to_compress_end",
        request_tokens_estimate=None,
        runtime_context_limit=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cache_eligible=True,
        fallback_reason=None,
        tool_call_violation=False,
    )
```

`ContextCompressor` 当前没有完整 agent 指针。实现时有两种可选方式：

- 推荐：`conversation_compression.compress_context()` 在调用 `compress()` 前，把一个轻量 `summary_runtime` 注入 compressor，包含 `build_api_kwargs`、main runtime、tools、system prompt、request client call、cache stats extractor。
- 备选：把 append-cached summary call 放在 `conversation_compression.py` 层执行，再把 summary text 传回 compressor assembly；但这会让 `compress()` 的职责边界变复杂。

推荐轻量 runtime 注入，避免 `ContextCompressor` 直接依赖完整 `AIAgent`。

### 6.4 Provider-visible prefix

v1 `append_cached` summary messages 取：

```python
summary_source_messages = provider_payload_messages[:compress_end]
summary_request_messages = summary_source_messages + [
    {"role": "user", "content": append_cached_instruction}
]
```

边界要求：

- `provider_payload_messages` 必须经过与主会话相同的 transport message conversion / media stripping / metadata stripping。
- 不能使用 raw DB rows 直接构造。
- 如果第 `compress_end` 附近会拆开 assistant tool_call / role=tool result group，必须沿用现有 boundary alignment，不能发非法 tool history。
- `tool_choice: "none"` 只作为请求参数尝试，不得删除 `tools` array；删除 tools 会改变 cached prefix。

### 6.5 Context limit 与 272K/1M 分离

新增概念：

- `compression_trigger_threshold_tokens`：当前 `threshold_tokens`，决定何时压缩。
- `summary_runtime_context_limit_tokens`：实际 summary call provider/model 的 context limit。
- `summary_request_tokens_estimate`：`summary_request_messages + tools + system + output_budget` 的 rough estimate。

判断规则：

```text
如果 summary_request_tokens_estimate + requested_summary_output_tokens <= summary_runtime_context_limit_tokens：允许 append_cached。
否则：记录 fallback_reason="append_cached_context_overflow"，转 serialized_prompt 或 abort，按现有策略处理。
```

不要再用 `min(context_length, threshold_tokens)` 作为 append-cached summary source 的上限。那是旧 serialized source budget 的概念。

---

## 7. Audit / Debug / 抽查设计

### 7.1 Content-free audit：`logs/compression_audit.jsonl`

在现有 `event="context_compression"` 记录中新增：

```json
"summary_call": {
  "mode": "append_cached",
  "source_binding": "provider_payload_prefix_to_compress_end",
  "rules_hash": "sha256:example-rules-hash",
  "cache_eligible": true,
  "cache_key_runtime": {
    "provider": "openai-codex",
    "model": "gpt-5.5",
    "api_mode": "codex_responses",
    "reasoning_effort": "medium",
    "tools_included": true,
    "tool_choice_none_requested": true
  },
  "request": {
    "message_count": 123,
    "prefix_message_count": 122,
    "instruction_chars": 18400,
    "tokens_estimate": 271000,
    "runtime_context_limit_tokens": 1000000,
    "requested_output_tokens": 12000
  },
  "cache": {
    "reported": true,
    "read_tokens": 245000,
    "write_tokens": 26000,
    "hit_rate_estimate": 0.90
  },
  "fallback_reason": null,
  "tool_call_violation": false
}
```

Content-free audit 继续禁止写入：message content、tool output、tool args、summary 正文、用户原文。

### 7.2 Summary quality sidecar：`logs/compression_summary_samples.jsonl`

新增安全 sidecar，只在 successful LLM summary 时写入，便于后续抽查质量。它可以包含 redacted summary text，但不能包含 raw tool outputs。

字段：

```json
{
  "event": "compression_summary_sample",
  "schema_version": 1,
  "compression_id": "20260708-example-1",
  "session_id": "20260708_example_session",
  "summary_call_mode": "append_cached",
  "rules_hash": "sha256:example-rules-hash",
  "summary_chars": 11023,
  "summary_excerpt": "[redacted first N chars or full if under cap]",
  "section_check": {
    "has_all_canonical_sections": true,
    "missing_sections": [],
    "noncanonical_heading_count": 0,
    "all_user_messages_count": 7,
    "pending_tasks_says_none": false,
    "current_work_present": true,
    "optional_next_step_present": true
  },
  "quality_flags": []
}
```

这个 sidecar 只放 redacted summary，不放压缩前原文。若 summary 超过 `audit_sample_summary_chars`，保留 head+tail，并显式标记 `truncated: true`。

### 7.3 继续保留 user-message ground truth sidecar

现有 `logs/compression_user_messages.jsonl` 继续作为 `## All User Messages` 抽查 ground truth：

- 它存 redacted real user messages。
- 通过 `compression_id` 关联 summary sample。
- 抽查脚本可以比较 summary 中 `All User Messages` 条目数与 ground truth count。

### 7.4 Debug 命令 / 脚本

新增一个只读诊断脚本，建议路径：

```text
scripts/compression_audit_report.py
```

功能：

```bash
python scripts/compression_audit_report.py --last 20
python scripts/compression_audit_report.py --session <session_id>
python scripts/compression_audit_report.py --compression-id <id> --show-summary
```

输出回答四类问题：

1. **缓存是否正常**：append_cached 次数、cache reported 率、median cache hit rate、fallback reasons。
2. **压缩是否有效**：before/after tokens、tail token share、summary chars/tokens、是否 fallback/static summary。
3. **摘要结构是否正常**：九段是否齐全、非 canonical heading、All User Messages count mismatch。
4. **持久化是否正常**：`context_compression_persist.output_row_ids` 是否存在、post_compression_injected_messages 是否被记录。

主动/周期性报告如果后续接入 Discord，默认送 `#ops-alerts`，不要刷 `#日常聊天`。

---

## 8. Fallback 与错误处理

必须显式记录 `fallback_reason`，推荐枚举：

- `summary_call_mode_disabled`
- `summary_runtime_not_main`
- `summary_runtime_context_unknown`
- `append_cached_context_overflow`
- `provider_rejected_tool_choice_none`
- `summary_returned_tool_call`
- `cache_stats_unavailable`
- `append_cached_transport_error`
- `append_cached_validation_failed`
- `rules_hash_mismatch`

处理策略：

1. `cache_stats_unavailable` 不应直接失败；只表示 provider 不报告 cache stats。
2. `summary_returned_tool_call` 必须视为失败；summary call 不允许工具执行。
3. auth/network failure 延续现有 abort-on-auth/network 语义，不能转成静默 static fallback。
4. append_cached 失败后若 `fallback_to_serialized_prompt=true`，可尝试旧模式；audit 必须记录 `mode="serialized_prompt"` 与 `fallback_from="append_cached"`。
5. 两种模式都失败时，沿用现有 `abort_on_summary_failure` / deterministic fallback 策略。

---

## 9. 测试计划

### 9.1 Unit tests

新增或扩展 `tests/agent/test_context_compressor.py`：

- `test_append_cached_summary_call_preserves_rules_hash`
  - 验证 serialized 与 append instruction 的 rules hash 相同。
- `test_append_cached_summary_request_uses_provider_prefix_to_compress_end`
  - 构造 messages，mock summary runtime，断言 summary request 前缀等于主会话 provider payload 到 `compress_end`。
- `test_append_cached_summary_does_not_duplicate_serialized_history`
  - 断言 final instruction 不包含完整 serialized `TURNS TO SUMMARIZE` 历史正文。
- `test_append_cached_tail_not_in_summary_source_v1`
  - retained tail 中的 unique marker 不进入 summary request prefix。
- `test_append_cached_rejects_tool_call_response`
  - mock summary response 含 tool_calls，断言 fallback/abort 且 audit `tool_call_violation=true`。
- `test_append_cached_context_limit_uses_runtime_limit_not_threshold`
  - `threshold_tokens=272000`，`summary_runtime_context_limit=1000000`，request 500K tokens，应允许 append_cached。
- `test_append_cached_context_overflow_falls_back_with_reason`
  - request 超过 runtime limit，audit 写 `append_cached_context_overflow`。

### 9.2 Transport/cache tests

- Chat Completions：cache stats 从 `prompt_tokens_details.cached_tokens/cache_write_tokens` 进入 `summary_call.cache`。
- Anthropic Messages：cache stats 从 Anthropic usage fields 进入统一 audit。
- Codex Responses：保持 Responses payload 的 tool/schema/order 不变，禁止删 tools；如有 provider-specific cache stats，归一化到同一字段。

### 9.3 Integration tests

运行：

```bash
pytest tests/agent/test_*compress*.py tests/run_agent/test_*compress*.py tests/run_agent/test_compression_*.py -q
pytest tests/gateway/test_compress_command.py tests/cli/test_manual_compress.py tests/cli/test_compress_here.py -q
python -m py_compile agent/context_compressor.py agent/conversation_compression.py agent/chat_completion_helpers.py
```

若改到 frontend marker 或 TUI status，再加：

```bash
npm run typecheck --workspace web
npm test --workspace web
```

### 9.4 Live smoke

在 Zongxin profile 上先不开默认，只手动启用：

```yaml
compression:
  summary_call_mode: append_cached
```

Smoke 步骤：

1. 准备一段长会话，包含 user/assistant/tool result，确保触发压缩。
2. 运行 `/compress` 或让 auto compression 触发。
3. 查看 `logs/compression_audit.jsonl` 最新 `context_compression`：
   - `summary_call.mode == "append_cached"`
   - `cache_eligible == true`
   - `cache.reported == true` 时 `read_tokens > 0`
   - `fallback_reason == null`
4. 查看 `logs/compression_summary_samples.jsonl`：
   - 九段齐全。
   - `quality_flags == []` 或只包含解释清楚的非阻断 warning。
5. 查看 `state.db` active rows：summary + retained tail 顺序正确，tool pairs 合法，persist audit 有 `output_row_ids`。

---

## 10. Rollout

1. **Phase 0：Spec review**
   只审本文档，不改代码。
2. **Phase 1：代码实现但默认关闭**
   `compression.summary_call_mode` 默认 `serialized_prompt`；append_cached 只在显式配置下运行。
3. **Phase 2：Zongxin profile canary**
   在 default profile 显式开启，观察至少 10 次压缩：cache hit、summary quality、fallback 率。
4. **Phase 3：默认策略评估**
   如果 canary 没有 summary 质量退化、没有 provider incompatibility、fallback 可解释，再考虑把支持 main-runtime 的 provider 默认切到 append_cached。

---

## 11. 审查清单

Code review 必须看：

- 新模式是否真正复用 `build_api_kwargs()` / transport conversion，而不是手写 provider payload。
- 是否没有删除 tools array 来换 `tool_choice:none`。
- 是否没有把 retained tail 压缩或复制进 summary。
- 是否没有把 raw tool output / tool args / secrets 写进 content-free audit。
- 是否能从 audit 明确判断 cache 命中、fallback、summary 结构质量。
- 是否把 272K threshold 与 1M runtime limit 分开。
- 是否保留现有 summary rules hash 与 canonical sections。
- 是否有失败回滚到旧模式或 abort 的显式路径。

---

## 12. 未来可选增强

- `summary_source_scope: full_history_with_tail_exclusion`：更接近 Claude Code full-history append 模式，但必须先证明 tail 不会被重复写入 summary。
- `compression_audit_report.py --discord #ops-alerts`：周期性把压缩健康报告发到 ops-alerts。
- Summary quality lightweight evaluator：结构检查之外，抽样让独立模型判断是否遗漏 pending/current work；只读 redacted summary + user-message sidecar，不读 raw tool outputs。
- Provider-specific cache key audit：更精确记录 model/effort/fast mode/request headers 对 cache key 的影响。

---

## 13. Spec 自查结果

- **无占位符：** 全文没有占位词或未定执行缺口。
- **范围一致：** 本 Spec 只设计 append-cached summary call 与审查/debug 能力，不进入实现计划。
- **关键歧义已定：** v1 采用 `provider_payload_prefix_to_compress_end`，不采用 full-history append。
- **审查可操作：** audit 字段、sidecar、脚本、测试、live smoke 都给出明确验收点。
