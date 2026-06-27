# 持续 origin 502 自动压缩自愈 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Hermes 在中转站 origin 持续过载(连续 502/503/524/529)时,自动压缩历史变轻后重试逃生,压缩仍救不了时诚实失败,而不是在同一个 origin 上死磕 10 次必死。

**Architecture:** 复用现有 `long_context_tier` 自愈机器(`_compress_context` + `restart_with_compressed_messages`)。新增一个每-请求的 `consecutive_overload_errors` 计数器,连续过载到阈值(默认 5)即触发压缩-重启;`compression_attempts`(对话级,≤3)与 `max_retries`(≤10)双护栏保证有界;压缩 aux 在坏窗也失败时依赖 compressor 现有的"优雅 abort 原样返回",用 `len` 未减小检测、不重启 → 落到诚实失败。判定逻辑抽成纯函数单测,接线用端到端测试(`test_413_compression.py` 同款 fixture)覆盖。

**Tech Stack:** Python 3, pytest, `unittest.mock`。改动集中在 `agent/conversation_loop.py`,配置在 `agent/agent_init.py` + `config.yaml`。

## Global Constraints

- **不改** `agent/error_classifier.py` 与 `agent/conversation_compression.py`(复用现有 `FailoverReason` 与"压缩失败优雅 abort 原样返回"行为)。
- **不动**全局 `agent._api_max_retries`(=10)、默认 `compression.threshold`;**不加**第二 origin / fallback provider。
- **fail-fast,不静默**:自愈每步 `_buffer_status` 显式提示;压缩救不了时归到现有 `max_retries` 耗尽路径诚实失败并增强错误信息。
- 行号基于 HEAD `c09ead9`,实现时以符号 + 邻近代码上下文定位(行号可能已漂移)。
- 设计文档:`docs/design/2026-06-22-persistent-502-self-heal.md`。
- 本改动动了多会话共用的核心重试循环,落地后须让 codex 做一轮对抗式 review 再报完成(用户协作约定)。

---

### Task 1: 自愈判定纯函数 + 单元测试

把"是否触发持续过载自愈"的判定从巨型循环里抽成一个可独立单测的纯函数(贴合 `test_long_context_tier_429.py` 的"判定逻辑单测"风格)。

**Files:**
- Modify: `agent/conversation_loop.py`(模块级新增函数,放在文件顶部 import 块之后、第一个类/函数之前)
- Test: `tests/agent/test_persistent_502_self_heal.py`(新建)

**Interfaces:**
- Produces: `_should_self_heal_persistent_overload(*, reason, consecutive_overload_errors, threshold, compression_enabled, compression_attempts, max_compression_attempts) -> bool`
- Consumes: `agent.error_classifier.FailoverReason`(`conversation_loop.py` 已 import)

- [ ] **Step 1: 写失败测试**

新建 `tests/agent/test_persistent_502_self_heal.py`:

```python
"""Unit tests for the persistent origin-overload self-heal decision.

The loop wiring is exercised end-to-end in
tests/run_agent/test_persistent_502_self_heal_e2e.py; this file pins the pure
decision function. See docs/design/2026-06-22-persistent-502-self-heal.md.
"""

from agent.conversation_loop import _should_self_heal_persistent_overload
from agent.error_classifier import FailoverReason


def _call(**override):
    base = dict(
        reason=FailoverReason.server_error,
        consecutive_overload_errors=5,
        threshold=5,
        compression_enabled=True,
        compression_attempts=0,
        max_compression_attempts=3,
    )
    base.update(override)
    return _should_self_heal_persistent_overload(**base)


def test_triggers_at_threshold():
    assert _call() is True


def test_overloaded_reason_also_triggers():
    # 503/529 classify as FailoverReason.overloaded — same self-heal.
    assert _call(reason=FailoverReason.overloaded) is True


def test_below_threshold_does_not_trigger():
    assert _call(consecutive_overload_errors=4) is False


def test_non_overload_reason_does_not_trigger():
    assert _call(reason=FailoverReason.rate_limit) is False
    assert _call(reason=FailoverReason.format_error) is False
    assert _call(reason=FailoverReason.billing) is False


def test_compression_disabled_does_not_trigger():
    assert _call(compression_enabled=False) is False


def test_compression_budget_exhausted_does_not_trigger():
    assert _call(compression_attempts=3) is False
    assert _call(compression_attempts=4) is False


def test_just_under_budget_triggers():
    assert _call(compression_attempts=2) is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/agent/test_persistent_502_self_heal.py -v`
Expected: FAIL，`ImportError: cannot import name '_should_self_heal_persistent_overload'`

- [ ] **Step 3: 写最小实现**

在 `agent/conversation_loop.py` 文件顶部 import 块之后新增(确认 `FailoverReason` 已在该文件 import;若只 import 了部分符号,补 import):

```python
def _should_self_heal_persistent_overload(
    *,
    reason,
    consecutive_overload_errors,
    threshold,
    compression_enabled,
    compression_attempts,
    max_compression_attempts,
):
    """Decide whether a persistent origin-overload streak should trigger a
    compression-based self-heal.

    Origin-level intermittent overload (Cloudflare ``origin_bad_gateway`` 502,
    or 503/529 ``overloaded``) lasts minutes — longer than the same-origin
    retry budget can outlast, and heavy structured sessions get rejected while
    light requests pass. Once we've seen ``threshold`` consecutive overload
    errors on one request, compress the history into a lighter request (far
    higher pass rate in a bad window, as manual ``/compress`` recovery proved)
    and retry. Bounded by ``max_compression_attempts`` so a truly-bad window
    (where even the light compression call 502s) fails honestly instead of
    looping. A single transient blip never reaches ``threshold``.
    """
    return (
        reason in (FailoverReason.server_error, FailoverReason.overloaded)
        and consecutive_overload_errors >= threshold
        and compression_enabled
        and compression_attempts < max_compression_attempts
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/agent/test_persistent_502_self_heal.py -v`
Expected: PASS（8 个测试全绿）

- [ ] **Step 5: 提交**

```bash
cd ~/.hermes/hermes-agent
git add agent/conversation_loop.py tests/agent/test_persistent_502_self_heal.py
git commit -m "feat: persistent origin-overload self-heal decision fn

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 配置项 `compression.persistent_overload_threshold`

让阈值可配(默认 5),跟随现有 `compression.*` 读取模式。

**Files:**
- Modify: `agent/agent_init.py:1330`(读取)、`:1561`(赋 agent 属性) — 锚点附近
- Modify: `config.yaml`(`compression` 段)
- Test: `tests/agent/test_persistent_502_self_heal.py`(追加一个默认值断言;复用 Task 3 的 agent fixture 风格的轻量构造)

**Interfaces:**
- Produces: `agent._persistent_overload_threshold: int`(默认 5),供 `conversation_loop` 用 `getattr(agent, "_persistent_overload_threshold", 5)` 读取。

- [ ] **Step 1: 写失败测试**

在 `tests/agent/test_persistent_502_self_heal.py` 末尾追加:

```python
def test_agent_reads_persistent_overload_threshold_default():
    """A freshly built AIAgent exposes the threshold attr, default 5."""
    from unittest.mock import patch, MagicMock
    from run_agent import AIAgent
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://gptcodex.top/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    assert getattr(a, "_persistent_overload_threshold", None) == 5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/agent/test_persistent_502_self_heal.py::test_agent_reads_persistent_overload_threshold_default -v`
Expected: FAIL，`assert None == 5`（属性不存在）

- [ ] **Step 3: 写最小实现**

在 `agent/agent_init.py` 的 `:1330`（`compression_protect_last = int(_compression_cfg.get("protect_last_n", 20))`）之后新增:

```python
    persistent_overload_threshold = int(
        _compression_cfg.get("persistent_overload_threshold", 5)
    )
```

在 `:1561`（`agent.compression_enabled = compression_enabled`）之后新增:

```python
    agent._persistent_overload_threshold = persistent_overload_threshold
```

在 `config.yaml` 的 `compression:` 段内新增一行(与 `threshold`/`protect_last_n` 同级):

```yaml
  # 连续多少次 origin 过载错误(502/503/524/529)后触发自动压缩自愈逃生。
  # 高=不误伤瞬时抖动但留给压缩后重试的预算少;低=更快逃生。见
  # docs/design/2026-06-22-persistent-502-self-heal.md。
  persistent_overload_threshold: 5
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/agent/test_persistent_502_self_heal.py -v`
Expected: PASS（含新增的默认值测试）

- [ ] **Step 5: 提交**

```bash
cd ~/.hermes/hermes-agent
git add agent/agent_init.py config.yaml tests/agent/test_persistent_502_self_heal.py
git commit -m "feat: configurable compression.persistent_overload_threshold (default 5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 接线计数器 + 自愈分支 + 诚实失败信息(端到端)

把判定函数接进重试循环:计数连续过载、达阈值触发压缩-重启、压缩没生效则诚实失败并增强错误信息。

**Files:**
- Modify: `agent/conversation_loop.py:912`（计数器初始化）、`:2236` 后（自增）、`:2758` 后（自愈分支）、`:3409`（错误信息增强）
- Test: `tests/run_agent/test_persistent_502_self_heal_e2e.py`(新建)

**Interfaces:**
- Consumes: `_should_self_heal_persistent_overload`(Task 1)、`agent._persistent_overload_threshold`(Task 2)、现有 `agent._compress_context(messages, system_message, *, approx_tokens, task_id) -> (messages, system_prompt)`、`_retry.restart_with_compressed_messages`、循环局部 `compression_attempts` / `max_compression_attempts` / `status_code` / `system_message` / `approx_tokens` / `effective_task_id` / `active_system_prompt`。

- [ ] **Step 1: 写失败的端到端测试**

新建 `tests/run_agent/test_persistent_502_self_heal_e2e.py`:

```python
"""End-to-end tests for persistent origin-overload (502/503/524/529) self-heal.

After N consecutive overload errors on one request, the loop compresses
history and retries with a lighter request (mirrors manual /compress); if
compression can't reduce (bad window 502s the aux call too), it fails
honestly. Fixture mirrors tests/run_agent/test_413_compression.py.
See docs/design/2026-06-22-persistent-502-self-heal.md.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
import run_agent


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Short-circuit retry backoff so 5 consecutive 502s run instantly."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


def _mock_response(content="ok", finish_reason="stop"):
    msg = SimpleNamespace(
        content=content, tool_calls=None, reasoning_content=None, reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp


def _make_overload(status=502, message="HTTP 502: origin_bad_gateway"):
    err = Exception(message)
    err.status_code = status
    return err


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://gptcodex.top/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = True
        a.save_trajectories = False
        a._persistent_overload_threshold = 5
        return a


def _prefill():
    return [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]


def test_5_consecutive_502_triggers_compression(agent):
    """5 × 502 → compress → lighter request succeeds on the 6th call."""
    agent.client.chat.completions.create.side_effect = [
        _make_overload(), _make_overload(), _make_overload(),
        _make_overload(), _make_overload(),
        _mock_response(content="recovered after compression"),
    ]
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        mock_compress.return_value = (
            [{"role": "user", "content": "summary"}], "compressed prompt",
        )
        result = agent.run_conversation("hello", conversation_history=_prefill())

    mock_compress.assert_called_once()
    assert result["completed"] is True
    assert result["final_response"] == "recovered after compression"


def test_below_threshold_no_compression(agent):
    """2 × 502 then success — a transient blip must NOT compress."""
    agent.client.chat.completions.create.side_effect = [
        _make_overload(), _make_overload(), _mock_response(content="ok"),
    ]
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=_prefill())

    mock_compress.assert_not_called()
    assert result["completed"] is True


def test_503_overload_also_triggers(agent):
    """503 (FailoverReason.overloaded) self-heals identically to 502."""
    agent.client.chat.completions.create.side_effect = [
        _make_overload(status=503), _make_overload(status=503),
        _make_overload(status=503), _make_overload(status=503),
        _make_overload(status=503),
        _mock_response(content="ok after overload"),
    ]
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        mock_compress.return_value = (
            [{"role": "user", "content": "summary"}], "compressed",
        )
        result = agent.run_conversation("hello", conversation_history=_prefill())

    mock_compress.assert_called_once()
    assert result["completed"] is True


def test_compression_noop_fails_honestly(agent):
    """When the bad window also 502s the aux call (compression no-op →
    unchanged messages), the loop must NOT loop forever; it exhausts the
    bounded budget and fails with a message that explains compression was
    tried."""
    agent.client.chat.completions.create.side_effect = _make_overload()
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        # No-op compression: returns same messages (graceful abort).
        mock_compress.side_effect = lambda msgs, *a, **k: (msgs, agent._cached_system_prompt)
        result = agent.run_conversation("hello", conversation_history=_prefill())

    assert result.get("completed") is not True
    blob = (result.get("final_response", "") or "") + (result.get("error", "") or "")
    assert "压缩" in blob or "中转站持续过载" in blob


def test_compression_disabled_no_self_heal(agent):
    """compression_enabled=False → never self-heals (honest fail on exhaustion)."""
    agent.compression_enabled = False
    agent.client.chat.completions.create.side_effect = _make_overload()
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=_prefill())

    mock_compress.assert_not_called()
    assert result.get("completed") is not True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/run_agent/test_persistent_502_self_heal_e2e.py -v`
Expected: FAIL — `test_5_consecutive_502_triggers_compression` 报 `mock_compress` 未被调用(自愈分支还没接线,502 会死磕到耗尽)。

- [ ] **Step 3a: 接线 — 初始化计数器**

`agent/conversation_loop.py:912`（`max_compression_attempts = 3`）之后新增:

```python
        # Consecutive origin-overload (502/503/524/529) errors on THIS request.
        # Re-initialised per api-call attempt (this block runs once per attempt
        # before the `while retry_count < max_retries` loop). Drives the
        # persistent-overload self-heal below; only a *streak* of overloads
        # (not a single transient blip) reaches the threshold.
        consecutive_overload_errors = 0
```

- [ ] **Step 3b: 接线 — 自增计数器**

`agent/conversation_loop.py:2236`（`_invoke_api_request_error_hook(...)` 调用结束的 `)` 之后、`:2238` 的 `if classified.reason == FailoverReason.billing` 之前）新增:

```python
                # Bump the consecutive origin-overload streak. Only overload
                # reasons bump it; other error types neither bump nor clear it,
                # so a 429 interleaved in a bad window won't reset the streak.
                # A successful API call resets it via the per-attempt re-init
                # at the top of the retry loop (success breaks out; the next
                # tool-call turn re-enters with the counter at 0).
                if classified.reason in (
                    FailoverReason.server_error,
                    FailoverReason.overloaded,
                ):
                    consecutive_overload_errors += 1
```

- [ ] **Step 3c: 接线 — 自愈分支**

`agent/conversation_loop.py:2758`（`long_context_tier` 分支结束的注释 `# Fall through to normal error handling ...` 之后、`:2760` 的 rate-limit eager-fallback 注释之前）新增:

```python
                # ── Persistent origin-overload self-heal (502/503/524/529) ──
                # See docs/design/2026-06-22-persistent-502-self-heal.md. An
                # origin-level intermittent overload lasts minutes; the same-
                # origin retry budget can't outlast it. After `threshold`
                # consecutive overload errors, compress history into a lighter
                # request and restart — mirrors the manual /compress recovery.
                # Bounded by max_compression_attempts; if the bad window 502s
                # the aux compression call too, the compressor returns messages
                # unchanged (graceful abort) → no restart → honest failure.
                if _should_self_heal_persistent_overload(
                    reason=classified.reason,
                    consecutive_overload_errors=consecutive_overload_errors,
                    threshold=getattr(agent, "_persistent_overload_threshold", 5),
                    compression_enabled=getattr(agent, "compression_enabled", True),
                    compression_attempts=compression_attempts,
                    max_compression_attempts=max_compression_attempts,
                ):
                    compression_attempts += 1
                    agent._buffer_status(
                        f"⚠️ 检测到中转站持续过载(HTTP {status_code}×{consecutive_overload_errors})"
                        f" → 自动压缩上下文后重试 ({compression_attempts}/{max_compression_attempts})…"
                    )
                    original_len = len(messages)
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message,
                        approx_tokens=approx_tokens,
                        task_id=effective_task_id,
                    )
                    # Compression created a new session — clear history so
                    # _flush_messages_to_session_db writes compressed messages
                    # to the new session (same as the long_context_tier path).
                    conversation_history = None
                    if len(messages) < original_len:
                        # Real reduction → retry with the lighter request.
                        consecutive_overload_errors = 0
                        _retry.restart_with_compressed_messages = True
                        break
                    # else: compression no-op (aux call also failed in the bad
                    # window → graceful abort returned messages unchanged).
                    # Don't restart; fall through to normal backoff, bounded by
                    # max_retries → honest failure.
```

- [ ] **Step 3d: 接线 — 诚实失败信息增强**

`agent/conversation_loop.py:3409`（`else:` 分支里 `_final_response = f"API call failed after {max_retries} retries: {_final_summary}"`）之后新增:

```python
                        # Persistent origin-overload self-heal exhausted: the
                        # terminal message should explain we compressed to try
                        # to slip a lighter request through a bad window, rather
                        # than a bare "502 after N retries".
                        if (
                            compression_attempts > 0
                            and classified.reason in (
                                FailoverReason.server_error,
                                FailoverReason.overloaded,
                            )
                        ):
                            _final_response = (
                                f"中转站持续过载(HTTP {classified.status_code}):"
                                f"已自动压缩 {compression_attempts} 次尝试绕过仍失败,"
                                f"可能 origin 持续抖动,请稍后重试或检查中转站。"
                                f"原始错误:{_final_summary}"
                            )
```

- [ ] **Step 4: 跑端到端测试确认通过**

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/run_agent/test_persistent_502_self_heal_e2e.py -v`
Expected: PASS（5 个测试全绿）

- [ ] **Step 5: 跑回归 — 现有压缩/重试路径不受影响**

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/run_agent/test_413_compression.py tests/run_agent/test_long_context_tier_429.py tests/run_agent/test_infinite_compaction_loop.py tests/agent/test_turn_retry_state.py tests/agent/test_error_classifier.py -v`
Expected: PASS（全绿，无回归）

- [ ] **Step 6: 提交**

```bash
cd ~/.hermes/hermes-agent
git add agent/conversation_loop.py tests/run_agent/test_persistent_502_self_heal_e2e.py
git commit -m "feat: self-heal on persistent origin overload (502/503/524/529)

连续 N 次 origin 过载 → 自动压缩历史变轻 + 重启重试(复用 long_context_tier
机器);压缩 aux 在坏窗也失败时优雅降级不死循环;耗尽后诚实失败并说明已压缩。
compression_attempts(<=3) 与 max_retries(<=10) 双护栏保证有界。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 真机冒烟验证

单测/端到端全绿后,在真实 Hermes 进程里验证自愈链路真的触发(而非只在 mock 里)。

**Files:** 无代码改动(验证脚本写到 scratchpad)。

- [ ] **Step 1: 全量相关测试再跑一遍**

Run: `cd ~/.hermes/hermes-agent && python -m pytest tests/agent/test_persistent_502_self_heal.py tests/run_agent/test_persistent_502_self_heal_e2e.py -v`
Expected: PASS（全绿）

- [ ] **Step 2: 真机注入式冒烟**

用一个最小脚本(scratchpad)构造真实 `AIAgent`(走真实 config 加载,验证 `agent._persistent_overload_threshold` 从 `config.yaml` 读到 = 5),用 `MagicMock` 客户端注入 `side_effect=[502×5, ok]`,跑 `run_conversation`,确认:
- 第 5 次 502 触发了 `_compress_context`(在真实 `_compress_context` 上挂一个计数 patch,或观察 status 输出含"检测到中转站持续过载");
- 最终 `result["completed"] is True`。

这一步验证"config → agent 属性 → 循环判定 → 压缩重启"的真实端到端接线(端到端测试里阈值是手动设的属性,这里验证它真从 config 读出)。

- [ ] **Step 3: codex 对抗式 review**

按用户协作约定,动了多会话共用的核心重试循环,落地后让 codex 做一轮对抗式 review。重点审:
- 计数器作用域/重置时机有无遗漏(restart 路径不经过 `:912` 重置 → 自愈分支已显式清零);
- 自愈分支引用的循环局部变量(`status_code`/`system_message`/`approx_tokens`/`effective_task_id`/`active_system_prompt`)在该作用域确实可用;
- 双护栏是否真的有界(无任何路径能让压缩/重试无限);
- 是否与现有 `long_context_tier`/`payload_too_large`/`context_overflow` 分支顺序冲突(本分支在 `long_context_tier` 之后、rate-limit eager-fallback 之前)。

按 review 结果:明显 bug 直接修;"表面修复、根因在别处"优先处理;与用户偏好冲突的(如劝加静默兜底)默认不采纳;有争议的列出来交用户拍板。

---

## Self-Review

**1. Spec coverage**（对照 `docs/design/2026-06-22-persistent-502-self-heal.md`）:
- 节 1 触发判定(计数器 + 阈值 + 只认 server_error/overloaded)→ Task 1(判定函数)+ Task 3a/3b(初始化/自增)。✓
- 节 2 自愈动作(复用 `_compress_context`+restart、`len` 检测、双护栏)→ Task 3c。✓
- 节 3 诚实失败 + 自死锁防护 + 可观测 → Task 3c(no-op 不重启)+ Task 3d(错误信息)+ `_buffer_status`。✓
- §4 配置项 → Task 2。✓
- §6 边界(确定性坏请求分流、短会话、429 交替、重启清零)→ 由"只认 overload reason 自增 + 成功/重启清零 + 不改 error_classifier 的 502 校验分流"覆盖;Task 3b 注释 + Task 3 端到端 `test_below_threshold` / `test_compression_noop` 覆盖。✓
- §8 验证计划 6 项 → Task 1 单测(瞬时抖动不触发=判定 below_threshold)+ Task 3 端到端(5×502 触发、503 触发、no-op 诚实失败、disabled 不触发)+ Task 3 Step 5 回归。✓

**2. Placeholder scan:** 无 TBD/TODO;每个代码步骤含完整可粘贴代码;命令含预期输出。✓

**3. Type consistency:** 判定函数签名(Task 1 Produces)与 Task 3c 调用处关键字参数逐一对应(`reason`/`consecutive_overload_errors`/`threshold`/`compression_enabled`/`compression_attempts`/`max_compression_attempts`)。计数器名 `consecutive_overload_errors` 在 3a/3b/3c 一致。`agent._persistent_overload_threshold`(Task 2 Produces)与 Task 3c 的 `getattr(agent, "_persistent_overload_threshold", 5)` 一致。✓
