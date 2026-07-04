# 设计:持续 origin 502 → 自动压缩自愈

- 日期:2026-06-22
- 状态:已实现(brainstorming → 本 spec → TDD port)
- 关联代码锚点基于 HEAD `c09ead9`
- 相关记忆:`hermes-gptcodex-502-intermittent`、`hermes-gptcodex-credential-pool`

## 1. 背景与根因

### 现象
gptcodex 中转站(`https://gptcodex.top/v1`,transport `codex_responses`,Cloudflare 后)**没有整体宕机**,但部分 Hermes 会话持续报:

```
API call failed after 10 retries: HTTP 502: ... 'error_name': 'origin_bad_gateway', 'error_category': 'origin'
```

同一时刻另一些会话(如"简短问候")正常。卡住的多是**长会话**(CRAFT 论文推进 = 559 项 input、清理 Notion = 113 项)。

### 根因(已逐一实验排除其它假设后确认)
**gptcodex 的 origin 间歇性过载**——分钟级时好时坏。坏时段倾向**拒绝"重的长会话请求"**(502 `origin_bad_gateway`,或挂起到超时),同时**放行轻请求**;好时段同一请求又 200。

- 与 token 量/字节大小无关:合成纯文本 800KB/1.2MB(含 xhigh reasoning)实测都 200,而真实 347KB 反而 502。重的是 **structured 历史的"项数"**(大量 `function_call`/`function_call_output`/`reasoning` 项),不是字节数。
- 不是 Hermes 发了非法请求:两个失败请求的 `function_call`/`function_call_output` 全配对、无悬空、无畸形(已用 compliance 校验)。
- 不是模型/key/body 大小/reasoning 时长/tools 触发。
- 当天 502 集中在 01:22–02:19(约 259 次,坏窗约 57 分钟)与 03:20–03:28(坏窗约 8 分钟)两段,混有 524/529/429。

人工 `/compress` 能立刻恢复(实测 105 条 → 11 条):它把多轮历史摘要成几条**纯文本 message**,请求大幅变轻,坏时段通过率显著提高;且摘要请求不带 tools、不背 structured 历史,自身不卡、不死锁。

### 为什么现状必死(harness 级结构性缺口)
502 在 Hermes 里的命运链路:

1. `agent/error_classifier.py:864` —— HTTP 500/502 → `FailoverReason.server_error, retryable=True`,而 `should_compress` 与 `should_fallback` **都默认 False**。(`:872-881` 会先把"502 但带请求校验信号"分流成 non-retryable `format_error`,所以走到 `server_error` 的已是真·瞬时/过载型 502。503/529 → `:884` `overloaded`,语义相同。)
2. 进重试循环后:`retryable=True` → 不触发 `:3176` 的"非重试错误 → fallback/abort",也不在 `:2760` 的 rate_limit/billing eager-fallback 集合里 → **直接走通用退避重试,最多 `max_retries=10` 次,全部打在同一个 origin 上**。
3. 退避 `wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=60.0)`(`:3445`),封顶 60s。10 次累计约 **7–8 分钟**。
4. 坏窗动辄 8 分钟(短)到 57 分钟(长)—— **10 次重试在结构上追不出坏窗**,必然耗尽 → 终端报 502。在卡住的会话里发"继续",等于把那坨重历史又塞进同一个坏窗,再死一轮。

**结论**:harness 把"origin 级持续过载"当成"几秒就好的瞬时抖动"处理,用了对分钟级故障毫无胜算的重试策略,且没接任何逃生通道(压缩 / 换 origin)。

## 2. 目标 / 非目标

### 目标
在**单一 origin**约束下(用户确认暂无第二个可切换的中转站),把"origin 持续过载"从"假装重试到死"改成:
- **自愈**:连续多次 502 判定为持续过载 → 自动压缩历史变轻 → 带压缩消息重启,提高坏窗通过率(复刻人工 `/compress` 的有效性)。
- **诚实失败**:压缩后仍持续 502(极坏窗连轻请求都被拒)→ 诚实失败并给清晰错误,不假装能救(符合 fail-fast)。
- 全程**显式不静默**:每步有 status/log。

### 非目标(明确不做)
- 不动全局 `max_retries`。
- 不调默认 `compression.threshold`(用户选"只做自愈 B";阈值与 502 关联弱、预防收益不确定且有一刀切副作用)。
- 不加第二 origin / fallback provider(单 origin 约束;origin 级故障同源凭据池救不了,见 `hermes-gptcodex-credential-pool`)。
- 不改 `error_classifier.py`(用现有 `FailoverReason` 判定即可)。
- 不在本期纳入"自愈压缩用更短 timeout"的加固(沿用默认 600s,保持最小;见 §7 风险)。

## 3. 设计

整体复用现有 `long_context_tier` 自愈路径(`conversation_loop.py:2709-2758`:压缩 → `restart_with_compressed_messages=True; break`)的同款机器,不新造。

### 节 1 — 触发判定(只认"持续",不误伤瞬时抖动)
- 新增重试循环局部计数器 `consecutive_server_errors`(与 `retry_count`/`compression_attempts` 并列初始化,约 `conversation_loop.py:909-912`)。
- 自增:在错误分类消费处(约 `:2217-2234`),当 `classified.reason in {FailoverReason.server_error, FailoverReason.overloaded}` 时 `+1`。
- 清零:① 收到任何成功响应时(成功路径,约 `:4000` 一带,与现有成功后逻辑同区);② 切换 fallback/凭据时(与 `retry_count` 一同清零,本场景无 fallback,但保持语义一致);③ 触发压缩重启后(见节 2)。
- 门槛 **5 次**(可配,见 §4):达到才判定"持续过载"。现有退避到第 5 次累计约 2–3 分钟,足以区分"几秒就好的瞬时抖动"(到不了 5 次)与"分钟级坏窗"(稳定累加)。

### 节 2 — 自愈动作(复用现成机器)
在错误处理区(`:2709` `long_context_tier` 分支同区域,即退避 `sleep` 之前)新增分支:

```text
if (classified.reason in {server_error, overloaded}
    and consecutive_server_errors >= PERSISTENT_OVERLOAD_THRESHOLD   # 5
    and agent.compression_enabled):
    compression_attempts += 1
    if compression_attempts <= max_compression_attempts:            # 3
        显式 status: "⚠️ 检测到 origin 持续过载(HTTP <code>×N)→ 自动压缩上下文重试 (<a>/<max>)…"
        original_len = len(messages)
        messages, active_system_prompt = agent._compress_context(messages, system_message,
                                            approx_tokens=approx_tokens, task_id=effective_task_id)
        conversation_history = None
        if len(messages) < original_len:        # 真压缩了才重启
            consecutive_server_errors = 0
            _retry.restart_with_compressed_messages = True
            break
        # else: 压缩没生效(坏窗里压缩 aux 调用也失败,见节 3 自死锁防护)
        #       → 不重启,fall through 到通用退避,由双护栏兜底
    # compression_attempts 超限 → fall through,由 max_retries 耗尽诚实失败
```

**双护栏天然防无限循环**(都是确定的有界性保证):
- 压缩计入 `max_compression_attempts = 3`(`:912`)。
- 压缩重启经 `:3489-3497` 时 `retry_count += 1`(现有注释明说"压缩重启计入重试上限以防无限循环")——总预算 `max_retries = 10` 始终是硬上限。

**参数交互的诚实预期(已验证 + 对抗式 review 修正)**:压缩重启走**外层循环**的 `continue`,会重新执行每轮初始化(`retry_count=0`、`consecutive_overload_errors=0`),所以 **`retry_count` 不是"压缩次数"的护栏**——真护栏是对话级的 `compression_attempts ≤ 3`(在 run 函数开头初始化、跨重启累加,不被外层循环重置)。因此持续坏窗下**最多自动压缩 3 次**,而非早先误写的"约 1 次";每次压缩后的轻请求都获得完整重试预算去碰坏窗的放行间隙,3 次仍救不了即极坏窗,诚实失败是对的归宿。压缩后请求大幅变轻(类比 105→11),坏窗通过率高,通常前 1-2 次就恢复。`consecutive_overload_errors` **按 origin 计数**(`base_url` 变化即重新数),避免 fallback 换 origin 时旧 origin 的 streak 泄漏到新 origin 误触发。有界性由 `compression_attempts ≤ 3` 单独保证(与 restart 是否重置 `retry_count` 无关)。

### 节 3 — 诚实失败 + 自死锁防护 + 可观测
- **诚实失败**:所有终止都归到**现有 `max_retries` 耗尽路径**(约 `:3406-3432`),不另开 return 分支(改动最小)。仅**增强错误信息**:若本轮 `compression_attempts > 0` 且 `classified.reason` 为 `server_error`/`overloaded`,终端错误改为
  `"origin 持续过载,已自动压缩 N 次尝试绕过 502 仍失败,可能中转站持续过载,请稍后重试。"`
  替代现状干巴巴的 `"API call failed after 10 retries: HTTP 502"`。
- **自死锁防护(关键)**:压缩的 aux 调用走同一个 gptcodex origin,坏窗里它本身也可能 502。已确认 `agent/conversation_compression.py:443-473` 的行为:压缩 aux 失败时 compressor **优雅 abort,原样返回 messages 不变**,并 emit "No messages were dropped" 警告。本设计据此用 `len(messages) < original_len` 检测:没压成就**不重启**,落到通用退避/诚实失败。**不会卡死、不会瞎转。**
- **可观测**:节 2 的压缩 status + 节 3 的增强错误信息 + 现有 `logger.warning` 重试日志,确保自愈与失败全程显式。

## 4. 配置

`config.yaml` 的 `compression` 段新增一项:

```yaml
compression:
  # 连续多少次 origin 过载错误(502/503/524/529)后触发自动压缩自愈。
  # 高=不误伤瞬时抖动但留给压缩后重试的预算少;低=更快逃生但抖动也可能触发。
  persistent_overload_threshold: 5
```

压缩次数上限复用现有 `max_compression_attempts`(硬编码 3),不新增配置。

## 5. 改动面

| 文件 | 改动 | 量级 |
|---|---|---|
| `agent/conversation_loop.py` | 新增 `consecutive_server_errors` 计数器(init + 自增 + 三处清零);新增持续过载自愈分支(`:2709` 同区);增强 `max_retries` 耗尽错误信息 | 主要改动,< 100 行 |
| `config.yaml` | 新增 `compression.persistent_overload_threshold: 5`;读取入口同 `compression.*` 现有逻辑 | 数行 |
| `agent/error_classifier.py` | **不改**(用现有 `FailoverReason`) | 0 |
| `agent/conversation_compression.py` | **不改**(依赖现有优雅 abort 行为) | 0 |

> 因动了**多会话共用的核心重试循环**,落地后按协作约定让 codex 做一轮对抗式 review 再报完成。

## 6. 边界 case

- **确定性坏请求**:`error_classifier.py:872-881` 已把"502 + 请求校验信号"分流为 non-retryable,不进 `server_error` 分支 → 不会被反复压缩。
- **短会话撞持续 502**:压缩压不动(`len` 不变)→ 不重启 → 通用退避兜底到诚实失败。合理:短会话本就轻,还被拒即极坏窗,只能等窗口过去。
- **502 与 429 在坏窗交替**:`consecutive_server_errors` 只在 `server_error`/`overloaded` 自增,429(`rate_limit`)有自己的 eager-fallback/退避路径,不影响本计数器的"持续过载"判定;只有成功响应才清零,故偶发 429 穿插不会重置累加。
- **压缩重启后**:`consecutive_server_errors` 清零重数;`compression_attempts` 保留累加(跨重启护栏);`retry_count` 由 `:3495` 累加(现有行为,不改)。

## 7. 已知风险(诚实标注)

1. **概率性,非根治**:本质是 origin 级故障,真正根治需异地 fallback(不同 base_url 的中转站)。单 origin 下自愈是"提高坏窗通过率",极坏窗仍诚实失败。
2. **压缩 aux timeout = 600s 未加固**(用户选保持最小):坏窗里若压缩调用是 524 类挂起(而非秒级 502),会话可能冻结至多 ~10 分钟才超时返回。后续可补:自愈路径给压缩传更短 timeout(如 120s)。秒级 502 不受影响(优雅 abort 秒级返回)。
3. **`len(messages)` 是"压缩是否生效"的粗代理**(对抗式 review 指出):正常多轮历史压缩会显著减少条目数,该检测可靠;但理论上若单条巨大消息被一条等长 summary 替换(压缩生效但条数不变),会被误判为 no-op 而不重启。本场景(重的是 structured 历史项数,非单条巨大消息)不触发此边界。
4. **仅覆盖以 HTTP 异常抛出的 502/503**(对抗式 review 指出):`codex_responses` transport 若把过载表达成 `status="failed"` 的响应对象而非抛 HTTP 异常,会走 invalid-response 重试路径、不进本自愈。实测 gptcodex 的 502 是 HTTP 异常(被 `classify_api_error` 归为 `server_error`),故当前覆盖;若未来 transport 行为变化需另行处理。

## 8. 验证计划(writing-plans 阶段细化)

用 mock provider 覆盖:
1. 连续返回 N 次 502 后返回 200:验证第 5 次触发压缩、压缩后用轻请求重试并成功、`consecutive_server_errors` 正确清零。
2. 压缩 aux 也返回 502(优雅 abort,`len` 不变):验证不重启、不卡死、落到退避/诚实失败。
3. 持续 502 到总预算耗尽:验证终端错误信息含"已自动压缩 N 次"。
4. 瞬时抖动(连续 2–3 次 502 后 200):验证**不触发**压缩(到不了门槛),上下文无损。
5. 503/529 与 502 同等触发(走 `overloaded` reason)。
6. 回归:`long_context_tier`/`payload_too_large`/`context_overflow` 等现有压缩路径不受影响。
