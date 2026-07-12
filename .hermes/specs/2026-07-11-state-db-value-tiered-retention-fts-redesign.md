# Hermes State DB Value-Tiered Retention + FTS Redesign

日期：2026-07-11
状态：Draft for Zongxin approval — live DB mutation未授权
范围：Hermes `state.db` 索引结构、compacted payload 生命周期、session retention；不改变用户可见聊天语义，不在本设计审批前修改 live DB。

## 1. 结论先行

采用三阶段、三个独立 rollout gate：

1. **Phase 1 — FTS v2（推荐先实施）**：只改派生搜索索引，不删/改 `messages` 与 `sessions` 语义。两套 FTS 改为 external-content；unicode 保留当前完整搜索投影，trigram 对 `user`/`assistant` 仍索引当前完整 `content + tool_name + tool_calls` 投影，只排除默认被视为 noise 的 tool/system/session_meta rows。现有显式 tool search 保留；tool/CJK 查询改走 role-aware LIKE fallback。
2. **Phase 2 — compacted payload value-tiering（本轮只做无schema写的read-only estimator）**：正确apply需要新增真正的 `compacted_at` 时钟，但该column不能随Phase 1兼容代码自动reconcile到live DB；必须等Phase 2单独审批后由显式schema command添加。被 `hermes://session/.../message/<rowid>` recovery handle 引用的 tool row 必须豁免，除非先原子迁移到可验证的 durable artifact resolver。
3. **Phase 3 — source-aware whole-session retention（设计保留，默认禁用）**：先修 `ended_at`/`last_activity_at` 生命周期并建立 retention hold/reference registry；`archived=1` 是强制hold。在此之前不启用自动整-session 删除。

**审批建议：批准 Phase 1 实现 + Phase 2 read-only estimator；Phase 2 schema/plumbing/apply 与 Phase 3 apply 分别再次审批。**

## 2. 当前真相与根因

### 2.1 物理与逻辑构成

2026-07-11 aggregate-only read-only audit（无 message 正文输出）：

- `state.db` 约 5.7–5.8 GiB；约 320k messages，历史跨度约 21.84 天。
- 两套 FTS 约占 74%；`messages` 本体约占 24%；二者合计约 98%。
- 现有 `messages_fts` 与 `messages_fts_trigram` 都把 `content + tool_name + tool_calls` 写入 inline FTS（`hermes_state.py:809-860`）。
- 当前 trigram 输入投影约 881.24M 字符；若只保留 user/assistant 但继续索引完整旧投影，约 203.88M，输入减少约 76.86%。不能只索引 visible content：assistant `tool_calls` 中的CJK term在v1可命中，丢掉它会让默认paired match set必然不一致。
- 两个 inline `%_content` shadow 合计约 1.95 GiB；external-content 可去掉这两份重复正文。

### 2.2 为什么 90 天 prune 不是眼前解法

- `sessions.auto_prune` 默认 false（`hermes_cli/config.py:2916-2941`）。
- `prune_sessions()` 用 `started_at < cutoff AND ended_at IS NOT NULL`（`hermes_state.py:5155-5208`），不是结束时间或最后活动时间。
- 当前 DB 只有约 22 天历史；90 天不会删除任何东西。
- `ended_at IS NULL` 在多个 source 中大量存在；它混合真实 active、可 resume、异常未 finalize 与某些工具/子代理生命周期，不能按年龄直接删。
- startup auto-maintenance 吞异常并只写 warning（`hermes_state.py:5828-5897`），不适合作为高风险 schema/data migration owner。

### 2.3 compacted payload 的时间与引用风险

`archive_and_compact()` 只执行 `active=0, compacted=1`，保留原 message `timestamp`，没有 `compacted_at`（`hermes_state.py:3506-3554`）。因此按原 timestamp 做 retention 会让“今天刚 compact 的旧消息”立即过期，是错误时钟。

Live aggregate：

| compacted row age（按旧 timestamp，仅敏感性分析） | rows | logical chars |
|---:|---:|---:|
| >0d | 141,721 | 472.08M |
| >1d | 92,748 | 338.99M |
| >3d | 44,640 | 178.36M |
| >7d | 32,385 | 137.10M |
| >14d | 575 | 2.72M |

14 天候选目前太小，无法解决即时空间；把窗口偷改成 7 天虽增加候选，但属于用户证据保留偏好变化，不能在 FTS rollout 中夹带。

另有约 21,026 个 `hermes://session/.../message/<rowid>` handles；约 14,230 个唯一有效 target 都是 `active=0, compacted=1, role=tool`，合计约 54.07M content chars。它们是现有 cheap-cleanup 的 durable recovery source；截断 target 会让 handle 名存实亡。

Evidence artifacts：

- `/Users/zongxin/clawd/work/worktree-storage-maintenance/state-db-audit-2026-07-11.md`
- `/Users/zongxin/clawd/work/state-db-value-tiered-retention/live-aggregate-2026-07-11.json`

## 3. Hard invariants

1. **Live DB mutation gate**：本 spec、estimator、copy benchmark 不得对 live DB 执行 UPDATE/DELETE/drop/rebuild/optimize/VACUUM。
2. **No semantic deletion in Phase 1**：`sessions`/`messages` row count、visible content、tool content、tool_calls、reasoning/replay fields全部不变。
3. **Search contract**：默认 `session_search` 的 `user,assistant` 搜索保留；显式 tool English/unicode search 保留；tool/CJK search 可从 trigram 改为 LIKE，但不能静默变成“搜不到”。
4. **Handle durability**：任何被 active 或 compacted message 引用的 archived row，在 resolver 成功迁移并 readback 前不得截断/删除。
5. **Protocol/replay durability**：`active=1` rows 永不进入 payload retention；Phase 2 不得破坏当前 continuation/Codex replay。
6. **No startup heavy migration**：SessionDB init 可以识别 v1/v2，但不能在 Gateway/CLI startup 自动重建 5+ GiB FTS 或 VACUUM。
7. **Fail closed and visible**：migration/retention failure 非零退出、保留 maintenance marker 与原 DB；不得记录“已迁移”后静默 fallback 到旧/半成品 index。
8. **Copy-first rollback**：apply 前必须有 SQLite-consistent 0600 backup；新 DB 在独立文件中验证后才能原子 swap。
9. **Privacy**：estimator/report 只输出 counts/bytes/hash/overlap/latency，不输出 message/query/session正文。

## 4. Phase 1 — FTS v2 target

### 4.1 Projection views

使用 view 而不是 generated base-table column，避免改 `messages` row layout：

```sql
CREATE VIEW messages_fts_unicode_content_v2 AS
SELECT id,
       coalesce(content,'') || ' ' ||
       coalesce(tool_name,'') || ' ' ||
       coalesce(tool_calls,'') AS content
FROM messages;

CREATE VIEW messages_fts_trigram_content_v2 AS
SELECT id,
       coalesce(content,'') || ' ' ||
       coalesce(tool_name,'') || ' ' ||
       coalesce(tool_calls,'') AS content
FROM messages
WHERE role IN ('user','assistant');
```

unicode projection与当前索引语义一致。trigram view也保留user/assistant的当前完整投影，只排除默认 `session_search` 已视为noise的其他roles（默认 discovery 明确使用 `user,assistant`：`tools/session_search_tool.py:499-520`）。

### 4.2 External-content indexes

```sql
CREATE VIRTUAL TABLE messages_fts USING fts5(
  content,
  content='messages_fts_unicode_content_v2',
  content_rowid='id'
);

CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
  content,
  content='messages_fts_trigram_content_v2',
  content_rowid='id',
  tokenize='trigram'
);
```

SQLite 官方文档允许 external content 指向 table、virtual table 或 view；内容一致性由应用和 triggers 负责，`rebuild` 可从 external source 重建。官方依据：<https://sqlite.org/fts5.html#external_content_tables>。

Disposable SQLite 3.53.1 spike 已证明：

- 两个 `%_content` shadow 都不存在；
- insert/update/delete、role transition、`rebuild`、rank-1 `integrity-check` 全通过；
- trigram CJK substring 与 `snippet()` 正常；
- tool row 不进入 selective trigram。

### 4.3 Trigger与repair owner

v2 triggers必须使用 external-content delete command并保持 projection一致：

```sql
INSERT INTO messages_fts(messages_fts,rowid,content)
VALUES('delete', old.id, <old unicode projection>);
```

trigram trigger只对 old/new role 属于 user/assistant 时 delete/insert，写入/删除值必须是同一个完整 `content + tool_name + tool_calls` projection；不能只写 `content`。

以下路径必须 schema-aware，不能只改 `FTS_SQL`：

- `_rebuild_fts_indexes()`（当前 inline `DELETE + INSERT SELECT`：`hermes_state.py:1066-1090`）；
- `_ensure_fts_schema()` / startup reconciliation（`hermes_state.py:1092-1134, 1340-1490`）；
- malformed/FTS corruption repair（`hermes_state.py:556-690`）；
- optimize/vacuum health checks；
- tests for no-FTS5/no-trigram fallback。

采用独立 `state_meta.fts_schema_version = 1|2`，不把 5+ GiB data migration塞进通用 `SCHEMA_VERSION` startup reconciliation：

- 新 DB：在同一创建transaction中直接创建v2 views/tables/triggers并写marker=2；
- 现有 inline DB：startup只读检测并在内存中视为effective v1；**不得**为了补marker或准备Phase 2而写任何schema/meta；
- 现有external tables但marker缺失/矛盾：fail closed，要求显式`fts-status`/`fts-resume`，不能猜测并repair；
- trigger repair必须根据effective v1或已验证marker=2选择DDL；
- 显式migration在candidate副本rank-1 integrity、paired search和field digest通过后写marker=2；该marker随candidate原子swap进入live。

### 4.4 Search routing

`search_messages()`当前对 ≥3 CJK token优先 trigram，并依赖 `snippet(...,0,...)`（`hermes_state.py:4375-4552`）。v2 routing：

- `role_filter` 是 user/assistant 子集：trigram fast path，索引/查询完整旧projection；
- `role_filter` 包含 tool/system/session_meta，或底层调用没有 role_filter（意味着 all roles）：跳过 selective trigram，走 escaped LIKE path；
- LIKE必须在与v1相同的 `content + tool_name + tool_calls` projection上匹配，并从同一projection生成snippet；metadata-only命中不能返回空/无term snippet，同时返回的message `content`字段仍保持原正文；
- English/unicode path继续走完整 external unicode FTS；
- short CJK继续 LIKE；
- fallback必须补齐并保留 source/exclude/role/active/compacted filters与dedupe。

必须新增显式 `role_filter=['tool']` CJK regression，防止 selective trigram让 tool search静默漏结果。

## 5. Phase 1 explicit migration and rollback

### 5.1 Commands

正式实现提供独立命令，而不是 startup side effect：

```text
hermes sessions fts-plan
hermes sessions fts-migrate --apply
hermes sessions fts-status
hermes sessions fts-resume
hermes sessions fts-abort
hermes sessions fts-rollback <backup>
```

`fts-plan`纯只读，报告：schema version、DB/WAL大小、free disk、预计临时空间、row counts、index object bytes、writers/maintenance state、paired-query corpus版本。

### 5.2 Durable phase journal and maintenance protocol

外部owner是 `~/.hermes/state-db-maintenance.json`（0600），不是DB内row。每次phase transition用temp file + file `fsync` + atomic replace + parent-directory `fsync`；至少记录 `operation_id`、source/backup/candidate path与SHA-256、phase、timestamps、expected schema/row counts。phase enum：

```text
planned → writers_stopped → checkpointed → backup_ready → candidate_ready
→ swapping → old_moved → candidate_live → canary_passed → complete
```

`aborted`与`rolled_back`是终态。任何非终态journal本身就是maintenance marker。实现必须提供一个按canonical DB path工作的中央pre-open guard（例如 `assert_state_db_maintenance_access(path, operation_id, write_capable)`），并在**每一个write-capable SQLite connect/checkpoint/write probe之前**调用；范围不只`SessionDB`，还包括 `_db_opens_cleanly()` 的rolled-back write probe、`repair_state_db_schema()`、FTS/schema repair、`hermes doctor --fix`、WAL checkpoint、migration与任何raw health/doctor helper。marker存在时：

- 普通write-capable路径全部fail closed并显示`fts-status/fts-resume/fts-abort`；
- read-only status/audit只允许显式 `mode=ro + query_only`，不得顺手执行write probe/checkpoint/repair；
- 只有当前migration/canary process持有与journal匹配的scoped `operation_id` 才能bypass；普通CLI flag不能伪造bypass；
- rollout在写`planned`前后都必须证明没有未加载该guard的旧binary participant，否则不能继续。

执行顺序：

1. 部署兼容代码但保持v1；startup只读detect，不写marker/meta/schema。所有长期 Hermes participants重启到兼容版本。
2. `fts-migrate --apply` 原子写`planned` journal；停止 Gateway/Desktop/CLI/Cron等writers并证明没有旧binary participant。
3. migration bypass打开source，**在任何write transaction之前**执行 `wal_checkpoint(TRUNCATE)`；要求返回 `busy=0` 且checkpoint后WAL frames=0，否则退出。关闭连接后确认sidecars absent或WAL=0；不能先拿write/exclusive transaction再checkpoint。
4. 写`checkpointed`；用SQLite backup API创建0600 consistent rollback backup和独立work copy，不能裸`cp state.db`忽略WAL。验证backup `quick_check`、row counts和hash后写`backup_ready`。
5. 只在work copy创建views、FTS v2、triggers并rebuild；执行row/field-digest equality、rank-1 FTS integrity、trigger write probe rollback、paired search、`quick_check`。
6. `VACUUM INTO`生成紧凑candidate；再次验证并关闭所有candidate连接。fsync candidate file；candidate不得带未checkpoint的`-wal/-shm`。写`candidate_ready`。
7. 所有source/candidate SQLite handles关闭后写`swapping`。source checkpoint后即使存在零frame `state.db-wal`/`state.db-shm`，也必须按journal逐个记录inode/size/hash并与source main一起移入rollback bundle/quarantine；安装candidate前live basename下必须不存在任何旧sidecar。同一filesystem执行可恢复rename序列：live main→recorded `state.db.pre-v2.original`，source sidecars→recorded rollback names，fsync parent并写`old_moved`；candidate→live main，确认无candidate sidecars，fsync parent并写`candidate_live`。每一步先后都由journal phase + recorded fingerprints判定，不能用“文件存在”猜状态。
8. canary用scoped bypass执行write/read/search，随后关闭，checkpoint candidate live DB到`TRUNCATE`并验证WAL frames=0；任何candidate `-wal/-shm` 在此之前都不能删。通过后写`canary_passed`。
9. 移除/归档journal前再次fsync live file+parent，写`complete`；普通participants此后才允许启动。

Free-space gate：可用空间至少 `2 × (state.db + WAL/SHM) + 10 GiB`；不足时拒绝运行。临时目录0700、DB/journal/manifest 0600。

### 5.3 Idempotent resume, abort, and rollback

`fts-status`只读journal与file fingerprints；`fts-resume`/`fts-abort`/`fts-rollback`必须按phase幂等：

- `planned`至`candidate_ready`：live v1未移动；可resume，或`fts-abort`删除已验证属于本operation的candidate/work copy，保留backup，写`aborted`。不得按glob清理。
- `swapping`：根据recorded inode/hash判断是否尚未rename；不得重复移动未知文件。
- `old_moved`：live path可能暂时缺失；`fts-resume`可安装已验证candidate，`fts-rollback`可恢复recorded original。两条路径都fsync parent并更新phase。
- `candidate_live`/canary失败：先停止并关闭canary/participants，checkpoint并关闭candidate；将candidate main及其verified sidecars整体移到quarantine，确认live path没有candidate WAL/SHM，再恢复v1 original/backup并fsync parent。**禁止只换main file而遗留candidate WAL frames。**
- `canary_passed`/`complete`：rollback仍要求maintenance journal、全participant停止、candidate checkpoint/sidecar quarantine和v1 search/write canary。
- journal或fingerprint矛盾：fail closed，输出人工恢复inventory；不得自动选“看起来最新”的DB。
- 失败路径不自动删除consistent backup；由用户在稳定观察期后确认清理。
- v2代码必须能读取/repair v1 backup，rollback不依赖旧binary。

## 6. Phase 1 verification oracle

### 6.1 Unit/integration

至少覆盖：

1. new DB直接创建v2且无 `%_content` shadows；
2. legacy inline v1 startup不自动迁移；
3. v1/v2 trigger repair各自正确；
4. external views与indexes rank-1 integrity；
5. insert/update/delete、role user↔tool transition；
6. unicode/English search parity与snippet；
7. default user/assistant CJK trigram；
8. explicit tool CJK LIKE fallback；
9. no-FTS5/no-trigram fallback；
10. repair/rebuild不把v2降回inline v1；
11. migration interruption、journal phase、resume/abort/rollback every crash boundary；
12. central pre-open guard覆盖`SessionDB`、`_db_opens_cleanly` write probe、repair、doctor fix/checkpoint；marker下read-only status可用但无写；
13. forward/rollback tests把main+WAL+SHM作为一组，覆盖zero-frame sidecar、candidate canary WAL和old_moved中断；
14. main DB row/field digests迁移前后相等（派生FTS对象除外）。

Baseline：

- State/search/prune focused：127 passed；
- `tests/test_hermes_state.py` + malformed repair：318 passed。

### 6.2 Paired search gate

在同一consistent snapshot上运行 old/new paired corpus，固定 denominator：

- default `role_filter=user,assistant`：match message-id set必须一致；top-10 overlap ≥90%，差异必须只来自BM25 corpus statistics并人工抽检aggregate IDs；
- explicit tool English/unicode：match set一致；
- explicit tool CJK：new LIKE与old trigram/LIKE union match set一致；
- snippets都包含match marker/term；
- source/exclude/active/compacted filters与lineage dedupe一致；
- p95 latency分别报告 English、default CJK、tool CJK；不能拿混合平均掩盖慢路径。

任何漏结果、orphan FTS row、rank-1 integrity失败或message digest变化都是NO-GO。

### 6.3 Final full-projection copy benchmark

最终候选在trigram中保留user/assistant完整旧projection（包括assistant `tool_calls`），只排除其他roles。SQLite-consistent snapshot aggregate report：

`/Users/zongxin/clawd/work/state-db-value-tiered-retention/fts-copy-benchmark-full-projection-report.json`

| 指标 | v1 copy | v2 full-projection compact copy |
|---|---:|---:|
| DB file bytes | 6,326,444,032 | 2,534,400,000 |
| FTS objects total | 4,689,182,720 | 922,415,104 |
| unicode FTS（不含trigram） | 1,407,332,352 | 343,187,456 |
| trigram FTS | 3,281,850,368 | 579,227,648 |
| FTS content shadows | 2,140,979,200 | 0 |
| messages rows | 323,015 | 323,015 |
| sessions rows | 2,303 | 2,303 |
| freelist pages | 4,113 | 0 |

- 实际文件减少 **3,792,044,032 bytes（59.94%）**；没有改/删message或session rows。
- unicode docsize rows 323,015；trigram docsize rows 147,233，精确等于user/assistant rows。
- 两套external-content FTS的rank-1 `integrity-check`都通过；compact copy `quick_check=ok`。
- consistent copy约246.9s；FTS migration/rebuild约249.7s；`VACUUM INTO`约63.5s；端到端约849.0s（14.2min）。前一轮在writers更活跃时端到端25.7min，因此正式rollout仍按 **30–40min maintenance budget**。
- reviewer同类tiny spike证明：CJK term仅存在于assistant `tool_calls`时，old/new都命中同一row，new snippet含term，rank-1 integrity通过。
- 两个含正文的0600临时DB在report成功写出后已自动删除，只保留aggregate report；先前visible-only rejected report保留用于设计审计，不作为批准数字。

这证明保留search projection的Phase 1在物理层面可把当前DB从约5.89GiB降到约2.36GiB，释放约3.53GiB；**它只证明size/timing/integrity/row-count，不证明paired-search parity**。Search parity仍必须由§6.2在实现candidate上取得后才能申请live rollout；无需为了即时空间提前启用payload deletion。

## 7. Phase 2 — value-tiered payload retention

### 7.1 Correct clock

正确apply最终需要：

- `messages.compacted_at REAL NULL`；
- `_archive_and_compact_write()` 在active→compacted transition时写同一transaction timestamp；
- 既有 `compacted=1 AND compacted_at IS NULL` rows保守回填为Phase 2 schema migration时间，而不是原message timestamp。

但这些都属于**Phase 2 schema mutation**：不得在Phase 1的`SCHEMA_SQL`、startup reconciliation或兼容代码部署中添加。Phase 1/本轮批准的estimator必须容忍column不存在，输出 `clock_status=unavailable`，不得宣称任何row已达到真实retention age；可附按原timestamp的敏感性分析，但必须标 `non_actionable_upper_bound`。

只有Phase 2另行批准并通过显式schema command后，retention窗口才从 `compacted_at` 起算。

### 7.2 Initial policy（estimator-only default）

- `active=1`：完整保留；
- `active=0, compacted=0` rewind rows：不由compacted retention处理；
- `active=0, compacted=1` 且 `compacted_at > now-14d`：完整保留；
- 超过14天：
  - user visible content、assistant final visible content保留；
  - Codex replay/reasoning字段可进入清理候选，但必须证明 inactive archived rows不被resume/provider replay读取；
  - assistant tool_calls只做 estimator；apply需要结构化保留name/id/status/args hash的schema与paired search审批；
  - tool content不得仅因>16KiB截断。

### 7.3 Durable-handle exemption

每次 estimator/apply必须建立引用集合：扫描active和compacted可搜索rows中的合法 `hermes://session/<sid>/message/<rowid>` handles，并验证target session/id/role/tool_call_id。

- 被引用target：豁免；
- malformed/missing handle：报警，不静默忽略；
- 只有先把raw body原子迁移到checksum可验证的durable artifact，并让resolver readback通过，才能替换DB target；
- artifact迁移必须同时考虑总磁盘：把正文从DB搬到未压缩sidecar不算节省。

Phase 2 apply必须单独审批；本项目初始实现仅允许 `--dry-run` estimator。

## 8. Phase 3 — source-aware session retention

### 8.1 Prerequisites

新增/修正：

- `sessions.last_activity_at`，每次message/session mutation更新；
- source lifecycle finalization tests，解释并收敛异常 `ended_at IS NULL`；
- retention reference/hold registry（至少覆盖 `archived=1`、active gateway binding、parent/child lineage、handoff pending/running、用户pin/bookmark、ledger/recovery reference、未完成background handle）；
- `archived=1` 延续现有“用户主动soft-hide但可restore且保留全部messages”的语义，是不可覆盖的mandatory hold；若未来要改变，必须另做durable pin机制、现有archived rows迁移和UI/API审批；
- deletion age按 `max(ended_at,last_activity_at)`，不得按started_at；
- child lineage不能简单orphan后丢失恢复关系。

### 8.2 Proposed policy（design-only）

- cron/subagent diagnostic ended sessions：候选30天；
- interactive Discord/Telegram/TUI/CLI ended sessions：候选180天（不是90天默认）；
- `tool`/unknown source、ended_at NULL、`archived=1`、任何hold/reference命中：不自动删；
- 首版只出dry-run manifest；整-session auto apply继续默认false。

在reference registry与lifecycle修复完成前，Phase 3不得上线。

## 9. Config and observability

建议新配置，默认安全关闭：

```yaml
sessions:
  fts_schema_target: 2
  payload_retention:
    enabled: false
    dry_run: true
    compacted_after_days: 14
  source_retention:
    enabled: false
```

报告必须包含：schema marker、last successful migration/estimator、candidate/exempt rows、logical bytes、physical bytes、handle exemptions、missing handles、paired-search verdict、backup/candidate paths、rollback readiness。不得记录raw content/query/session IDs。

旧的 `sessions.auto_prune` 不得与新分层机制同时运行；检测到两者同时enabled时fail config check。

## 10. Non-goals

本设计不授权：

- 直接修改live DB；
- 按年龄清空所有tool output；
- 立即启用7天payload retention；
- 立即启用30/90天whole-session prune；
- 在startup自动rebuild/VACUUM；
- 删除Chrome profile、session JSON、Cron output或普通logs；
- 用archive UI flag冒充数据删除；
- 用FTS optimize/VACUUM冒充row/index-content reduction。

## 11. Approval boundary

推荐批准范围：

- **实现 Phase 1 FTS v2 code/tests/migration CLI；只在副本验证，不apply live。**
- **实现 Phase 2 read-only estimator；不得在Phase 1 code deploy中添加`compacted_at`、meta marker或其他live schema plumbing，不清理payload。**
- Phase 3仅保留设计和read-only lifecycle report。

完成实现与copy paired gates后，再提交一张live rollout卡：预计停机时间、实际副本节省、backup/rollback、所有participant停止/恢复步骤。只有再次明确批准才apply live migration。
