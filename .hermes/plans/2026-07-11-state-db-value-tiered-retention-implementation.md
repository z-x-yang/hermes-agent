# State DB Value-Tiered Retention + FTS v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并提交 Phase 1 FTS v2、可恢复 migration CLI 与无 schema 写的 Phase 2 read-only estimator，在副本上取得完整验证证据，但不 apply live `state.db`。

**Architecture:** 新增三个窄模块：`state_db_maintenance.py` 负责外部 phase journal、中央 pre-open guard 与文件指纹；`state_db_fts.py` 负责 v1/v2 schema detection、external-content DDL、repair/rebuild 与 search projection；`state_db_fts_migration.py` 负责 read-only plan/estimator、copy candidate、swap/resume/abort/rollback state machine。`hermes_state.py` 只保留 SessionDB integration 与 search API，`hermes_cli/main.py` 只做 argparse/dispatch。

**Tech Stack:** Python 3.11、stdlib `sqlite3`/`dataclasses`/`hashlib`/`os`/`pathlib`/`shutil`/`subprocess`、SQLite FTS5 external-content、pytest。

**Authoritative spec:** `.hermes/specs/2026-07-11-state-db-value-tiered-retention-fts-redesign.md`

## Global Constraints

- 不连接或修改 live `~/.hermes/state.db`；所有 tests 与 migration smoke 使用 `tmp_path` 或显式0600副本。
- 不添加 `messages.compacted_at`、`sessions.last_activity_at`、新的 live schema column 或启动时 schema/meta backfill。
- legacy inline FTS startup 只读检测为 effective v1；不得自动 migration、rebuild、VACUUM 或写 marker。
- new empty DB 可在同一 create transaction 中创建 FTS v2 与 `state_meta.fts_schema_version=2`。
- FTS v2 unicode 与 user/assistant trigram 都索引完整 `content + tool_name + tool_calls` projection；trigram只排除其他roles。
- maintenance journal 是DB外0600文件，所有 write-capable SQLite connect/checkpoint/write probe 先经过中央guard。
- migration CLI在本任务只对测试DB/副本运行；live apply仍需新的明确批准。
- `archived=1` session不进入任何retention candidate；payload estimator不改schema/meta/data。
- 每个生产行为必须先有真实 RED test，确认因功能缺失而失败，再写最小实现。

---

## File Map

- Create `state_db_maintenance.py`: journal schema、atomic/fsync writes、canonical path、permit、guard、fingerprints、sidecar inventory。
- Create `state_db_fts.py`: v1/v2 schema detection、projection views、external-content tables/triggers、rebuild/integrity helpers。
- Create `state_db_fts_migration.py`: read-only plan/estimator、copy builder、paired verifier、phase state machine、resume/abort/rollback。
- Modify `hermes_state.py`: guard integration、v1/v2 init/repair、search routing。
- Modify `hermes_cli/main.py`: sessions subcommands and dispatch only。
- Modify `hermes_cli/doctor.py`: raw state DB write/checkpoint paths use central guard。
- Create `tests/test_state_db_maintenance.py`。
- Create `tests/test_state_db_fts_v2.py`。
- Create `tests/test_state_db_fts_migration.py`。
- Create `tests/hermes_cli/test_state_db_maintenance_cli.py`。
- Modify `tests/test_hermes_state.py`。
- Modify `tests/test_state_db_malformed_repair.py`。
- Modify `website/docs/user-guide/sessions.md`。
- Modify `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/sessions.md`。

---

### Task 1: External maintenance journal and central pre-open guard

**Files:**
- Create: `state_db_maintenance.py`
- Create: `tests/test_state_db_maintenance.py`

**Interfaces:**
- Produces: `JournalPhase`, `MaintenanceJournal`, `MaintenancePermit`, `MaintenanceBlockedError`。
- Produces: `maintenance_journal_path(db_path: Path) -> Path`。
- Produces: `load_maintenance_journal(db_path: Path) -> MaintenanceJournal | None`。
- Produces: `write_maintenance_journal(db_path: Path, record: MaintenanceJournal) -> None`。
- Produces: `assert_state_db_maintenance_access(db_path: Path, *, write_capable: bool, permit: MaintenancePermit | None = None) -> None`。
- Produces: `issue_maintenance_permit(db_path: Path, operation_id: str, allowed_phases: frozenset[JournalPhase]) -> MaintenancePermit`；此函数不由CLI暴露参数。
- Produces: `fingerprint_path(path: Path) -> dict[str, int | str] | None` and `state_db_file_inventory(db_path: Path) -> dict[str, dict | None]`。

- [ ] **Step 1: Write RED tests for journal durability and permissions**

```python
def test_write_journal_is_0600_and_fsyncs_parent(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    fsynced_modes = []
    real_fsync = os.fsync
    def record(fd):
        fsynced_modes.append(os.fstat(fd).st_mode)
        real_fsync(fd)
    monkeypatch.setattr(os, "fsync", record)
    write_maintenance_journal(db_path, MaintenanceJournal.new("op-1", db_path))
    path = maintenance_journal_path(db_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_state_db_maintenance.py::test_write_journal_is_0600_and_fsyncs_parent -q`
Expected: FAIL because `state_db_maintenance` does not exist。

- [ ] **Step 3: Implement immutable journal records and atomic writer**

```python
class JournalPhase(str, Enum):
    PLANNED = "planned"
    WRITERS_STOPPED = "writers_stopped"
    CHECKPOINTED = "checkpointed"
    BACKUP_READY = "backup_ready"
    CANDIDATE_READY = "candidate_ready"
    SWAPPING = "swapping"
    OLD_MOVED = "old_moved"
    CANDIDATE_LIVE = "candidate_live"
    CANARY_PASSED = "canary_passed"
    COMPLETE = "complete"
    ABORTED = "aborted"
    ROLLED_BACK = "rolled_back"

TERMINAL_PHASES = {JournalPhase.COMPLETE, JournalPhase.ABORTED, JournalPhase.ROLLED_BACK}
```

`write_maintenance_journal()` 必须 temp→file fsync→`os.replace`→parent directory fsync；JSON保存version、operation_id、phase、db/backup/work/candidate paths、fingerprints、timestamps、expected row counts。

- [ ] **Step 4: Write RED tests for guard semantics**

```python
def test_nonterminal_journal_blocks_write_but_allows_read_only(tmp_path):
    db = tmp_path / "state.db"
    write_maintenance_journal(db, MaintenanceJournal.new("op-1", db))
    with pytest.raises(MaintenanceBlockedError, match="fts-status"):
        assert_state_db_maintenance_access(db, write_capable=True)
    assert_state_db_maintenance_access(db, write_capable=False) is None
```

另外覆盖：terminal phase允许普通write；wrong operation id拒绝；permit journal inode/operation/phase不匹配拒绝；canonical/symlink path指向同一journal；malformed journal fail closed。

- [ ] **Step 5: Run RED, implement guard/permit/fingerprint, run GREEN**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_state_db_maintenance.py -q`
Expected before implementation: FAIL at first missing behavior；after implementation: all PASS。

- [ ] **Step 6: Commit**

```bash
git add state_db_maintenance.py tests/test_state_db_maintenance.py
git commit -m "feat: add state db maintenance journal guard"
```

---

### Task 2: Enforce the guard on every current write-capable connection path

**Files:**
- Modify: `hermes_state.py:476-690, SessionDB.__init__`
- Modify: `hermes_cli/doctor.py:1237-1345`
- Modify: `tests/test_state_db_malformed_repair.py`
- Create: `tests/hermes_cli/test_state_db_maintenance_cli.py`

**Interfaces:**
- Consumes: Task 1 guard and permit。
- Produces: raw repair/doctor/status paths with explicit `write_capable` contract。

- [ ] **Step 1: Write RED tests for direct bypasses**

Tests create a nonterminal journal beside a temp DB, then assert:

```python
with pytest.raises(MaintenanceBlockedError):
    _db_opens_cleanly(db_path)  # current implementation performs a write probe
with pytest.raises(MaintenanceBlockedError):
    repair_state_db_schema(db_path)
```

CLI test monkeypatches doctor/session repair DB path and asserts `hermes sessions repair`/doctor fix prints blocked recovery instructions without checkpoint/repair calls。Read-only `fts-status` remains allowed。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_state_db_malformed_repair.py tests/hermes_cli/test_state_db_maintenance_cli.py -q`
Expected: FAIL because direct sqlite paths ignore journal。

- [ ] **Step 3: Add guard calls before connect/checkpoint/write probe**

- `SessionDB.__init__`: call guard before the first writable `sqlite3.connect`。
- `_db_opens_cleanly`: accept `write_probe: bool = True` and optional permit；guard before connect。`write_probe=False` uses `mode=ro` + `PRAGMA query_only=ON` and performs no DDL/DML/checkpoint。
- `repair_state_db_schema`: guard before backup/open/repair/VACUUM。
- doctor fix/checkpoint paths: guard before raw connect or `wal_checkpoint`；doctor read-only diagnostics use query-only connection。

- [ ] **Step 4: Run GREEN and regression subset**

Run:

```bash
PYTHONPATH=. venv/bin/python -m pytest \
  tests/test_state_db_maintenance.py \
  tests/test_state_db_malformed_repair.py \
  tests/hermes_cli/test_state_db_maintenance_cli.py -q
```

Expected: all PASS；no live DB access。

- [ ] **Step 5: Commit**

```bash
git add hermes_state.py hermes_cli/doctor.py tests/test_state_db_malformed_repair.py tests/hermes_cli/test_state_db_maintenance_cli.py
git commit -m "fix: gate direct state db writers during maintenance"
```

---

### Task 3: FTS v2 schema owner, detection, rebuild, and repair

**Files:**
- Create: `state_db_fts.py`
- Create: `tests/test_state_db_fts_v2.py`
- Modify: `hermes_state.py:809-1134, 1340-1490`
- Modify: `tests/test_hermes_state.py`
- Modify: `tests/test_state_db_malformed_repair.py`

**Interfaces:**
- Produces: `FtsSchemaKind = Literal["missing", "v1_inline", "v2_external", "inconsistent"]`。
- Produces: `detect_fts_schema(conn: sqlite3.Connection) -> FtsSchemaKind`。
- Produces: `create_fts_v2(conn)`, `create_fts_v1(conn)`, `rebuild_fts(conn, kind)`, `integrity_check_fts_v2(conn)`。
- Produces constants `FULL_PROJECTION_SQL`, `FTS_V2_VIEW_SQL`, `FTS_V2_TABLE_SQL`, `FTS_V2_TRIGGER_SQL`。

- [ ] **Step 1: Write RED tests for new DB vs legacy DB behavior**

```python
def test_new_empty_db_uses_v2_without_content_shadows(tmp_path):
    db = SessionDB(tmp_path / "new.db")
    names = schema_names(db._conn)
    assert "messages_fts_unicode_content_v2" in names
    assert "messages_fts_content" not in names
    assert db.get_meta("fts_schema_version") == "2"

def test_legacy_inline_db_opens_without_schema_or_meta_write(tmp_path):
    path = make_v1_fixture(tmp_path)
    before = schema_and_meta_digest(path)
    SessionDB(path).close()
    assert schema_and_meta_digest(path) == before
```

Also cover external tables + missing marker => fail closed；v1 missing trigger repairs with v1 DDL；v2 missing trigger repairs with v2 DDL；no FTS5 fallback。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_state_db_fts_v2.py -q`
Expected: FAIL because v2 module/schema does not exist。

- [ ] **Step 3: Implement projection views and external-content DDL**

Use exactly:

```sql
CREATE VIEW messages_fts_unicode_content_v2 AS
SELECT id, coalesce(content,'') || ' ' || coalesce(tool_name,'') || ' ' || coalesce(tool_calls,'') AS content
FROM messages;

CREATE VIEW messages_fts_trigram_content_v2 AS
SELECT id, coalesce(content,'') || ' ' || coalesce(tool_name,'') || ' ' || coalesce(tool_calls,'') AS content
FROM messages WHERE role IN ('user','assistant');
```

Both FTS tables use `content='<view>'`, `content_rowid='id'`; trigram uses `tokenize='trigram'`。Triggers use identical old/new projection；trigram delete/insert is conditional on role。

- [ ] **Step 4: Implement pure detection and schema-aware init/repair**

- New empty DB: create v2 + marker 2 in same transaction。
- Existing inline DB with no marker: effective v1 in memory, no schema/meta write。
- Existing v2 + marker2: verify views/table SQL/trigger owner。
- External/mixed schema without marker2: raise explicit inconsistency；do not guess。
- `_rebuild_fts_indexes()` dispatches by effective schema；v2 uses `rebuild` + rank-1 integrity。

- [ ] **Step 5: Run GREEN and main FTS regressions**

Run:

```bash
PYTHONPATH=. venv/bin/python -m pytest \
  tests/test_state_db_fts_v2.py \
  tests/test_state_db_malformed_repair.py \
  tests/test_hermes_state.py -k 'fts or trigram or schema or repair' -q
```

Expected: all PASS。

- [ ] **Step 6: Commit**

```bash
git add state_db_fts.py hermes_state.py tests/test_state_db_fts_v2.py tests/test_hermes_state.py tests/test_state_db_malformed_repair.py
git commit -m "feat: add external-content fts v2 schema"
```

---

### Task 4: Search-routing parity and metadata-aware snippets

**Files:**
- Modify: `hermes_state.py:4375-4552`
- Modify: `tests/test_hermes_state.py:1490-1700, 4090-4160`

**Interfaces:**
- Consumes: Task 3 v2 schema detection。
- Produces: `search_messages()` routing that selects trigram only when `role_filter` is a nonempty subset of `{"user", "assistant"}`。
- Produces: `_search_projection_sql(alias: str = "m") -> str` shared by LIKE match/snippet logic。

- [ ] **Step 1: Write RED metadata-only CJK parity tests**

```python
def test_v2_trigram_finds_cjk_only_in_assistant_tool_calls(db_v2):
    add_message(role="assistant", content="visible", tool_calls='{"q":"工具调用关键词"}')
    rows = db_v2.search_messages("调用关", role_filter=["user", "assistant"])
    assert [r["role"] for r in rows] == ["assistant"]
    assert "调用关" in strip_markers(rows[0]["snippet"])

def test_tool_cjk_uses_like_and_preserves_filters(db_v2):
    # One active tool hit, one inactive compacted hit, one excluded-source hit.
    rows = db_v2.search_messages("工具关", role_filter=["tool"], include_inactive=False)
    assert only_expected_active_tool(rows)
```

Also cover `role_filter=None` all-role LIKE；short CJK；escaped `%/_`；source/exclude filters；v1/v2 same message-id set。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_hermes_state.py -k 'metadata_only_cjk or tool_cjk_uses_like' -q`
Expected: FAIL because current routing indexes all roles and LIKE snippet only uses `content`/misses active-compacted filter。

- [ ] **Step 3: Implement routing and shared projection snippet**

- user/assistant-only, ≥3 CJK: trigram MATCH。
- tool/system/session_meta/all-role: escaped LIKE over full projection。
- LIKE SQL includes `m.active=1` unless `include_inactive=True`；when inactive included, preserves compacted/source filters。
- Build snippet from the same projection, while returned `content` remains original `m.content`。

- [ ] **Step 4: Run GREEN and session_search regressions**

Run:

```bash
PYTHONPATH=. venv/bin/python -m pytest \
  tests/test_hermes_state.py \
  tests/tools/test_session_search.py -k 'search or trigram or cjk or role' -q
```

Expected: all PASS。

- [ ] **Step 5: Commit**

```bash
git add hermes_state.py tests/test_hermes_state.py
git commit -m "fix: preserve search parity with selective trigram"
```

---

### Task 5: Read-only migration plan/status and payload estimator

**Files:**
- Create: `state_db_fts_migration.py`
- Create: `tests/test_state_db_fts_migration.py`

**Interfaces:**
- Produces: `MigrationPlan` dataclass and `plan_fts_migration(db_path: Path) -> MigrationPlan`。
- Produces: `RetentionEstimate` and `estimate_payload_retention(db_path: Path, age_days: tuple[int, ...] = (0,1,3,7,14)) -> RetentionEstimate`。
- Produces: `status_fts_migration(db_path: Path) -> dict`。
- All three open `file:<path>?mode=ro`, execute `PRAGMA query_only=ON`, and never issue checkpoint/write probe/meta/schema writes。

- [ ] **Step 1: Write RED immutability tests**

```python
def test_plan_and_estimator_do_not_mutate_db(tmp_path):
    path = make_v1_fixture(tmp_path)
    before = file_and_schema_digest(path)
    plan_fts_migration(path)
    estimate = estimate_payload_retention(path)
    assert estimate.clock_status == "unavailable"
    assert estimate.rows_by_age_basis == "non_actionable_upper_bound"
    assert file_and_schema_digest(path) == before
```

Also assert archived sessions excluded from any session summary；valid/missing persisted handles counted but no URI/session id emitted；free-space requirement is `2 * (db+wal+shm) + 10 GiB`；malformed journal shown without repair。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_state_db_fts_migration.py -k 'plan or estimator or status' -q`
Expected: FAIL because migration module does not exist。

- [ ] **Step 3: Implement aggregate-only read-only reports**

`MigrationPlan` fields: schema kind/marker、DB/WAL/SHM bytes、free bytes、required free bytes、message/session counts、FTS object bytes、writer/maintenance status、can_apply boolean/reasons。

`RetentionEstimate` fields: `clock_status="unavailable"` when no column；age sensitivity labelled `non_actionable_upper_bound`；field logical chars；handle targets/exemptions/missing counts；no raw content/query/session IDs。

- [ ] **Step 4: Run GREEN**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_state_db_fts_migration.py -k 'plan or estimator or status' -q`
Expected: all PASS。

- [ ] **Step 5: Commit**

```bash
git add state_db_fts_migration.py tests/test_state_db_fts_migration.py
git commit -m "feat: add read-only state db migration estimator"
```

---

### Task 6: Copy candidate builder and paired verifier

**Files:**
- Modify: `state_db_fts_migration.py`
- Modify: `tests/test_state_db_fts_migration.py`

**Interfaces:**
- Produces: `build_v2_candidate(source: Path, work_dir: Path, journal: MaintenanceJournal, permit: MaintenancePermit) -> CandidateReport`。
- Produces: `verify_v2_candidate(source_copy: Path, candidate: Path, corpus: Sequence[SearchCase]) -> VerificationReport`。
- Produces: `field_digest(conn) -> str` excluding FTS/views/triggers only。

- [ ] **Step 1: Write RED candidate-copy test**

Build a v1 fixture with user/assistant/tool rows, tool_calls-only CJK term, active/inactive/compacted states and archived session。Call builder under a valid permit, then assert:

```python
assert report.source_message_count == report.candidate_message_count
assert report.source_session_count == report.candidate_session_count
assert report.field_digest_equal
assert report.unicode_integrity == "passed_rank1"
assert report.trigram_integrity == "passed_rank1"
assert report.quick_check == "ok"
assert not report.candidate_wal_exists
assert not report.candidate_shm_exists
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_state_db_fts_migration.py -k 'candidate_copy' -q`
Expected: FAIL because builder is absent。

- [ ] **Step 3: Implement backup→v2→VACUUM INTO candidate path**

- SQLite backup API, not raw copy。
- Source never receives DDL/DML。
- Work copy creates v2 via Task 3 owner；sets marker2 only inside candidate。
- Rank-1 integrity、quick_check、field digest、row counts。
- `VACUUM INTO` candidate；close/checkpoint；0600 mode；fsync file。
- On failure leave journal and aggregate inventory；do not delete unknown paths。

- [ ] **Step 4: Write RED paired-search test and implement verifier**

Corpus includes English/unicode、default CJK、tool_calls-only CJK、explicit tool CJK、short CJK。For each case compare full match-id set; report top-10 overlap and p50/p95 latency。Output contains only corpus case IDs and aggregate counts/latency, not message text/session IDs。

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_state_db_fts_migration.py -k 'paired_search' -q`
Expected before verifier: FAIL；after implementation: PASS with exact set equality。

- [ ] **Step 5: Commit**

```bash
git add state_db_fts_migration.py tests/test_state_db_fts_migration.py
git commit -m "feat: build and verify fts v2 candidates"
```

---

### Task 7: Crash-durable apply/resume/abort/rollback state machine

**Files:**
- Modify: `state_db_fts_migration.py`
- Modify: `tests/test_state_db_fts_migration.py`

**Interfaces:**
- Produces: `apply_fts_migration(db_path: Path) -> MigrationResult`。
- Produces: `resume_fts_migration(db_path: Path)`, `abort_fts_migration(db_path: Path)`, `rollback_fts_migration(db_path: Path, backup: Path | None = None)`。
- Produces: `find_live_state_db_users(db_path: Path) -> LivenessReport`；`lsof` unavailable/ambiguous => fail closed。

- [ ] **Step 1: Write RED liveness/checkpoint tests**

- live foreign holder => apply refuses before `planned`。
- journal written => second liveness check catches old writer。
- `wal_checkpoint(TRUNCATE)` must return `(0,0,0)` before backup；busy/nonzero fails without advancing phase。
- Tests inject `lsof` output/parser; do not inspect real processes。

- [ ] **Step 2: Run RED, implement pre-swap phases**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_state_db_fts_migration.py -k 'liveness or checkpoint or backup_ready' -q`
Expected before implementation: FAIL；after: PASS。

- [ ] **Step 3: Write RED crash-boundary table tests**

Parametrize phases `planned`, `checkpointed`, `backup_ready`, `candidate_ready`, `swapping`, `old_moved`, `candidate_live`, `canary_passed`。For each phase create exact main/WAL/SHM/backup/candidate fingerprint inventory and assert one allowed idempotent action；unknown fingerprint raises and preserves files。

- [ ] **Step 4: Implement recoverable rename and sidecar handling**

- Move source main plus even zero-frame WAL/SHM to recorded rollback names before candidate install。
- fsync parent after each rename and journal transition。
- Candidate live basename must have no stale source sidecars。
- Canary uses scoped permit, closes, checkpoints to `(0,0,0)`。
- Rollback quarantines candidate main/WAL/SHM as one recorded bundle before restoring v1。
- `abort` only pre-swap；post-swap requires rollback。

- [ ] **Step 5: Run GREEN for full state-machine suite**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/test_state_db_fts_migration.py -k 'apply or resume or abort or rollback or crash or sidecar' -q`
Expected: all PASS。

- [ ] **Step 6: Commit**

```bash
git add state_db_fts_migration.py tests/test_state_db_fts_migration.py
git commit -m "feat: add recoverable fts migration state machine"
```

---

### Task 8: CLI registration, output contracts, and docs

**Files:**
- Modify: `hermes_cli/main.py:13374-13708`
- Create: `tests/hermes_cli/test_state_db_maintenance_cli.py` (extend Task 2 file)
- Modify: `website/docs/user-guide/sessions.md`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/sessions.md`

**Interfaces:**
- Consumes: Task 5-7 functions。
- Produces commands: `fts-plan`, `fts-status`, `fts-migrate --apply --yes`, `fts-resume --yes`, `fts-abort --yes`, `fts-rollback [backup] --yes`, `retention-estimate [--json]`。
- CLI never accepts `operation_id`/permit token。

- [ ] **Step 1: Write RED parser/dispatch tests**

Assert each parser action exists, destructive commands require `--yes`/TTY confirmation, JSON estimator has stable keys, maintenance-blocked errors are nonzero and include recovery commands, and `retention-estimate` leaves DB digest unchanged。

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=. venv/bin/python -m pytest tests/hermes_cli/test_state_db_maintenance_cli.py -q`
Expected: FAIL because actions are absent。

- [ ] **Step 3: Implement thin argparse/dispatch**

`main.py` imports migration functions only inside selected action。Read-only actions run before ordinary `SessionDB()` construction。Apply/resume/abort/rollback prompt before journal/file mutation；all return nonzero on blocked/failure and concise stdout/stderr。

- [ ] **Step 4: Update English/Chinese docs**

Document: v1/v2 status、plan first、30–40min maintenance budget、backup/rollback、live apply requires stopped writers and explicit confirmation、estimator is non-actionable without `compacted_at`、no payload deletion in this release。

- [ ] **Step 5: Run GREEN and CLI help smoke**

```bash
PYTHONPATH=. venv/bin/python -m pytest tests/hermes_cli/test_state_db_maintenance_cli.py -q
venv/bin/python -m hermes_cli.main sessions --help
```

Expected: tests PASS；help lists all seven new actions。

- [ ] **Step 6: Commit**

```bash
git add hermes_cli/main.py tests/hermes_cli/test_state_db_maintenance_cli.py website/docs/user-guide/sessions.md website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/sessions.md
git commit -m "feat: expose safe state db migration commands"
```

---

### Task 9: Integrated verification, copy smoke, and branch evidence

**Files:**
- Modify only if a failing regression exposes a scoped defect in files above。
- Evidence: `work/state-db-value-tiered-retention/implementation-evidence.md` outside repo, aggregate-only。

**Interfaces:**
- Consumes all earlier tasks。
- Produces fresh test/build/copy evidence without live mutation。

- [ ] **Step 1: Run complete targeted suite**

```bash
PYTHONPATH=. venv/bin/python -m pytest \
  tests/test_state_db_maintenance.py \
  tests/test_state_db_fts_v2.py \
  tests/test_state_db_fts_migration.py \
  tests/test_state_db_malformed_repair.py \
  tests/test_hermes_state.py \
  tests/hermes_state \
  tests/tools/test_session_search.py \
  tests/hermes_cli/test_state_db_maintenance_cli.py \
  tests/hermes_cli/test_sessions_delete.py \
  tests/gateway/test_session_store_prune.py \
  tests/gateway/test_session_store_stale_prune.py -q
```

Expected: 0 failures。

- [ ] **Step 2: Static and diff checks**

```bash
venv/bin/python -m py_compile state_db_maintenance.py state_db_fts.py state_db_fts_migration.py hermes_state.py hermes_cli/main.py hermes_cli/doctor.py
git diff --check main...HEAD
git status --short --untracked-files=all
```

Expected: compile/diff check rc=0；worktree clean after commits。

- [ ] **Step 3: Run an aggregate-only copy smoke**

Use SQLite backup API to create a 0600 snapshot under `~/.hermes/tmp/state-db-v2-implementation-smoke-*`；run only `fts-plan`、candidate builder、paired verifier。Do not call apply/swap on live path。Require:

- source/candidate message/session counts and field digest equal；
- both rank-1 integrity checks pass；
- quick_check=ok；
- default user/assistant and explicit tool paired match sets equal for fixed corpus；
- physical result is consistent with approved full-projection benchmark: source `6,326,444,032` bytes, candidate `2,534,400,000` bytes, saving `3,792,044,032` bytes (`59.94%`), allowing only explainable concurrent growth/compression drift；Unicode docsize equals all messages and trigram docsize equals exact user/assistant count；
- no temp DB remains after aggregate report readback；
- live DB inode/size/schema digest unchanged except ordinary concurrent writer growth explicitly separated from this process。

- [ ] **Step 4: Independent final audit within review ceiling**

Codex review budget for this change was exhausted by two design passes。Do not start a third Codex pass。Use one read-only Hermes Explore reviewer on committed `main...HEAD` to check spec traceability, guard coverage, crash state machine, search parity and no-live-mutation evidence；main agent must reproduce any claimed blocker before patching。

- [ ] **Step 5: Write/readback aggregate evidence and checkpoint ledger**

`implementation-evidence.md` includes commits、test commands/counts、copy sizes/timings、paired denominator/verdict、temp cleanup、known limits、explicit statement `live migration not applied`。Read back the artifact and run strict ledger validation。

- [ ] **Step 6: Stop before live rollout**

Do not merge/apply migration to live DB in this plan。Present implementation evidence and request a separate live-rollout approval with maintenance window、all-participant stop/restart、backup path、rollback and post-swap canary。
