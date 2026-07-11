# Worktree 安全归档与 Hermes 空间哨兵 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不删除 dirty/locked/活跃 worktree、未合并 branch、会话历史或 Chrome profile 的前提下，上线 14 天去重预警、30 天 clean checkout 安全归档与 weekly Hermes 空间哨兵。

**Architecture:** `scripts/worktree_doctor.py` 负责确定性 worktree 分类、重验与回收；`scripts/worktree_doctor_cron.py` 负责 no-agent stdout UX、去重状态和有界恢复记录。`scripts/hermes_storage_sentinel.py` 独立测量 Hermes 目录，只清理已结束且超过 30 天的 `process_logs/proc_*`，其余大项只做阈值/增长报警。live Cron 继续调用 `~/.hermes/scripts/` 下的实体脚本，repo 是逻辑与测试的 canonical source。

**Tech Stack:** Python 3 标准库、Git CLI、macOS `/usr/sbin/lsof`、pytest、Hermes no-agent Cron。

## Global Constraints

- Warning threshold = 14 days；archive threshold = 30 days。
- `ARCHIVABLE` 只删除 checkout，必须保留 branch 与 pinned HEAD。
- dirty、locked、process-cwd occupied、状态/进程检查失败时 fail closed。
- `MERGED` 保留现有 squash-safe、moving-ref-safe checkout + branch 回收语义。
- 不使用 `git worktree remove --force`。
- 同一 `(branch, HEAD, class/reason)` 不重复通知。
- action manifest 最多 1000 条；状态文件与 manifest 均原子写。
- 空间哨兵唯一自动清理对象：已结束、超过 30 天、未在 `processes.json` 活跃集合中的 `process_logs/proc_*`。
- 不删除普通日志、Cron output、sessions、`state.db`、Chrome profile、项目、模型、数据、环境或备份。
- live script-only Cron：healthy stdout 为空；非空 stdout 必须是可直接发给人的简短终态。
- 所有行为变更严格 RED → GREEN → REFACTOR；不新增第三方依赖。

---

### Task 1: Worktree 活动时间、locked 与 stale 分类

**Files:**
- Modify: `scripts/worktree_doctor.py`
- Modify: `tests/scripts/test_worktree_doctor.py`

**Interfaces:**
- Consumes: existing `_git`, `_list_worktrees`, `_ahead_behind`, `_is_merged`, `Verdict`。
- Produces:
  - `_changed_paths(path: str) -> list[Path]`
  - `_activity_age_days(branch: str, path: str) -> tuple[float, bool]`
  - `Verdict.cls ∈ {ACTIVE, STALE, STALE_DIRTY, ARCHIVABLE, MERGED, SKIP}`
  - parsed `wt["locked"]: str | bool`
  - CLI `--stale-days` default 14 and `--archive-days` default 30 with `archive_days >= stale_days > 0` validation。

- [ ] **Step 1: Add RED fixtures for activity and classification**

Extend `_git` test helpers with deterministic commit dates and add tests equivalent to:

```python
def test_clean_unmerged_29_days_is_stale(repo_with_worktree, monkeypatch):
    doctor, wt = repo_with_worktree(age_days=29, dirty=False, merged=False)
    monkeypatch.setattr(doctor, "time", FakeTime(NOW))
    verdict = doctor._classify(wt, stale_days=14, archive_days=30)
    assert verdict.cls == "STALE"


def test_clean_unmerged_31_days_is_archivable(repo_with_worktree, monkeypatch):
    doctor, wt = repo_with_worktree(age_days=31, dirty=False, merged=False)
    monkeypatch.setattr(doctor, "time", FakeTime(NOW))
    verdict = doctor._classify(wt, stale_days=14, archive_days=30)
    assert verdict.cls == "ARCHIVABLE"


def test_recent_uncommitted_mtime_resets_activity(repo_with_worktree, monkeypatch):
    doctor, wt = repo_with_worktree(age_days=31, dirty=True, dirty_mtime_days=1)
    monkeypatch.setattr(doctor, "time", FakeTime(NOW))
    verdict = doctor._classify(wt, stale_days=14, archive_days=30)
    assert verdict.cls == "ACTIVE"


def test_old_dirty_worktree_is_stale_dirty(repo_with_worktree, monkeypatch):
    doctor, wt = repo_with_worktree(age_days=31, dirty=True, dirty_mtime_days=31)
    monkeypatch.setattr(doctor, "time", FakeTime(NOW))
    verdict = doctor._classify(wt, stale_days=14, archive_days=30)
    assert verdict.cls == "STALE_DIRTY"


def test_locked_worktree_is_skip(repo_with_worktree):
    doctor, wt = repo_with_worktree(age_days=31, dirty=False, locked="manual hold")
    assert doctor._classify(wt, 14, 30).cls == "SKIP"
```

Also add a parser test using real `git worktree lock --reason "manual hold" <path>` and assert `_list_worktrees()` captures `locked`.

- [ ] **Step 2: Run focused RED tests**

Run:

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/scripts/test_worktree_doctor.py \
  -k '29_days or 31_days or uncommitted_mtime or stale_dirty or locked' -q
```

Expected: FAIL because `_classify` lacks `archive_days`, locked parsing, activity mtime and the new classes.

- [ ] **Step 3: Implement deterministic activity helpers**

Use Git-native NUL-delimited paths rather than parsing human porcelain rename text:

```python
def _changed_paths(path: str) -> list[Path]:
    rels: set[str] = set()
    for args in (
        ("diff", "--name-only", "-z"),
        ("diff", "--cached", "--name-only", "-z"),
        ("ls-files", "--others", "--exclude-standard", "-z"),
    ):
        raw = _git(*args, cwd=path)
        rels.update(x for x in raw.split("\0") if x)
    return [Path(path) / rel for rel in sorted(rels)]


def _activity_age_days(branch: str, path: str) -> tuple[float, bool]:
    commit_ts = float(_git("log", "-1", "--format=%ct", branch))
    dirty = bool(_git("status", "--porcelain", "-z", cwd=path))
    latest = commit_ts
    if dirty:
        for changed in _changed_paths(path):
            if changed.exists():
                latest = max(latest, changed.stat().st_mtime)
    return (time.time() - latest) / 86400, dirty
```

Parse both `locked` and `locked <reason>` lines in `_list_worktrees()`. Refactor `_classify(wt, stale_days, archive_days)` so protected/prunable/locked checks occur before any checkout command, dirty recent work is `ACTIVE`, old dirty is `STALE_DIRTY`, clean merged is `MERGED`, clean age ≥ archive threshold is `ARCHIVABLE`, then `STALE`/`ACTIVE`.

Add argparse validation:

```python
if args.stale_days <= 0 or args.archive_days < args.stale_days:
    ap.error("require archive-days >= stale-days > 0")
```

- [ ] **Step 4: Run Task 1 GREEN tests and regression**

Run:

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest tests/scripts/test_worktree_doctor.py -q
/Users/zongxin/.hermes/hermes-agent/venv/bin/python -m py_compile scripts/worktree_doctor.py
git diff --check
```

Expected: all worktree doctor tests PASS, including the existing prunable regression.

- [ ] **Step 5: Commit Task 1**

```bash
git status --short
git diff --stat
git add -- scripts/worktree_doctor.py tests/scripts/test_worktree_doctor.py
git diff --cached --stat
git commit -m "feat: classify inactive worktrees safely"
```

---

### Task 2: 30 天 checkout-only 安全归档与 race/process gates

**Files:**
- Modify: `scripts/worktree_doctor.py`
- Modify: `tests/scripts/test_worktree_doctor.py`

**Interfaces:**
- Consumes: Task 1 `ARCHIVABLE`, `Verdict.head`, `_activity_age_days`, `_list_worktrees`。
- Produces:
  - `_worktree_has_live_cwd(path: str) -> bool` using `/usr/sbin/lsof -a -d cwd +D <path> -F p`；return code 0 = occupied, 1 = no match, other = error。
  - `_archive(v: Verdict, archive_days: int) -> None` revalidating OID, clean status, age, lock, live cwd and checkout validity。
  - CLI `--archive` action。
  - JSON keys `archived`, `archivable`, `stale_dirty` while retaining `reaped`, `skipped`, `merged`, `stale`, `active`。

- [ ] **Step 1: Add RED archive tests**

Add behavior tests equivalent to:

```python
def test_archive_removes_checkout_but_keeps_branch_and_head(temp_repo, monkeypatch):
    doctor, verdict = make_archivable(temp_repo, age_days=31)
    old_head = _git(temp_repo, "rev-parse", verdict.branch)
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _p: False)
    doctor._archive(verdict, archive_days=30)
    assert not Path(verdict.path).exists()
    assert _git(temp_repo, "rev-parse", verdict.branch) == old_head


def test_archive_skips_when_live_cwd_detected(temp_repo, monkeypatch):
    doctor, verdict = make_archivable(temp_repo, age_days=31)
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _p: True)
    with pytest.raises(RuntimeError, match="active process cwd"):
        doctor._archive(verdict, 30)
    assert Path(verdict.path).exists()


def test_archive_fails_closed_when_process_probe_errors(temp_repo, monkeypatch):
    doctor, verdict = make_archivable(temp_repo, age_days=31)
    monkeypatch.setattr(
        doctor, "_worktree_has_live_cwd", MagicMock(side_effect=RuntimeError("lsof failed"))
    )
    with pytest.raises(RuntimeError, match="lsof failed"):
        doctor._archive(verdict, 30)
    assert Path(verdict.path).exists()


def test_archive_skips_when_head_moves_after_classification(temp_repo, monkeypatch):
    doctor, verdict = make_archivable(temp_repo, age_days=31)
    commit_on_branch(temp_repo, verdict.branch, "new work")
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _p: False)
    with pytest.raises(RuntimeError, match="被改动"):
        doctor._archive(verdict, 30)
    assert Path(verdict.path).exists()
```

- [ ] **Step 2: Run focused RED tests**

Run:

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest tests/scripts/test_worktree_doctor.py \
  -k 'archive_' -q
```

Expected: FAIL because `_archive`, process probe and JSON archive surface do not exist.

- [ ] **Step 3: Implement process probe and revalidated archive**

Implement `lsof` semantics without a fallback:

```python
def _worktree_has_live_cwd(path: str) -> bool:
    proc = subprocess.run(
        ["/usr/sbin/lsof", "-a", "-d", "cwd", "+D", path, "-F", "p"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return bool(proc.stdout.strip())
    if proc.returncode == 1:
        return False
    raise RuntimeError(f"lsof cwd probe failed ({proc.returncode}): {proc.stderr.strip()}")
```

Before `git worktree remove`, `_archive` must locate the current path entry from `_list_worktrees`, reject missing/prunable/locked entries, compare current branch OID to `v.head`, require `_activity_age_days(...).dirty is False`, require age ≥ threshold, and require no live cwd. Use unforced `_git("worktree", "remove", v.path)` and do not call `update-ref`.

In `main()`, apply `_archive` only when `--archive` and `v.cls == "ARCHIVABLE"`; catch only expected race/occupied `CalledProcessError`/`RuntimeError` into `skipped`, preserving unexpected fail-fast behavior elsewhere.

- [ ] **Step 4: Run Task 2 GREEN and full worktree tests**

Run:

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest tests/scripts/test_worktree_doctor.py -q
/Users/zongxin/.hermes/hermes-agent/venv/bin/python scripts/worktree_doctor.py --stale-days 14 --archive-days 30 --json >/tmp/worktree-doctor-plan-smoke.json
python -m json.tool /tmp/worktree-doctor-plan-smoke.json >/dev/null
/Users/zongxin/.hermes/hermes-agent/venv/bin/python -m py_compile scripts/worktree_doctor.py
git diff --check
```

Expected: tests PASS; real smoke is valid JSON and does not mutate because `--archive`/`--reap` are absent.

- [ ] **Step 5: Commit Task 2**

```bash
git add -- scripts/worktree_doctor.py tests/scripts/test_worktree_doctor.py
git diff --cached --stat
git commit -m "feat: archive inactive worktree checkouts"
```

---

### Task 3: Worktree Cron 去重、恢复记录与 no-agent UX

**Files:**
- Create: `scripts/worktree_doctor_cron.py`
- Create: `tests/scripts/test_worktree_doctor_cron.py`

**Interfaces:**
- Consumes: active repo `scripts/worktree_doctor.py --reap --archive --archive-manifest <path> --stale-days 14 --archive-days 30 --json`。
- Produces:
  - `main(argv: list[str] | None = None) -> int`
  - state file default `~/.hermes/worktree_doctor_state.json`
  - forwards bounded manifest path `~/.hermes/worktree_doctor_actions.json` to doctor；doctor is the only manifest writer and persists `pending` before checkout removal。
  - atomic dedupe state via temp file + `os.replace`。
  - stdout only for new stale/stale-dirty/skipped signatures or archive actions；reaped-only diagnostics to stderr。

- [ ] **Step 1: Add RED wrapper tests**

Use temp doctor executables returning controlled JSON and temp state/manifest paths:

```python
def test_same_stale_signature_alerts_once(tmp_path, capsys):
    cron = load_cron_module()
    report = {"stale": [stale("b", "abc")], "stale_dirty": [], "archived": [],
              "reaped": [], "skipped": []}
    run_cron(cron, tmp_path, report)
    assert "b" in capsys.readouterr().out
    run_cron(cron, tmp_path, report)
    assert capsys.readouterr().out == ""


def test_head_or_class_change_realerts(tmp_path, capsys):
    cron = load_cron_module()
    run_cron(cron, tmp_path, {"stale": [stale("b", "abc")], ...})
    capsys.readouterr()
    run_cron(cron, tmp_path, {"stale_dirty": [stale("b", "def")], ...})
    assert "b" in capsys.readouterr().out


def test_archived_report_prints_restore_without_rewriting_manifest(tmp_path, capsys):
    cron = load_cron_module()
    run_cron(cron, tmp_path, {"archived": [archived("b", "abc", "/old")], ...})
    assert "git worktree add /old b" in capsys.readouterr().out
```

Doctor tests own the manifest contract: `pending` is written before deletion, result becomes `archived|skipped`, records are capped at 1000, and a failed post-action write leaves the pending recovery record durable. Also cover doctor nonzero exit, invalid JSON, reaped-only stderr, unchanged skipped-reason dedupe, manifest forwarding, and atomic state replacement.

- [ ] **Step 2: Run RED wrapper tests**

Run:

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest tests/scripts/test_worktree_doctor_cron.py -q
```

Expected: FAIL because the repo source wrapper does not exist.

- [ ] **Step 3: Implement wrapper**

Use CLI overrides for tests while keeping production defaults:

```python
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--doctor", type=Path, default=DEFAULT_DOCTOR)
    p.add_argument("--state", type=Path, default=DEFAULT_STATE)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return p.parse_args(argv)
```

The subprocess command is exact and fail-fast:

```python
[sys.executable, str(args.doctor), "--reap", "--archive",
 "--archive-manifest", str(args.manifest),
 "--stale-days", "14", "--archive-days", "30", "--json"]
```

Construct signatures from stable fields only: `branch`, `head`, class, and skipped reason. Write current state only after valid report processing. The wrapper preflights and forwards the manifest but never writes it; doctor atomically owns the bounded `pending → archived|skipped` action lifecycle. Never include raw subprocess traceback in normal stdout; failure returns nonzero with concise stderr.

- [ ] **Step 4: Run Task 3 GREEN and script checks**

Run:

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest tests/scripts/test_worktree_doctor_cron.py -q
/Users/zongxin/.hermes/hermes-agent/venv/bin/python -m py_compile scripts/worktree_doctor_cron.py
git diff --check
```

Expected: all wrapper tests PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add -- scripts/worktree_doctor_cron.py tests/scripts/test_worktree_doctor_cron.py
git diff --cached --stat
git commit -m "feat: bound worktree cron alerts and recovery records"
```

---

### Task 4: Weekly Hermes 空间哨兵

**Files:**
- Create: `scripts/hermes_storage_sentinel.py`
- Create: `tests/scripts/test_hermes_storage_sentinel.py`

**Interfaces:**
- Produces:
  - `collect_sizes(home: Path) -> dict[str, int]`
  - `load_active_process_ids(processes_json: Path) -> set[str]` reading top-level list `session_id`
  - `prune_finished_process_logs(root: Path, active_ids: set[str], cutoff: float, dry_run: bool) -> PruneResult`
  - `evaluate_alerts(current: Snapshot, previous: dict | None) -> list[str]`
  - `main(argv: list[str] | None = None) -> int`
  - defaults: 30-day process-log retention；50 MiB reclaim notification；thresholds from approved spec；weekly state at `~/.hermes/storage_sentinel_state.json`。

- [ ] **Step 1: Add RED cleanup safety tests**

```python
def test_prunes_only_finished_old_process_logs(tmp_path):
    root = tmp_path / "process_logs"
    old_finished = proc_dir(root, "proc_old", age_days=31, exit_code=True)
    old_unknown = proc_dir(root, "proc_unknown", age_days=31, exit_code=False)
    new_finished = proc_dir(root, "proc_new", age_days=1, exit_code=True)
    result = prune_finished_process_logs(root, set(), cutoff=NOW - 30 * DAY, dry_run=False)
    assert not old_finished.exists()
    assert old_unknown.exists()
    assert new_finished.exists()
    assert result.deleted_count == 1


def test_active_checkpoint_id_is_never_pruned(tmp_path):
    old_finished = proc_dir(tmp_path, "proc_live", age_days=31, exit_code=True)
    prune_finished_process_logs(tmp_path, {"proc_live"}, NOW - 30 * DAY, False)
    assert old_finished.exists()


def test_symlink_and_escape_are_never_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "process_logs" / "proc_link"
    link.symlink_to(outside, target_is_directory=True)
    prune_finished_process_logs(link.parent, set(), NOW, False)
    assert outside.exists()
```

Cover malformed `processes.json` as fail-closed/nonzero, missing root as empty success, dry-run, and newest-descendant mtime rather than directory mtime alone.

- [ ] **Step 2: Run cleanup RED tests**

Run:

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest tests/scripts/test_hermes_storage_sentinel.py \
  -k 'prune or active or symlink or dry_run or malformed' -q
```

Expected: FAIL because the sentinel does not exist.

- [ ] **Step 3: Implement bounded path-safe cleanup**

Use `os.walk(..., followlinks=False)` to calculate apparent bytes/newest mtime. Require direct child name `proc_*`, non-symlink real directory, `resolve().parent == root.resolve()`, `exit_code` exists, id not active, and newest mtime ≤ cutoff. Any stat/read/remove error is appended to warnings and preserves the path.

`load_active_process_ids` must accept only a JSON list of dicts and collect nonempty string `session_id`; malformed existing JSON raises a concise `ValueError` so cleanup does not proceed.

- [ ] **Step 4: Add RED threshold and dedupe tests**

```python
def test_threshold_alerts_once_until_recovery_or_growth():
    high = snapshot(state_db=6 * GIB, chrome_debug=6 * GIB)
    first = evaluate_alerts(high, None)
    steady = evaluate_alerts(high, state_from(high))
    grown = evaluate_alerts(snapshot(state_db=8 * GIB, chrome_debug=6 * GIB), state_from(high))
    recovered = evaluate_alerts(snapshot(state_db=4 * GIB, chrome_debug=4 * GIB), state_from(high))
    crossed_again = evaluate_alerts(high, state_from(snapshot(state_db=4 * GIB, chrome_debug=4 * GIB)))
    assert {"state.db", "chrome-debug"} <= labels(first)
    assert steady == []
    assert "state.db" in labels(grown)
    assert recovered == []
    assert {"state.db", "chrome-debug"} <= labels(crossed_again)
```

Also cover disk free `<50 GiB` or `<10%`, logs/process/cron/sessions absolute thresholds, +1 GiB growth, and +25% with at least +250 MiB growth.

- [ ] **Step 5: Run threshold RED tests**

Run:

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest tests/scripts/test_hermes_storage_sentinel.py \
  -k 'threshold or growth or recovery or disk_free' -q
```

Expected: FAIL until threshold state logic exists.

- [ ] **Step 6: Implement measurement, state and stdout UX**

Measure exact roots from the approved spec and `shutil.disk_usage(home)`. State schema:

```json
{
  "version": 1,
  "measured_at": 0.0,
  "sizes": {},
  "over_threshold": []
}
```

Write it atomically only after successful measurement and cleanup. `--dry-run` performs no deletion and no state mutation. Normal stdout is empty unless warnings, a new/renewed threshold/growth alert, or reclaimed bytes ≥ 50 MiB. Do not create an append-only audit log; Cron's retained output is the audit trail.

- [ ] **Step 7: Run Task 4 GREEN and all script tests**

Run:

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/scripts/test_worktree_doctor.py \
  tests/scripts/test_worktree_doctor_cron.py \
  tests/scripts/test_hermes_storage_sentinel.py -q
/Users/zongxin/.hermes/hermes-agent/venv/bin/python -m py_compile \
  scripts/worktree_doctor.py \
  scripts/worktree_doctor_cron.py \
  scripts/hermes_storage_sentinel.py
git diff --check
```

Expected: all tests PASS; no warnings/errors.

- [ ] **Step 8: Commit Task 4**

```bash
git add -- scripts/hermes_storage_sentinel.py tests/scripts/test_hermes_storage_sentinel.py
git diff --cached --stat
git commit -m "feat: add bounded Hermes storage sentinel"
```

---

### Task 5: Independent review, integration and live Cron rollout

**Files:**
- Modify live: `~/.hermes/scripts/worktree_doctor_cron.py` copied from committed repo source
- Create live: `~/.hermes/scripts/hermes_storage_sentinel.py` copied from committed repo source
- Update live Cron job `d59e9ef27883`
- Create one weekly no-agent storage sentinel Cron
- Update ledger/artifact evidence only; no new repo behavior unless review finds a bug

**Interfaces:**
- Consumes: Tasks 1–4 commits and approved spec。
- Produces: active `main` squash commit, executable live scripts, Cron readback, real dry-run/smoke evidence, final state.db storage recommendation。

- [ ] **Step 1: Run canonical pre-review verification**

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/scripts/test_worktree_doctor.py \
  tests/scripts/test_worktree_doctor_cron.py \
  tests/scripts/test_hermes_storage_sentinel.py -q
/Users/zongxin/.hermes/hermes-agent/venv/bin/python -m py_compile scripts/worktree_doctor.py scripts/worktree_doctor_cron.py scripts/hermes_storage_sentinel.py
git diff --check
git status --short --untracked-files=all
```

Expected: tests PASS, compile PASS, diff check clean, only intended plan/spec state remains.

- [ ] **Step 2: Run Codex independent review pass 1/2**

State before invocation: `pass 1/2 — allowed because this spans multiple files, changes automatic deletion behavior, and rolls into live Cron.`

Review requirements:

- adversarial deletion/race/path/symlink/process-check correctness；
- exact spec traceability for 14/30 days, branch retention, fail-closed gates, bounded state, stdout UX, storage non-goals；
- tests exercise behavior rather than mocks-only surfaces。

Fix clear findings with focused RED/GREEN tests. A second Codex review is allowed only if pass 1 exposes broad/systemic problems or materially reshapes the diff.

- [ ] **Step 3: Squash implementation onto active main**

From active checkout, first verify clean scoped state, then:

```bash
git status --short --untracked-files=all
git merge --squash spec/worktree-storage-maintenance
git status --short
git diff --cached --stat
git commit -m "feat: archive stale worktrees and monitor Hermes storage"
```

Do not stage unrelated files. Rerun the canonical targeted suite from active main.

- [ ] **Step 4: Install live scripts with readback**

Copy committed source bytes to the scheduler-approved real paths, preserving executable mode:

```bash
install -m 755 scripts/worktree_doctor_cron.py ~/.hermes/scripts/worktree_doctor_cron.py
install -m 755 scripts/hermes_storage_sentinel.py ~/.hermes/scripts/hermes_storage_sentinel.py
cmp scripts/worktree_doctor_cron.py ~/.hermes/scripts/worktree_doctor_cron.py
cmp scripts/hermes_storage_sentinel.py ~/.hermes/scripts/hermes_storage_sentinel.py
```

Read back file mode and SHA-256 for repo/live pairs. The worktree wrapper must point to active `~/.hermes/hermes-agent/scripts/worktree_doctor.py`.

- [ ] **Step 5: Run non-destructive real dry-runs**

```bash
/Users/zongxin/.hermes/hermes-agent/venv/bin/python scripts/worktree_doctor.py --stale-days 14 --archive-days 30 --json >/tmp/worktree-doctor-live-dry.json
python ~/.hermes/scripts/hermes_storage_sentinel.py --dry-run --verbose >/tmp/hermes-storage-sentinel-dry.txt
```

Verify valid JSON, no 30-day archive candidate today unless live state changed, no deleted process logs, and the sentinel reports current known `state.db`/Chrome thresholds without touching them.

- [ ] **Step 6: Update/create Cron and read back**

- Update existing job `d59e9ef27883` in place: keep schedule `0 5 * * *`, `no_agent=true`, script `worktree_doctor_cron.py`, deliver `discord:1473223036667826268`, prompt updated to describe 14-day deduped warning and 30-day checkout-only archive.
- Create weekly job named `Hermes storage sentinel:30d process logs + growth alerts`: schedule `40 5 * * 0`, `no_agent=true`, script `hermes_storage_sentinel.py`, deliver `discord:1473223036667826268`, workdir `/Users/zongxin/clawd`.
- Read back job IDs, schedules, scripts, destinations, enabled state and `no_agent=true` via one `cronjob list` registry check.

- [ ] **Step 7: Run live smokes and verify output semantics**

Run both jobs manually with `cronjob run`.

Acceptance:

- worktree job exits 0；the current 16-day stale branch may alert once, but a repeat direct wrapper run is silent unless state changes；no branch/head is deleted。
- storage job exits 0；first run may alert once for `state.db >5 GiB` and `chrome-debug >5 GiB`；repeat is silent absent growth/warnings；no current log/session/db/Chrome data is deleted。
- latest job states read back `last_status=ok` and no delivery error。

- [ ] **Step 8: Complete state.db aggregate recommendation**

Combine the read-only audit with source semantics. Report:

- top SQLite objects and amplification；
- aggregate contribution from message content/tool role/provider replay metadata/FTS；
- whether current growth is mostly low-value tool outputs or required conversational history；
- safe phased retention proposal with a dry-run estimator and separate user approval gate before any delete/prune/VACUUM/index change。

Do not mutate `state.db` in this task.

- [ ] **Step 9: Final verification and cleanup**

```bash
PYTHONPATH=. /Users/zongxin/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/scripts/test_worktree_doctor.py \
  tests/scripts/test_worktree_doctor_cron.py \
  tests/scripts/test_hermes_storage_sentinel.py -q
git status --short --untracked-files=all
git worktree list --porcelain
```

Validate ledger, complete it with test/commit/cron/smoke evidence, then remove the merged implementation worktree and spec branch only after a lossless diff check.
