"""Regression tests for scripts/worktree_doctor.py."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _load_worktree_doctor():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "worktree_doctor.py"
    spec = importlib.util.spec_from_file_location("worktree_doctor", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("# temp repo\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")


def _branch_worktree(tmp_path: Path, *, age_days: int) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    _init_repo(repo)
    worktree = tmp_path / "feature-worktree"
    _git(repo, "worktree", "add", str(worktree), "-b", "feature", "HEAD")
    (worktree / "feature.txt").write_text("branch-only work\n")
    _git(worktree, "add", "feature.txt")
    commit_ts = int(time.time() - age_days * 86400)
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = f"{commit_ts} +0000"
    env["GIT_COMMITTER_DATE"] = f"{commit_ts} +0000"
    _git(worktree, "commit", "-m", "branch work", env=env)
    return repo, worktree


def _feature_record(doctor, repo: Path) -> dict:
    doctor.REPO = str(repo)
    return next(wt for wt in doctor._list_worktrees() if wt.get("branch") == "feature")


def test_prunable_worktree_records_do_not_abort(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    _init_repo(repo)

    broken = tmp_path / "broken-worktree"
    _git(repo, "worktree", "add", str(broken), "-b", "broken", "HEAD")
    (broken / ".git").unlink()

    doctor = _load_worktree_doctor()
    monkeypatch.setattr(doctor, "REPO", str(repo))
    monkeypatch.setattr(sys, "argv", ["worktree_doctor.py", "--json"])

    assert doctor.main() == 0

    report = json.loads(capsys.readouterr().out)
    listed_paths = {
        item["path"]
        for bucket in ("merged", "stale", "active")
        for item in report[bucket]
    }
    assert str(broken) not in listed_paths


def test_clean_unmerged_29_days_is_stale(tmp_path):
    repo, _ = _branch_worktree(tmp_path, age_days=29)
    doctor = _load_worktree_doctor()
    verdict = doctor._classify(_feature_record(doctor, repo), 14, 30)
    assert verdict.cls == "STALE"


def test_clean_unmerged_31_days_is_archivable(tmp_path):
    repo, _ = _branch_worktree(tmp_path, age_days=31)
    doctor = _load_worktree_doctor()
    verdict = doctor._classify(_feature_record(doctor, repo), 14, 30)
    assert verdict.cls == "ARCHIVABLE"


def test_recent_uncommitted_mtime_resets_activity(tmp_path):
    repo, worktree = _branch_worktree(tmp_path, age_days=31)
    dirty = worktree / "recent.txt"
    dirty.write_text("new uncommitted work\n")
    doctor = _load_worktree_doctor()
    verdict = doctor._classify(_feature_record(doctor, repo), 14, 30)
    assert verdict.cls == "ACTIVE"
    assert verdict.dirty is True


def test_old_dirty_worktree_is_stale_dirty(tmp_path):
    repo, worktree = _branch_worktree(tmp_path, age_days=31)
    dirty = worktree / "old.txt"
    dirty.write_text("old uncommitted work\n")
    old_ts = time.time() - 31 * 86400
    os.utime(dirty, (old_ts, old_ts))
    doctor = _load_worktree_doctor()
    verdict = doctor._classify(_feature_record(doctor, repo), 14, 30)
    assert verdict.cls == "STALE_DIRTY"
    assert verdict.dirty is True


def test_locked_worktree_is_skip(tmp_path):
    repo, worktree = _branch_worktree(tmp_path, age_days=31)
    _git(repo, "worktree", "lock", "--reason", "manual hold", str(worktree))
    doctor = _load_worktree_doctor()
    record = _feature_record(doctor, repo)
    assert record["locked"] == "manual hold"
    assert doctor._classify(record, 14, 30).cls == "SKIP"


def _archivable_case(tmp_path: Path):
    repo, worktree = _branch_worktree(tmp_path, age_days=31)
    doctor = _load_worktree_doctor()
    verdict = doctor._classify(_feature_record(doctor, repo), 14, 30)
    assert verdict.cls == "ARCHIVABLE"
    return doctor, repo, worktree, verdict


def test_archive_removes_checkout_but_keeps_branch_and_head(tmp_path, monkeypatch):
    doctor, repo, worktree, verdict = _archivable_case(tmp_path)
    old_head = _git(repo, "rev-parse", "feature")
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    doctor._archive(verdict, archive_days=30)

    assert not worktree.exists()
    assert _git(repo, "rev-parse", "feature") == old_head


def test_archive_skips_when_live_cwd_detected(tmp_path, monkeypatch):
    doctor, _, worktree, verdict = _archivable_case(tmp_path)
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: True)

    with pytest.raises(RuntimeError, match="active process cwd"):
        doctor._archive(verdict, archive_days=30)

    assert worktree.exists()


def test_archive_fails_closed_when_process_probe_errors(tmp_path, monkeypatch):
    doctor, _, worktree, verdict = _archivable_case(tmp_path)

    def fail_probe(_path: str) -> bool:
        raise RuntimeError("lsof failed")

    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", fail_probe)
    with pytest.raises(RuntimeError, match="lsof failed"):
        doctor._archive(verdict, archive_days=30)

    assert worktree.exists()


def test_archive_skips_when_head_moves_after_classification(tmp_path, monkeypatch):
    doctor, _, worktree, verdict = _archivable_case(tmp_path)
    (worktree / "new.txt").write_text("new commit after classification\n")
    _git(worktree, "add", "new.txt")
    _git(worktree, "commit", "-m", "new work")
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    with pytest.raises(RuntimeError, match="被改动"):
        doctor._archive(verdict, archive_days=30)

    assert worktree.exists()


def test_main_archive_json_keeps_branch_and_records_manifest(tmp_path, monkeypatch, capsys):
    repo, worktree = _branch_worktree(tmp_path, age_days=31)
    doctor = _load_worktree_doctor()
    old_head = _git(repo, "rev-parse", "feature")
    manifest = tmp_path / "actions.json"
    monkeypatch.setattr(doctor, "REPO", str(repo))
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worktree_doctor.py",
            "--archive",
            "--archive-manifest",
            str(manifest),
            "--stale-days",
            "14",
            "--archive-days",
            "30",
            "--json",
        ],
    )

    assert doctor.main() == 0

    report = json.loads(capsys.readouterr().out)
    assert report["archived"][0]["branch"] == "feature"
    assert not worktree.exists()
    assert _git(repo, "rev-parse", "feature") == old_head
    action = json.loads(manifest.read_text())["actions"][-1]
    assert action["branch"] == "feature"
    assert action["head"] == old_head
    assert action["path"] == str(worktree)
    assert action["result"] == "archived"
    assert action["restore"] == f"git worktree add {worktree} feature"


def test_atomic_manifest_write_fsyncs_parent_directory(tmp_path, monkeypatch):
    doctor = _load_worktree_doctor()
    real_fsync = doctor.os.fsync
    fsynced_modes: list[int] = []

    def record_fsync(fd):
        fsynced_modes.append(os.fstat(fd).st_mode)
        real_fsync(fd)

    monkeypatch.setattr(doctor.os, "fsync", record_fsync)
    doctor._atomic_write_json(tmp_path / "actions.json", {"version": 1, "actions": []})

    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)


def test_archive_keeps_pending_recovery_record_if_result_write_fails(
    tmp_path, monkeypatch
):
    repo, worktree = _branch_worktree(tmp_path, age_days=31)
    doctor = _load_worktree_doctor()
    manifest = tmp_path / "actions.json"
    original_write = doctor._atomic_write_json
    writes = 0

    def fail_second_write(path, payload):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full after archive")
        original_write(path, payload)

    monkeypatch.setattr(doctor, "REPO", str(repo))
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)
    monkeypatch.setattr(doctor, "_atomic_write_json", fail_second_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worktree_doctor.py",
            "--archive",
            "--archive-manifest",
            str(manifest),
            "--stale-days",
            "14",
            "--archive-days",
            "30",
            "--json",
        ],
    )

    with pytest.raises(OSError, match="disk full after archive"):
        doctor.main()

    assert not worktree.exists()
    action = json.loads(manifest.read_text())["actions"][-1]
    assert action["result"] == "pending"
    assert action["branch"] == "feature"
    assert action["restore"] == f"git worktree add {worktree} feature"


def test_archive_blocks_unknown_ignored_artifact(tmp_path, monkeypatch):
    doctor, _, worktree, verdict = _archivable_case(tmp_path)
    (worktree / ".gitignore").write_text("artifact.bin\n")
    _git(worktree, "add", ".gitignore")
    old_commit_ts = int(time.time() - 31 * 86400)
    old_env = os.environ.copy()
    old_env["GIT_AUTHOR_DATE"] = f"{old_commit_ts} +0000"
    old_env["GIT_COMMITTER_DATE"] = f"{old_commit_ts} +0000"
    _git(worktree, "commit", "-m", "ignore artifact", env=old_env)
    verdict = doctor._classify(_feature_record(doctor, Path(doctor.REPO)), 14, 30)
    artifact = worktree / "artifact.bin"
    artifact.write_text("valuable ignored output\n")
    old_ts = time.time() - 31 * 86400
    os.utime(artifact, (old_ts, old_ts))
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    with pytest.raises(RuntimeError, match="unsafe ignored content"):
        doctor._archive(verdict, archive_days=30)

    assert artifact.exists()
    assert worktree.exists()


def test_archive_blocks_protected_ignored_root_even_with_cache_component(
    tmp_path, monkeypatch
):
    doctor, _, worktree, verdict = _archivable_case(tmp_path)
    protected = worktree / ".hermes"
    protected.mkdir()
    (protected / "keep").write_text("tracked marker\n")
    (worktree / ".gitignore").write_text(".hermes/node_modules/\n")
    _git(worktree, "add", ".gitignore", ".hermes/keep")
    old_commit_ts = int(time.time() - 31 * 86400)
    old_env = os.environ.copy()
    old_env["GIT_AUTHOR_DATE"] = f"{old_commit_ts} +0000"
    old_env["GIT_COMMITTER_DATE"] = f"{old_commit_ts} +0000"
    _git(worktree, "commit", "-m", "ignore protected cache path", env=old_env)
    verdict = doctor._classify(_feature_record(doctor, Path(doctor.REPO)), 14, 30)
    artifact = worktree / ".hermes" / "node_modules" / "valuable.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("must survive\n")
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    with pytest.raises(RuntimeError, match="unsafe ignored content"):
        doctor._archive(verdict, archive_days=30)

    assert artifact.exists()
    assert worktree.exists()


def test_archive_allows_known_rebuildable_ignored_cache(tmp_path, monkeypatch):
    doctor, repo, worktree, verdict = _archivable_case(tmp_path)
    (worktree / ".gitignore").write_text("__pycache__/\n")
    _git(worktree, "add", ".gitignore")
    old_commit_ts = int(time.time() - 31 * 86400)
    old_env = os.environ.copy()
    old_env["GIT_AUTHOR_DATE"] = f"{old_commit_ts} +0000"
    old_env["GIT_COMMITTER_DATE"] = f"{old_commit_ts} +0000"
    _git(worktree, "commit", "-m", "ignore cache", env=old_env)
    verdict = doctor._classify(_feature_record(doctor, repo), 14, 30)
    cache = worktree / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"cache")
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    doctor._archive(verdict, archive_days=30)

    assert not worktree.exists()
    assert _git(repo, "rev-parse", "feature") == verdict.head


def test_reap_merged_blocks_unknown_ignored_artifact(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitignore").write_text("artifact.bin\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore artifact")
    worktree = tmp_path / "merged-worktree"
    _git(repo, "worktree", "add", str(worktree), "-b", "feature", "HEAD")
    artifact = worktree / "artifact.bin"
    artifact.write_text("valuable ignored output\n")
    doctor = _load_worktree_doctor()
    verdict = doctor._classify(_feature_record(doctor, repo), 14, 30)
    assert verdict.cls == "MERGED"
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    with pytest.raises(RuntimeError, match="unsafe ignored content"):
        doctor._reap(verdict)

    assert artifact.exists()
    assert worktree.exists()
    assert _git(repo, "rev-parse", "feature") == verdict.head


def test_reap_merged_blocks_live_cwd(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_repo(repo)
    worktree = tmp_path / "merged-worktree"
    _git(repo, "worktree", "add", str(worktree), "-b", "feature", "HEAD")
    doctor = _load_worktree_doctor()
    verdict = doctor._classify(_feature_record(doctor, repo), 14, 30)
    assert verdict.cls == "MERGED"
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: True)

    with pytest.raises(RuntimeError, match="active process cwd"):
        doctor._reap(verdict)

    assert worktree.exists()
    assert _git(repo, "rev-parse", "feature") == verdict.head


def _merged_case(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    worktree = tmp_path / "merged-worktree"
    _git(repo, "worktree", "add", str(worktree), "-b", "feature", "HEAD")
    doctor = _load_worktree_doctor()
    verdict = doctor._classify(_feature_record(doctor, repo), 14, 30)
    assert verdict.cls == "MERGED"
    return doctor, repo, worktree, verdict


def test_reap_rechecks_lock_after_classification(tmp_path, monkeypatch):
    doctor, repo, worktree, verdict = _merged_case(tmp_path)
    _git(repo, "worktree", "lock", "--reason", "new hold", str(worktree))
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    with pytest.raises(RuntimeError, match="locked"):
        doctor._reap(verdict)

    assert worktree.exists()
    assert _git(repo, "rev-parse", "feature") == verdict.head


def test_reap_rechecks_dirty_after_classification(tmp_path, monkeypatch):
    doctor, repo, worktree, verdict = _merged_case(tmp_path)
    (worktree / "dirty.txt").write_text("new uncommitted work\n")
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    with pytest.raises(RuntimeError, match="dirty"):
        doctor._reap(verdict)

    assert worktree.exists()
    assert _git(repo, "rev-parse", "feature") == verdict.head


def test_reap_rechecks_content_still_absorbed(tmp_path, monkeypatch):
    repo, worktree = _branch_worktree(tmp_path, age_days=31)
    patch = _git(worktree, "show", "--format=", "--binary", "HEAD") + "\n"
    subprocess.run(
        ["git", "apply", "--index", "-"],
        cwd=repo,
        input=patch,
        text=True,
        check=True,
        capture_output=True,
    )
    _git(repo, "commit", "-m", "squash feature content")
    doctor = _load_worktree_doctor()
    verdict = doctor._classify(_feature_record(doctor, repo), 14, 30)
    assert verdict.cls == "MERGED"
    (repo / "feature.txt").write_text("main diverged after classification\n")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "diverge merged content")
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    with pytest.raises(RuntimeError, match="no longer absorbed"):
        doctor._reap(verdict)

    assert worktree.exists()
    assert _git(repo, "rev-parse", "feature") == verdict.head


def test_archive_rejects_same_path_reused_by_different_branch(tmp_path, monkeypatch):
    doctor, repo, worktree, verdict = _archivable_case(tmp_path)
    _git(repo, "worktree", "remove", str(worktree))
    _git(repo, "worktree", "add", str(worktree), "-b", "replacement", "main")
    valuable = worktree / "valuable.txt"
    valuable.write_text("replacement checkout\n")
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    with pytest.raises(RuntimeError, match="identity changed"):
        doctor._archive(verdict, archive_days=30)

    assert worktree.exists()
    assert valuable.exists()
    assert _git(repo, "rev-parse", "replacement")


def test_archive_rechecks_lock_after_classification(tmp_path, monkeypatch):
    doctor, repo, worktree, verdict = _archivable_case(tmp_path)
    _git(repo, "worktree", "lock", "--reason", "new hold", str(worktree))
    monkeypatch.setattr(doctor, "_worktree_has_live_cwd", lambda _path: False)

    with pytest.raises(RuntimeError, match="locked"):
        doctor._archive(verdict, archive_days=30)

    assert worktree.exists()
    assert _git(repo, "rev-parse", "feature") == verdict.head


def test_lsof_exit_one_with_stdout_still_means_live_cwd(monkeypatch):
    doctor = _load_worktree_doctor()
    completed = subprocess.CompletedProcess(
        args=["lsof"], returncode=1, stdout="p123\nfcwd\n", stderr=""
    )
    monkeypatch.setattr(doctor.subprocess, "run", lambda *args, **kwargs: completed)
    assert doctor._worktree_has_live_cwd("/tmp/worktree") is True


def test_actual_lsof_detects_current_pytest_cwd():
    doctor = _load_worktree_doctor()
    assert doctor._worktree_has_live_cwd(str(Path.cwd())) is True


def test_classification_failure_is_reported_as_skipped(tmp_path, monkeypatch, capsys):
    repo, _ = _branch_worktree(tmp_path, age_days=31)
    doctor = _load_worktree_doctor()
    monkeypatch.setattr(doctor, "REPO", str(repo))
    original = doctor._classify

    def fail_feature(record, stale_days, archive_days):
        if record.get("branch") == "feature":
            raise RuntimeError("status probe failed")
        return original(record, stale_days, archive_days)

    monkeypatch.setattr(doctor, "_classify", fail_feature)
    monkeypatch.setattr(sys, "argv", ["worktree_doctor.py", "--json"])

    assert doctor.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["skipped"] == [
        {
            "path": str(tmp_path / "feature-worktree"),
            "branch": "feature",
            "reason": "status probe failed",
        }
    ]


def test_squash_absorbed_branch_is_merged(tmp_path):
    repo, worktree = _branch_worktree(tmp_path, age_days=31)
    patch = _git(worktree, "show", "--format=", "--binary", "HEAD") + "\n"
    subprocess.run(
        ["git", "apply", "--index", "-"],
        cwd=repo,
        input=patch,
        text=True,
        check=True,
        capture_output=True,
    )
    _git(repo, "commit", "-m", "squash feature content")
    doctor = _load_worktree_doctor()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(doctor, "REPO", str(repo))
    try:
        assert doctor._is_merged("feature") is True
    finally:
        monkeypatch.undo()
