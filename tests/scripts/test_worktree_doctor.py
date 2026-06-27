"""Regression tests for scripts/worktree_doctor.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_worktree_doctor():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "worktree_doctor.py"
    spec = importlib.util.spec_from_file_location("worktree_doctor", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
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
