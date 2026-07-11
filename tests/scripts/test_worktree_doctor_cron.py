"""Tests for the no-agent worktree doctor cron wrapper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "worktree_doctor_cron.py"
    spec = importlib.util.spec_from_file_location("worktree_doctor_cron", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(**overrides) -> dict:
    base = {
        "reaped": [],
        "archived": [],
        "skipped": [],
        "merged": [],
        "stale": [],
        "stale_dirty": [],
        "archivable": [],
        "active": [],
    }
    base.update(overrides)
    return base


def _stale(branch: str, head: str, *, dirty: bool = False) -> dict:
    return {
        "path": f"/tmp/{branch}",
        "branch": branch,
        "head": head,
        "age_days": 16.0,
        "ahead": 2,
        "behind": 3,
        "dirty": dirty,
        "cls": "STALE_DIRTY" if dirty else "STALE",
    }


def _fake_doctor(tmp_path: Path, report: dict, *, exit_code: int = 0) -> Path:
    path = tmp_path / "fake_doctor.py"
    payload = json.dumps(report)
    path.write_text(
        "import sys\n"
        f"print({payload!r})\n"
        f"raise SystemExit({exit_code})\n"
    )
    return path


def _run(cron, tmp_path: Path, report: dict, *, exit_code: int = 0) -> int:
    doctor = _fake_doctor(tmp_path, report, exit_code=exit_code)
    return cron.main(
        [
            "--doctor",
            str(doctor),
            "--state",
            str(tmp_path / "state.json"),
            "--manifest",
            str(tmp_path / "actions.json"),
        ]
    )


def test_same_stale_signature_alerts_once(tmp_path, capsys):
    cron = _load_module()
    report = _report(stale=[_stale("feature", "abc")])

    assert _run(cron, tmp_path, report) == 0
    assert "feature" in capsys.readouterr().out

    assert _run(cron, tmp_path, report) == 0
    assert capsys.readouterr().out == ""


def test_head_or_class_change_realerts(tmp_path, capsys):
    cron = _load_module()
    assert _run(cron, tmp_path, _report(stale=[_stale("feature", "abc")])) == 0
    capsys.readouterr()

    changed = _report(stale_dirty=[_stale("feature", "def", dirty=True)])
    assert _run(cron, tmp_path, changed) == 0
    out = capsys.readouterr().out
    assert "feature" in out
    assert "dirty" in out


def test_archived_report_prints_restore_without_rewriting_manifest(tmp_path, capsys):
    cron = _load_module()
    manifest = tmp_path / "actions.json"
    initial = {
        "version": 1,
        "actions": [
            {"branch": "existing", "head": "123", "path": "/old/existing"}
        ],
    }
    manifest.write_text(json.dumps(initial))
    archived = _stale("feature", "abc") | {
        "path": "/old/path",
        "age_days": 31.0,
        "cls": "ARCHIVABLE",
    }

    assert _run(cron, tmp_path, _report(archived=[archived])) == 0

    out = capsys.readouterr().out
    assert json.loads(manifest.read_text()) == initial
    assert "git worktree add /old/path feature" in out


def test_unchanged_skipped_reason_is_deduplicated(tmp_path, capsys):
    cron = _load_module()
    report = _report(
        skipped=[{"path": "/tmp/feature", "branch": "feature", "reason": "active process cwd"}]
    )
    assert _run(cron, tmp_path, report) == 0
    assert "active process cwd" in capsys.readouterr().out
    assert _run(cron, tmp_path, report) == 0
    assert capsys.readouterr().out == ""


def test_doctor_failure_is_nonzero_and_stderr_only(tmp_path, capsys):
    cron = _load_module()
    assert _run(cron, tmp_path, _report(), exit_code=7) == 7
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exit 7" in captured.err


def test_wrapper_passes_manifest_to_doctor(tmp_path):
    cron = _load_module()
    argv_record = tmp_path / "argv.json"
    doctor = tmp_path / "recording_doctor.py"
    doctor.write_text(
        "import json, sys\n"
        f"open({str(argv_record)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        f"print({json.dumps(_report())!r})\n"
    )
    manifest = tmp_path / "actions.json"

    assert cron.main(
        [
            "--doctor",
            str(doctor),
            "--state",
            str(tmp_path / "state.json"),
            "--manifest",
            str(manifest),
        ]
    ) == 0

    forwarded = json.loads(argv_record.read_text())
    index = forwarded.index("--archive-manifest")
    assert forwarded[index + 1] == str(manifest)


def test_fresh_healthy_run_has_empty_stdout(tmp_path, capsys):
    cron = _load_module()
    assert _run(cron, tmp_path, _report()) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_invalid_doctor_json_is_nonzero_and_stderr_only(tmp_path, capsys):
    cron = _load_module()
    doctor = tmp_path / "invalid_doctor.py"
    doctor.write_text("print('not-json')\n")
    assert cron.main(
        [
            "--doctor",
            str(doctor),
            "--state",
            str(tmp_path / "state.json"),
            "--manifest",
            str(tmp_path / "actions.json"),
        ]
    ) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "JSON parse failed" in captured.err
