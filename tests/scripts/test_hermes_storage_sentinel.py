"""Tests for scripts/hermes_storage_sentinel.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

DAY = 86400
MIB = 1024**2
GIB = 1024**3
NOW = 1_800_000_000.0


def _load_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "hermes_storage_sentinel.py"
    spec = importlib.util.spec_from_file_location("hermes_storage_sentinel", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _proc_dir(
    root: Path,
    name: str,
    *,
    age_days: int,
    exit_code: bool,
    size: int = 32,
) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "stdout.log").write_bytes(b"x" * size)
    if exit_code:
        (path / "exit_code").write_text("0\n")
    ts = NOW - age_days * DAY
    for item in path.iterdir():
        os.utime(item, (ts, ts))
    os.utime(path, (ts, ts))
    return path


def _snapshot(sentinel, **sizes):
    defaults = {
        "logs": 10 * MIB,
        "process_logs": 10 * MIB,
        "cron_output": 10 * MIB,
        "sessions": 10 * MIB,
        "state.db": 1 * GIB,
        "chrome-debug": 1 * GIB,
    }
    defaults.update(sizes)
    return sentinel.Snapshot(
        measured_at=NOW,
        sizes=defaults,
        disk_total=500 * GIB,
        disk_free=200 * GIB,
    )


def test_prunes_only_finished_old_process_logs(tmp_path):
    sentinel = _load_module()
    root = tmp_path / "process_logs"
    old_finished = _proc_dir(root, "proc_old", age_days=31, exit_code=True)
    old_unknown = _proc_dir(root, "proc_unknown", age_days=31, exit_code=False)
    new_finished = _proc_dir(root, "proc_new", age_days=1, exit_code=True)

    result = sentinel.prune_finished_process_logs(
        root, set(), cutoff=NOW - 30 * DAY, dry_run=False
    )

    assert not old_finished.exists()
    assert old_unknown.exists()
    assert new_finished.exists()
    assert result.deleted_count == 1
    assert result.reclaimed_bytes == 34


def test_active_checkpoint_id_is_never_pruned(tmp_path):
    sentinel = _load_module()
    root = tmp_path / "process_logs"
    old_finished = _proc_dir(root, "proc_live", age_days=31, exit_code=True)

    result = sentinel.prune_finished_process_logs(
        root, {"proc_live"}, cutoff=NOW - 30 * DAY, dry_run=False
    )

    assert old_finished.exists()
    assert result.deleted_count == 0


def test_symlink_is_never_followed(tmp_path):
    sentinel = _load_module()
    root = tmp_path / "process_logs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "exit_code").write_text("0\n")
    (root / "proc_link").symlink_to(outside, target_is_directory=True)

    result = sentinel.prune_finished_process_logs(
        root, set(), cutoff=NOW, dry_run=False
    )

    assert outside.exists()
    assert (outside / "exit_code").exists()
    assert result.deleted_count == 0


def test_dry_run_reports_without_deleting(tmp_path):
    sentinel = _load_module()
    root = tmp_path / "process_logs"
    old_finished = _proc_dir(root, "proc_old", age_days=31, exit_code=True)

    result = sentinel.prune_finished_process_logs(
        root, set(), cutoff=NOW - 30 * DAY, dry_run=True
    )

    assert old_finished.exists()
    assert result.deleted_count == 1


def test_active_process_checkpoint_requires_valid_list(tmp_path):
    sentinel = _load_module()
    path = tmp_path / "processes.json"
    path.write_text(json.dumps([{"session_id": "proc_a"}, {"session_id": "proc_b"}]))
    assert sentinel.load_active_process_ids(path) == {"proc_a", "proc_b"}

    path.write_text("{}")
    with pytest.raises(ValueError, match="JSON list"):
        sentinel.load_active_process_ids(path)


def test_threshold_alerts_once_until_growth_or_recrossing():
    sentinel = _load_module()
    high = _snapshot(sentinel, **{"state.db": 6 * GIB, "chrome-debug": 6 * GIB})

    first = sentinel.evaluate_alerts(high, None)
    steady = sentinel.evaluate_alerts(high, sentinel.state_from_snapshot(high))
    grown = _snapshot(sentinel, **{"state.db": 8 * GIB, "chrome-debug": 6 * GIB})
    growth_alerts = sentinel.evaluate_alerts(grown, sentinel.state_from_snapshot(high))
    recovered = _snapshot(sentinel, **{"state.db": 4 * GIB, "chrome-debug": 4 * GIB})
    assert sentinel.evaluate_alerts(recovered, sentinel.state_from_snapshot(high)) == []
    crossed_again = sentinel.evaluate_alerts(high, sentinel.state_from_snapshot(recovered))

    assert any("state.db" in line for line in first)
    assert any("chrome-debug" in line for line in first)
    assert steady == []
    assert any("state.db" in line and "growth" in line for line in growth_alerts)
    assert any("state.db" in line for line in crossed_again)
    assert any("chrome-debug" in line for line in crossed_again)


def test_disk_free_threshold_and_twenty_five_percent_growth():
    sentinel = _load_module()
    low_disk = _snapshot(sentinel)
    low_disk = sentinel.Snapshot(
        measured_at=NOW,
        sizes=low_disk.sizes,
        disk_total=500 * GIB,
        disk_free=40 * GIB,
    )
    assert any("disk free" in line for line in sentinel.evaluate_alerts(low_disk, None))

    old = _snapshot(sentinel, sessions=800 * MIB)
    new = _snapshot(sentinel, sessions=1100 * MIB)
    alerts = sentinel.evaluate_alerts(new, sentinel.state_from_snapshot(old))
    assert any("sessions" in line and "growth" in line for line in alerts)


def test_main_dry_run_does_not_delete_or_write_state(tmp_path, capsys):
    sentinel = _load_module()
    home = tmp_path / ".hermes"
    old_finished = _proc_dir(
        home / "process_logs", "proc_old", age_days=31, exit_code=True
    )

    assert sentinel.main(
        ["--hermes-home", str(home), "--dry-run", "--verbose", "--now", str(NOW)]
    ) == 0

    assert old_finished.exists()
    assert not (home / "storage_sentinel_state.json").exists()
    assert "would prune" in capsys.readouterr().out


def test_main_malformed_state_fails_before_cleanup(tmp_path, capsys):
    sentinel = _load_module()
    home = tmp_path / ".hermes"
    old_finished = _proc_dir(
        home / "process_logs", "proc_old", age_days=31, exit_code=True
    )
    state = home / "storage_sentinel_state.json"
    state.write_text("[]")

    assert sentinel.main(
        ["--hermes-home", str(home), "--now", str(NOW)]
    ) == 1

    assert old_finished.exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "preflight failed" in captured.err


def test_ancestor_swap_cannot_redirect_process_log_deletion(tmp_path, monkeypatch):
    sentinel = _load_module()
    root = tmp_path / "process_logs"
    original_candidate = _proc_dir(root, "proc_old", age_days=31, exit_code=True)
    outside = tmp_path / "outside"
    outside_candidate = _proc_dir(outside, "proc_old", age_days=31, exit_code=True)
    (outside_candidate / "valuable.txt").write_text("must survive\n")
    moved_root = tmp_path / "process_logs-moved"
    original_candidate_stats = sentinel._candidate_tree_stats
    swapped = False

    def swap_after_measure(child_fd):
        nonlocal swapped
        result = original_candidate_stats(child_fd)
        if not swapped:
            root.rename(moved_root)
            root.symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(sentinel, "_candidate_tree_stats", swap_after_measure)
    result = sentinel.prune_finished_process_logs(
        root, set(), cutoff=NOW - 30 * DAY, dry_run=False
    )

    assert result.deleted_count == 1
    assert not (moved_root / "proc_old").exists()
    assert outside_candidate.exists()
    assert (outside_candidate / "valuable.txt").exists()


def test_candidate_swapped_before_open_is_not_deleted(tmp_path, monkeypatch):
    sentinel = _load_module()
    root = tmp_path / "process_logs"
    candidate = _proc_dir(root, "proc_old", age_days=31, exit_code=True)
    staged_root = tmp_path / "staged"
    replacement = _proc_dir(staged_root, "proc_old", age_days=31, exit_code=True)
    (replacement / "valuable.txt").write_text("must survive\n")
    old_ts = NOW - 31 * DAY
    os.utime(replacement / "valuable.txt", (old_ts, old_ts))
    moved_original = tmp_path / "original-moved"
    real_open = sentinel.os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "proc_old" and kwargs.get("dir_fd") is not None and not swapped:
            candidate.rename(moved_original)
            replacement.rename(candidate)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(sentinel.os, "open", swap_before_open)
    result = sentinel.prune_finished_process_logs(
        root, set(), cutoff=NOW - 30 * DAY, dry_run=False
    )

    assert result.deleted_count == 0
    assert candidate.exists()
    assert (candidate / "valuable.txt").exists()
    assert moved_original.exists()
    assert any("replaced" in warning for warning in result.warnings)


def test_same_name_real_directory_replacement_is_not_deleted(tmp_path, monkeypatch):
    sentinel = _load_module()
    root = tmp_path / "process_logs"
    candidate = _proc_dir(root, "proc_old", age_days=31, exit_code=True)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "valuable.txt").write_text("must survive\n")
    moved_original = tmp_path / "original-moved"
    original_clear = sentinel._clear_directory_fd
    swapped = False

    def clear_then_swap(child_fd):
        nonlocal swapped
        original_clear(child_fd)
        if not swapped:
            candidate.rename(moved_original)
            replacement.rename(candidate)
            swapped = True

    monkeypatch.setattr(sentinel, "_clear_directory_fd", clear_then_swap)
    result = sentinel.prune_finished_process_logs(
        root, set(), cutoff=NOW - 30 * DAY, dry_run=False
    )

    assert result.deleted_count == 0
    assert candidate.exists()
    assert (candidate / "valuable.txt").exists()
    assert moved_original.exists()
    assert any("replaced" in warning for warning in result.warnings)


def test_symlinked_exit_code_is_retained_and_warned(tmp_path):
    sentinel = _load_module()
    root = tmp_path / "process_logs"
    candidate = _proc_dir(root, "proc_old", age_days=31, exit_code=False)
    outside_marker = tmp_path / "outside-exit-code"
    outside_marker.write_text("0\n")
    old_ts = NOW - 31 * DAY
    os.utime(outside_marker, (old_ts, old_ts))
    (candidate / "exit_code").symlink_to(outside_marker)

    result = sentinel.prune_finished_process_logs(
        root, set(), cutoff=NOW - 30 * DAY, dry_run=False
    )

    assert candidate.exists()
    assert result.deleted_count == 0
    assert any("exit_code" in warning and "proc_old" in warning for warning in result.warnings)


def test_only_old_active_or_missing_sidecar_entries_warn(tmp_path):
    sentinel = _load_module()
    root = tmp_path / "process_logs"
    _proc_dir(root, "proc_old_active", age_days=31, exit_code=True)
    _proc_dir(root, "proc_old_unknown", age_days=31, exit_code=False)
    _proc_dir(root, "proc_new_active", age_days=1, exit_code=True)
    _proc_dir(root, "proc_new_unknown", age_days=1, exit_code=False)

    result = sentinel.prune_finished_process_logs(
        root,
        {"proc_old_active", "proc_new_active"},
        cutoff=NOW - 30 * DAY,
        dry_run=False,
    )

    warnings = "\n".join(result.warnings)
    assert "proc_old_active" in warnings
    assert "proc_old_unknown" in warnings
    assert "proc_new_active" not in warnings
    assert "proc_new_unknown" not in warnings


def test_main_deduplicates_persistent_cleanup_warnings(tmp_path, capsys):
    sentinel = _load_module()
    home = tmp_path / ".hermes"
    root = home / "process_logs"
    _proc_dir(root, "proc_old_unknown", age_days=31, exit_code=False)

    assert sentinel.main(["--hermes-home", str(home), "--now", str(NOW)]) == 0
    assert "proc_old_unknown" in capsys.readouterr().out

    assert sentinel.main(["--hermes-home", str(home), "--now", str(NOW)]) == 0
    assert capsys.readouterr().out == ""

    _proc_dir(root, "proc_second_unknown", age_days=31, exit_code=False)
    assert sentinel.main(["--hermes-home", str(home), "--now", str(NOW)]) == 0
    out = capsys.readouterr().out
    assert "proc_second_unknown" in out
    assert "proc_old_unknown" not in out

    state = json.loads((home / "storage_sentinel_state.json").read_text())
    assert len(state["warning_signatures"]) == 2
    assert all(len(signature) == 64 for signature in state["warning_signatures"])


def test_main_batches_more_than_twenty_warnings_without_losing_details(tmp_path, capsys):
    sentinel = _load_module()
    home = tmp_path / ".hermes"
    root = home / "process_logs"
    for index in range(21):
        _proc_dir(root, f"proc_unknown_{index:02d}", age_days=31, exit_code=False)

    assert sentinel.main(["--hermes-home", str(home), "--now", str(NOW)]) == 0
    first = capsys.readouterr().out
    assert first.count("retain old process log without exit_code") == 20
    assert "additional new warning(s) deferred" in first
    first_state = json.loads((home / "storage_sentinel_state.json").read_text())
    assert len(first_state["warning_signatures"]) == 20

    assert sentinel.main(["--hermes-home", str(home), "--now", str(NOW + 1)]) == 0
    second = capsys.readouterr().out
    assert second.count("retain old process log without exit_code") == 1
    second_state = json.loads((home / "storage_sentinel_state.json").read_text())
    assert len(second_state["warning_signatures"]) == 21


def test_fresh_healthy_sentinel_run_is_silent_and_writes_state(tmp_path, capsys):
    sentinel = _load_module()
    home = tmp_path / ".hermes"
    home.mkdir()

    assert sentinel.main(["--hermes-home", str(home), "--now", str(NOW)]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    state = json.loads((home / "storage_sentinel_state.json").read_text())
    assert state["warning_signatures"] == []
    assert state["sizes"]["state.db"] == 0
