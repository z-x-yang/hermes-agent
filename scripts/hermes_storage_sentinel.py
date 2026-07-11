#!/usr/bin/env python3
"""Weekly Hermes storage sentinel with one narrow, safe cleanup policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MIB = 1024**2
GIB = 1024**3
PROCESS_LOG_RETENTION_DAYS = 30
RECLAIM_REPORT_BYTES = 50 * MIB
MIN_GROWTH_BYTES = 250 * MIB
ABSOLUTE_GROWTH_BYTES = 1 * GIB
GROWTH_RATIO = 0.25
DISK_FREE_MIN_BYTES = 50 * GIB
DISK_FREE_MIN_RATIO = 0.10

SIZE_THRESHOLDS = {
    "logs": 250 * MIB,
    "process_logs": 250 * MIB,
    "cron_output": 250 * MIB,
    "sessions": 1 * GIB,
    "state.db": 5 * GIB,
    "chrome-debug": 5 * GIB,
}
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
_FILE_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True)
class Snapshot:
    measured_at: float
    sizes: dict[str, int]
    disk_total: int
    disk_free: int


@dataclass
class PruneResult:
    deleted_count: int = 0
    reclaimed_bytes: int = 0
    warnings: list[str] = field(default_factory=list)


def _fmt_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def _tree_stats(path: Path) -> tuple[int, float]:
    """Return apparent file bytes and newest mtime without following symlinks."""
    if path.is_symlink():
        raise ValueError(f"refuse symlink: {path}")
    stat = path.stat()
    if path.is_file():
        return stat.st_size, stat.st_mtime
    total = 0
    newest = stat.st_mtime
    for root, dirs, files in os.walk(path, topdown=True, followlinks=False):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if not (root_path / name).is_symlink()]
        for name in files:
            child = root_path / name
            if child.is_symlink():
                continue
            child_stat = child.stat()
            total += child_stat.st_size
            newest = max(newest, child_stat.st_mtime)
    return total, newest


def _candidate_tree_stats(child_fd: int) -> tuple[int, float]:
    """Measure one process-log tree through a pinned directory descriptor."""
    total = 0
    newest = os.fstat(child_fd).st_mtime
    for _dirpath, dirnames, filenames, dir_fd in os.fwalk(
        ".", topdown=True, follow_symlinks=False, dir_fd=child_fd
    ):
        for name in [*dirnames, *filenames]:
            entry = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            newest = max(newest, entry.st_mtime)
            if stat.S_ISREG(entry.st_mode):
                total += entry.st_size
    return total, newest


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _clear_directory_fd(dir_fd: int) -> None:
    """Clear the pinned directory tree without resolving its pathname again."""
    with os.scandir(dir_fd) as entries:
        names = [entry.name for entry in entries]
    for name in names:
        before = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if stat.S_ISDIR(before.st_mode):
            child_fd = os.open(name, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
            try:
                pinned = os.fstat(child_fd)
                if not _same_inode(before, pinned):
                    raise RuntimeError(f"nested directory replaced before open: {name}")
                _clear_directory_fd(child_fd)
                current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                if not _same_inode(pinned, current):
                    raise RuntimeError(f"nested directory replaced before rmdir: {name}")
                os.rmdir(name, dir_fd=dir_fd)
            finally:
                os.close(child_fd)
            continue

        if stat.S_ISREG(before.st_mode):
            file_fd = os.open(name, _FILE_OPEN_FLAGS, dir_fd=dir_fd)
            try:
                pinned = os.fstat(file_fd)
                current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                if not _same_inode(pinned, current):
                    raise RuntimeError(f"file replaced before unlink: {name}")
                os.unlink(name, dir_fd=dir_fd)
            finally:
                os.close(file_fd)
            continue

        current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if not _same_inode(before, current):
            raise RuntimeError(f"entry replaced before unlink: {name}")
        os.unlink(name, dir_fd=dir_fd)


def load_active_process_ids(processes_json: Path) -> set[str]:
    if not processes_json.exists():
        return set()
    payload = json.loads(processes_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"process checkpoint must be a JSON list: {processes_json}")
    active: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"process checkpoint entry is not an object: {processes_json}")
        session_id = item.get("session_id")
        if isinstance(session_id, str) and session_id:
            active.add(session_id)
    return active


def prune_finished_process_logs(
    root: Path,
    active_ids: set[str],
    cutoff: float,
    dry_run: bool,
) -> PruneResult:
    result = PruneResult()
    if not root.exists():
        return result
    if root.is_symlink() or not root.is_dir():
        result.warnings.append(f"refuse unsafe process_logs root: {root}")
        return result

    try:
        root_fd = os.open(root, _DIR_OPEN_FLAGS)
    except OSError as exc:
        result.warnings.append(f"cannot open pinned process_logs root {root}: {exc}")
        return result

    try:
        try:
            names = os.listdir(root_fd)
        except OSError as exc:
            result.warnings.append(f"cannot list {root}: {exc}")
            return result

        for name in names:
            if not name.startswith("proc_"):
                continue
            child_fd: int | None = None
            try:
                entry = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if not stat.S_ISDIR(entry.st_mode):
                    result.warnings.append(f"skip unsafe process log path: {root / name}")
                    continue
                child_fd = os.open(name, _DIR_OPEN_FLAGS, dir_fd=root_fd)
                pinned = os.fstat(child_fd)
                if not _same_inode(entry, pinned):
                    result.warnings.append(
                        f"retain replaced process log path before open: {root / name}"
                    )
                    continue
                size, newest = _candidate_tree_stats(child_fd)
                if newest > cutoff:
                    continue
                if name in active_ids:
                    result.warnings.append(f"retain old active process log: {root / name}")
                    continue
                try:
                    exit_entry = os.stat(
                        "exit_code", dir_fd=child_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    result.warnings.append(
                        f"retain old process log without exit_code: {root / name}"
                    )
                    continue
                if not stat.S_ISREG(exit_entry.st_mode):
                    result.warnings.append(
                        f"retain unsafe exit_code sidecar: {root / name / 'exit_code'}"
                    )
                    continue

                current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if not _same_inode(current, pinned):
                    result.warnings.append(
                        f"retain replaced process log path: {root / name}"
                    )
                    continue
                if not dry_run:
                    _clear_directory_fd(child_fd)
                    current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                    if not _same_inode(current, pinned):
                        result.warnings.append(
                            f"retain replaced process log path after clear: {root / name}"
                        )
                        continue
                    os.rmdir(name, dir_fd=root_fd)
                result.deleted_count += 1
                result.reclaimed_bytes += size
            except (OSError, ValueError, RuntimeError) as exc:
                result.warnings.append(f"failed to inspect/remove {root / name}: {exc}")
            finally:
                if child_fd is not None:
                    os.close(child_fd)
    finally:
        os.close(root_fd)
    return result


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        size, _ = _tree_stats(path)
        return size
    except (OSError, ValueError):
        raise RuntimeError(f"failed to measure {path}")


def collect_snapshot(hermes_home: Path, measured_at: float | None = None) -> Snapshot:
    sizes = {
        "logs": _path_size(hermes_home / "logs"),
        "process_logs": _path_size(hermes_home / "process_logs"),
        "cron_output": _path_size(hermes_home / "cron" / "output"),
        "sessions": _path_size(hermes_home / "sessions"),
        "chrome-debug": _path_size(hermes_home / "chrome-debug"),
    }
    sizes["state.db"] = sum(
        path.stat().st_size if path.exists() else 0
        for path in (
            hermes_home / "state.db",
            hermes_home / "state.db-wal",
            hermes_home / "state.db-shm",
        )
    )
    disk = shutil.disk_usage(hermes_home)
    return Snapshot(
        measured_at=time.time() if measured_at is None else measured_at,
        sizes=sizes,
        disk_total=disk.total,
        disk_free=disk.free,
    )


def _over_threshold(snapshot: Snapshot) -> set[str]:
    labels = {
        label
        for label, threshold in SIZE_THRESHOLDS.items()
        if snapshot.sizes.get(label, 0) > threshold
    }
    if (
        snapshot.disk_free < DISK_FREE_MIN_BYTES
        or snapshot.disk_free / max(snapshot.disk_total, 1) < DISK_FREE_MIN_RATIO
    ):
        labels.add("disk free")
    return labels


def _warning_signature(warning: str) -> str:
    return hashlib.sha256(warning.encode("utf-8")).hexdigest()


def state_from_snapshot(
    snapshot: Snapshot, warning_signatures: set[str] | None = None
) -> dict[str, Any]:
    return {
        "version": 1,
        "measured_at": snapshot.measured_at,
        "sizes": dict(snapshot.sizes),
        "disk_total": snapshot.disk_total,
        "disk_free": snapshot.disk_free,
        "over_threshold": sorted(_over_threshold(snapshot)),
        "warning_signatures": sorted(warning_signatures or set()),
    }


def evaluate_alerts(snapshot: Snapshot, previous: dict[str, Any] | None) -> list[str]:
    alerts: list[str] = []
    current_over = _over_threshold(snapshot)
    previous_over = set(previous.get("over_threshold", [])) if previous else set()

    for label in sorted(current_over - previous_over):
        if label == "disk free":
            alerts.append(
                f"disk free threshold crossed: {_fmt_bytes(snapshot.disk_free)} free "
                f"of {_fmt_bytes(snapshot.disk_total)}"
            )
        else:
            alerts.append(
                f"{label} threshold crossed: {_fmt_bytes(snapshot.sizes.get(label, 0))}"
            )

    if previous:
        old_sizes = previous.get("sizes", {})
        if isinstance(old_sizes, dict):
            for label, current in snapshot.sizes.items():
                old = old_sizes.get(label)
                if not isinstance(old, (int, float)) or old < 0:
                    continue
                delta = current - int(old)
                ratio = delta / max(int(old), 1)
                if delta >= ABSOLUTE_GROWTH_BYTES or (
                    delta >= MIN_GROWTH_BYTES and ratio >= GROWTH_RATIO
                ):
                    alerts.append(
                        f"{label} growth: +{_fmt_bytes(delta)} "
                        f"({_fmt_bytes(int(old))} → {_fmt_bytes(current)})"
                    )
    return alerts


def _load_previous_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("sizes"), dict):
        raise ValueError(f"invalid storage sentinel state: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hermes storage sentinel")
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=Path.home() / ".hermes",
    )
    parser.add_argument("--state", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--now", type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hermes_home = args.hermes_home.resolve()
    state_path = args.state or hermes_home / "storage_sentinel_state.json"
    now = time.time() if args.now is None else args.now

    try:
        previous = _load_previous_state(state_path)
        active_ids = load_active_process_ids(hermes_home / "processes.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Hermes storage sentinel preflight failed: {exc}", file=sys.stderr)
        return 1

    prune = prune_finished_process_logs(
        hermes_home / "process_logs",
        active_ids,
        cutoff=now - PROCESS_LOG_RETENTION_DAYS * 86400,
        dry_run=args.dry_run,
    )
    try:
        snapshot = collect_snapshot(hermes_home, measured_at=now)
        alerts = evaluate_alerts(snapshot, previous)
        current_warning_signatures = {
            _warning_signature(warning) for warning in prune.warnings
        }
        previous_warning_signatures = set(
            previous.get("warning_signatures", []) if previous else []
        )
        new_warnings = [
            warning
            for warning in prune.warnings
            if args.dry_run
            or _warning_signature(warning) not in previous_warning_signatures
        ]
        delivered_new_warnings = new_warnings[:20]
        acknowledged_warning_signatures = (
            previous_warning_signatures & current_warning_signatures
        ) | {_warning_signature(warning) for warning in delivered_new_warnings}
        if not args.dry_run:
            _atomic_write_json(
                state_path,
                state_from_snapshot(snapshot, acknowledged_warning_signatures),
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Hermes storage sentinel failed: {exc}", file=sys.stderr)
        return 1

    lines = list(delivered_new_warnings)
    if len(new_warnings) > 20:
        lines.append(f"... {len(new_warnings) - 20} additional new warning(s) deferred")
    if prune.deleted_count and (
        args.dry_run or args.verbose or prune.reclaimed_bytes >= RECLAIM_REPORT_BYTES
    ):
        verb = "would prune" if args.dry_run else "pruned"
        lines.append(
            f"{verb} {prune.deleted_count} finished process log dir(s), "
            f"{_fmt_bytes(prune.reclaimed_bytes)}"
        )
    lines.extend(alerts)
    if args.verbose and not lines:
        lines.append("Hermes storage sentinel healthy: no cleanup or new alerts")
    if lines:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
