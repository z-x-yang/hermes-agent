#!/usr/bin/env python3
"""No-agent cron adapter for safe worktree cleanup and deduplicated alerts."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HOME = Path.home()
DEFAULT_DOCTOR = HOME / ".hermes" / "hermes-agent" / "scripts" / "worktree_doctor.py"
DEFAULT_STATE = HOME / ".hermes" / "worktree_doctor_state.json"
DEFAULT_MANIFEST = HOME / ".hermes" / "worktree_doctor_actions.json"


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


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "reported": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("reported"), dict):
        raise ValueError(f"invalid worktree doctor state: {path}")
    return payload


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "actions": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
        raise ValueError(f"invalid worktree doctor manifest: {path}")
    return payload


def _signature(kind: str, item: dict[str, Any]) -> str:
    stable = [kind, item.get("branch"), item.get("head"), item.get("reason")]
    return json.dumps(stable, ensure_ascii=False, separators=(",", ":"))


def _alert_key(kind: str, item: dict[str, Any]) -> str:
    return f"{kind}:{item.get('branch') or item.get('path') or '?'}"


def _restore_command(item: dict[str, Any]) -> str:
    return "git worktree add {} {}".format(
        shlex.quote(str(item["path"])),
        shlex.quote(str(item["branch"])),
    )


def _new_alerts(
    report: dict[str, Any], previous: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    current: dict[str, str] = {}
    lines: list[str] = []

    for bucket, kind in (("stale", "STALE"), ("stale_dirty", "STALE_DIRTY")):
        for item in report.get(bucket, []):
            key = _alert_key(kind, item)
            sig = _signature(kind, item)
            current[key] = sig
            if previous.get(key) == sig:
                continue
            dirty = "dirty, " if kind == "STALE_DIRTY" else ""
            lines.append(
                f"⚠️ worktree {kind.lower()}: {item.get('branch')} "
                f"({item.get('age_days', 0):.0f}d, {dirty}ahead {item.get('ahead', 0)}, "
                f"{item.get('path')}) — 只提醒，不会自动删除"
            )

    for item in report.get("skipped", []):
        key = _alert_key("SKIPPED", item)
        sig = _signature("SKIPPED", item)
        current[key] = sig
        if previous.get(key) == sig:
            continue
        lines.append(
            f"⚠️ worktree skipped: {item.get('branch') or item.get('path')} — "
            f"{item.get('reason', 'unknown reason')}"
        )

    return current, lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="worktree doctor cron adapter")
    parser.add_argument("--doctor", type=Path, default=DEFAULT_DOCTOR)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        state = _load_state(args.state)
        _load_manifest(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"worktree-doctor state preflight failed: {exc}", file=sys.stderr)
        return 1

    if not args.doctor.exists():
        print(f"worktree-doctor 不存在: {args.doctor}", file=sys.stderr)
        return 1

    proc = subprocess.run(
        [
            sys.executable,
            str(args.doctor),
            "--reap",
            "--archive",
            "--archive-manifest",
            str(args.manifest),
            "--stale-days",
            "14",
            "--archive-days",
            "30",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip()
        suffix = f": {detail}" if detail else ""
        print(f"worktree-doctor failed (exit {proc.returncode}){suffix}", file=sys.stderr)
        return proc.returncode

    try:
        report = json.loads(proc.stdout)
        if not isinstance(report, dict):
            raise ValueError("report is not an object")
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"worktree-doctor JSON parse failed: {exc}", file=sys.stderr)
        return 1

    previous = state.get("reported", {})
    current, lines = _new_alerts(report, previous)

    for item in report.get("archived", []):
        restore = _restore_command(item)
        lines.append(
            f"🧹 archived worktree checkout: {item.get('branch')} "
            f"({item.get('head', '')[:10]}, {item.get('age_days', 0):.0f}d)\n"
            f"恢复: `{restore}`"
        )

    if report.get("reaped"):
        print(
            f"reaped {len(report['reaped'])} merged worktree(s): {report['reaped']}",
            file=sys.stderr,
        )

    try:
        _atomic_write_json(args.state, {"version": 1, "reported": current})
    except OSError as exc:
        print(f"worktree-doctor state write failed: {exc}", file=sys.stderr)
        return 1

    if lines:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
