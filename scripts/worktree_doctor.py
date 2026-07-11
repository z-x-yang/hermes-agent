#!/usr/bin/env python3
"""worktree-doctor — 体检 git worktree，回收已合并的，安全归档久未动 checkout。

确定性部分列出每个 worktree 的分支、相对 main 的 ahead/behind、最近 commit 或
未提交文件活动距今天数、dirty/locked 状态，以及改动是否已并入 main。

自动动作分两类：
- ``--reap`` 仅作用于 MERGED：删除 checkout，并以 pinned OID 原子删除 branch；
- ``--archive`` 仅作用于超过 archive-days 的 clean 未合并 worktree：只删除
  checkout，保留 branch 与 HEAD，之后可用 ``git worktree add`` 恢复。

两条删除路径都会在执行前重验分支 OID、拒绝 unknown ignored artifact、拒绝
live process cwd，且从不使用 ``--force``。STALE/STALE_DIRTY 只报告，不自动删。
MERGED 判断不依赖 commit 祖先关系，因此对 squash-merge 稳健；main 若在合并后
又改了同一文件会保守地不回收。

配合 cron：``--json`` 输出供 no-agent 告警管线消费，提醒路由到 #ops-alerts。
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

MAIN = "main"
REPO = str(Path.home() / ".hermes" / "hermes-agent")  # 固定操作生产仓库,不随调用方 cwd 漂移
DEFAULT_STALE_DAYS = 14
DEFAULT_ARCHIVE_DAYS = 30
_REBUILDABLE_IGNORED_COMPONENTS = {
    "__pycache__",
    ".pytest_cache",
    ".pytest-cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "node_modules",
}
_REBUILDABLE_IGNORED_FILES = {"test_durations.json", ".DS_Store"}
_PROTECTED_IGNORED_ROOTS = {".hermes", "work"}
MAX_ARCHIVE_ACTIONS = 1000


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_archive_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "actions": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
        raise ValueError(f"invalid archive manifest: {path}")
    return payload


def _pending_archive_action(v: Verdict) -> dict:
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "path": v.path,
        "branch": v.branch,
        "head": v.head,
        "age_days": v.age_days,
        "result": "pending",
        "restore": shlex.join(["git", "worktree", "add", v.path, v.branch or ""]),
    }


def _git(*args: str, cwd: str | None = None) -> str:
    # fail-fast:git 出错直接抛(不静默)。默认操作 REPO,防 cron wrapper 里 stray cd 误伤别的 repo。
    return subprocess.run(
        ["git", *args], cwd=cwd or REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _list_worktrees() -> list[dict]:
    """解析 `git worktree list --porcelain` → [{path, branch, head}]."""
    items: list[dict] = []
    cur: dict = {}
    for line in _git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            if cur:
                items.append(cur)
            cur = {"path": line[len("worktree "):], "branch": None}
        elif line.startswith("HEAD "):
            cur["head"] = line[len("HEAD "):]
        elif line.startswith("branch "):
            cur["branch"] = line[len("branch "):].replace("refs/heads/", "")
        elif line.startswith("prunable "):
            # `git worktree list --porcelain` can report metadata for a worktree
            # whose .git file is already gone. It is not a valid checkout, so the
            # doctor must not run `git status` inside it.
            cur["prunable"] = line[len("prunable "):]
        elif line == "locked":
            cur["locked"] = True
        elif line.startswith("locked "):
            cur["locked"] = line[len("locked "):]
        elif line == "detached":
            cur["branch"] = None
    if cur:
        items.append(cur)
    if items:
        items[0]["is_main"] = True  # 主 worktree(list 恒列第一)绝不回收
    return items


def _ahead_behind(branch: str) -> tuple[int, int]:
    # left = main 独有(branch 落后),right = branch 独有(领先)
    behind, ahead = _git(
        "rev-list", "--left-right", "--count", f"{MAIN}...{branch}"
    ).split()
    return int(ahead), int(behind)


def _changed_paths(path: str) -> list[Path]:
    """Return tracked, staged, and untracked changed paths without rename parsing."""
    rels: set[str] = set()
    for args in (
        ("diff", "--name-only", "-z"),
        ("diff", "--cached", "--name-only", "-z"),
        ("ls-files", "--others", "--exclude-standard", "-z"),
    ):
        raw = _git(*args, cwd=path)
        rels.update(item for item in raw.split("\0") if item)
    return [Path(path) / rel for rel in sorted(rels)]


def _activity_age_days(branch: str, path: str) -> tuple[float, bool]:
    """Return age since the newest commit or uncommitted source-file activity."""
    commit_ts = float(_git("log", "-1", "--format=%ct", branch))
    dirty = bool(_git("status", "--porcelain", "-z", cwd=path))
    latest = commit_ts
    if dirty:
        for changed in _changed_paths(path):
            if changed.exists():
                latest = max(latest, changed.stat().st_mtime)
    return (time.time() - latest) / 86400, dirty


def _is_merged(branch: str) -> bool:
    """branch 自分叉点引入改动的文件,在 branch 与 main 上内容是否已一致。"""
    base = _git("merge-base", MAIN, branch)
    changed = _git("diff", "--name-only", base, branch)
    if not changed:
        return True  # 相对分叉点无改动 → 空分支,安全可删
    files = changed.splitlines()
    # --quiet:有差异返回 1、无差异返回 0;不打印。固定 cwd=REPO。
    return subprocess.run(
        ["git", "diff", "--quiet", branch, MAIN, "--", *files], cwd=REPO
    ).returncode == 0


@dataclass
class Verdict:
    path: str
    branch: str | None
    ahead: int
    behind: int
    age_days: float
    dirty: bool
    cls: str  # ACTIVE | MERGED | STALE | SKIP
    head: str = ""  # 分类时 pin 的分支 OID,删除前据此防 moving-ref race


def _classify(wt: dict, stale_days: int, archive_days: int) -> Verdict:
    path, branch = wt["path"], wt.get("branch")
    if wt.get("is_main") or branch in (None, MAIN):
        # 主 worktree(gateway 生产)/ detached / main 本身:绝不回收
        return Verdict(path, branch, 0, 0, 0.0, False, "SKIP")
    if wt.get("prunable") or wt.get("locked") or not (Path(path) / ".git").exists():
        # Broken/locked records are not eligible for classification or removal.
        return Verdict(path, branch, 0, 0, 0.0, False, "SKIP")
    ahead, behind = _ahead_behind(branch)
    age, dirty = _activity_age_days(branch, path)
    head = _git("rev-parse", f"refs/heads/{branch}")  # pin,供删除前防 race
    if dirty:
        cls = "STALE_DIRTY" if age >= stale_days else "ACTIVE"
    elif _is_merged(branch):
        cls = "MERGED"            # 改动已进 main,无损可删
    elif age >= archive_days:
        cls = "ARCHIVABLE"        # checkout 可归档,branch/HEAD 必须保留
    elif age >= stale_days:
        cls = "STALE"             # 久未动 + 有独有改动,先提醒
    else:
        cls = "ACTIVE"
    return Verdict(path, branch, ahead, behind, age, dirty, cls, head)


def _ignored_path_is_rebuildable(rel: str) -> bool:
    path = Path(rel.rstrip("/"))
    if path.parts and path.parts[0] in _PROTECTED_IGNORED_ROOTS:
        return False
    if path.name in _REBUILDABLE_IGNORED_FILES or path.suffix == ".pyc":
        return True
    return any(
        part in _REBUILDABLE_IGNORED_COMPONENTS or part.endswith(".egg-info")
        for part in path.parts
    )


def _unsafe_ignored_paths(path: str) -> list[str]:
    """Return ignored entries that are not explicitly known rebuildable caches."""
    proc = subprocess.run(
        ["git", "status", "--ignored", "--porcelain=v1", "-z"],
        cwd=path,
        capture_output=True,
        text=False,
        check=True,
    )
    unsafe: list[str] = []
    for record in proc.stdout.split(b"\0"):
        if not record.startswith(b"!! "):
            continue
        rel = record[3:].decode("utf-8", "surrogateescape")
        if not _ignored_path_is_rebuildable(rel):
            unsafe.append(rel)
    return unsafe


def _worktree_has_live_cwd(path: str) -> bool:
    """Return whether a live process has cwd inside path; fail on probe errors."""
    proc = subprocess.run(
        ["/usr/sbin/lsof", "-a", "-d", "cwd", "+D", path, "-F", "p"],
        capture_output=True,
        text=True,
    )
    if proc.stdout.strip():
        # macOS lsof can emit valid p/f records yet exit 1 for +D scans.
        return True
    if proc.returncode in (0, 1):
        return False
    raise RuntimeError(
        f"lsof cwd probe failed ({proc.returncode}): {proc.stderr.strip()}"
    )


def _ensure_removal_side_effects_safe(v: Verdict) -> None:
    unsafe_ignored = _unsafe_ignored_paths(v.path)
    if unsafe_ignored:
        preview = ", ".join(unsafe_ignored[:5])
        extra = f" (+{len(unsafe_ignored) - 5} more)" if len(unsafe_ignored) > 5 else ""
        raise RuntimeError(
            f"{v.branch} has unsafe ignored content: {preview}{extra}"
        )
    if _worktree_has_live_cwd(v.path):
        raise RuntimeError(f"{v.branch} has active process cwd inside worktree")


def _current_worktree_record(v: Verdict) -> dict:
    current = next(
        (wt for wt in _list_worktrees() if wt.get("path") == v.path),
        None,
    )
    if current is None or current.get("prunable"):
        raise RuntimeError(f"{v.branch} worktree missing or prunable")
    if current.get("branch") != v.branch or current.get("head") != v.head:
        raise RuntimeError(
            f"{v.branch} worktree identity changed at {v.path}: "
            f"branch={current.get('branch')} head={current.get('head')}"
        )
    if current.get("locked"):
        raise RuntimeError(f"{v.branch} worktree is locked: {current.get('locked')}")
    if not (Path(v.path) / ".git").exists():
        raise RuntimeError(f"{v.branch} checkout is no longer valid")
    return current


def _archive(v: Verdict, archive_days: int) -> None:
    """Remove an inactive clean checkout while preserving its branch and HEAD."""
    current_head = _git("rev-parse", f"refs/heads/{v.branch}")
    if current_head != v.head:
        raise RuntimeError(
            f"{v.branch} 在分类后被改动({v.head[:8]}→{current_head[:8]}),跳过归档"
        )

    _current_worktree_record(v)

    age, dirty = _activity_age_days(v.branch or "", v.path)
    if dirty:
        raise RuntimeError(f"{v.branch} became dirty after classification")
    if age < archive_days:
        raise RuntimeError(
            f"{v.branch} activity age dropped below {archive_days}d ({age:.1f}d)"
        )
    _ensure_removal_side_effects_safe(v)

    _git("worktree", "remove", v.path)


def _reap(v: Verdict) -> None:
    """删除已合并的 worktree + 分支。删前重验分支 OID 未变,并用
    `update-ref -d <old-oid>` 原子删除 —— 分类后若另一 agent 在该分支提交了新
    工作,OID 会变、删除失败,绝不丢未合并的 commit。"""
    current = _git("rev-parse", f"refs/heads/{v.branch}")
    if current != v.head:
        raise RuntimeError(
            f"{v.branch} 在分类后被改动({v.head[:8]}→{current[:8]}),跳过回收"
        )
    _current_worktree_record(v)
    _age, dirty = _activity_age_days(v.branch or "", v.path)
    if dirty:
        raise RuntimeError(f"{v.branch} became dirty after classification")
    if not _is_merged(v.branch or ""):
        raise RuntimeError(f"{v.branch} is no longer absorbed by {MAIN}")
    _ensure_removal_side_effects_safe(v)
    _git("worktree", "remove", v.path)
    _git("update-ref", "-d", f"refs/heads/{v.branch}", v.head)


def main() -> int:
    ap = argparse.ArgumentParser(description="git worktree 体检 / 回收")
    ap.add_argument(
        "--reap", action="store_true",
        help="自动删除 MERGED 的 worktree+分支(无损)",
    )
    ap.add_argument(
        "--archive", action="store_true",
        help="归档超过 archive-days 的 clean checkout,保留 branch/HEAD",
    )
    ap.add_argument(
        "--archive-manifest",
        type=Path,
        help="archive action manifest；--archive 时必填，删除前先写 pending",
    )
    ap.add_argument("--stale-days", type=int, default=DEFAULT_STALE_DAYS)
    ap.add_argument("--archive-days", type=int, default=DEFAULT_ARCHIVE_DAYS)
    ap.add_argument("--json", action="store_true", help="JSON 输出(供 cron/告警)")
    args = ap.parse_args()
    if args.stale_days <= 0 or args.archive_days < args.stale_days:
        ap.error("require archive-days >= stale-days > 0")
    if args.archive and args.archive_manifest is None:
        ap.error("--archive requires --archive-manifest")
    archive_manifest = (
        _load_archive_manifest(args.archive_manifest)
        if args.archive_manifest is not None
        else None
    )

    verdicts: list[Verdict] = []
    skipped: list[dict] = []
    for wt in _list_worktrees():
        try:
            verdicts.append(_classify(wt, args.stale_days, args.archive_days))
        except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
            skipped.append(
                {
                    "path": wt.get("path"),
                    "branch": wt.get("branch"),
                    "reason": str(exc),
                }
            )

    reaped: list[str] = []
    archived: list[dict] = []
    if args.reap:
        for v in verdicts:
            if v.cls == "MERGED":
                try:
                    _reap(v)
                    reaped.append(v.path)
                except (subprocess.CalledProcessError, RuntimeError) as e:
                    # race 或 worktree 占用:跳过、记录,绝不强删
                    skipped.append({"path": v.path, "branch": v.branch, "reason": str(e)})
    if args.archive:
        assert args.archive_manifest is not None and archive_manifest is not None
        actions = archive_manifest["actions"]
        for v in verdicts:
            if v.cls != "ARCHIVABLE":
                continue
            action = _pending_archive_action(v)
            actions.append(action)
            archive_manifest["actions"] = actions[-MAX_ARCHIVE_ACTIONS:]
            _atomic_write_json(args.archive_manifest, archive_manifest)
            try:
                _archive(v, args.archive_days)
            except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
                action["result"] = "skipped"
                action["reason"] = str(e)
                action["completed_at"] = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
                _atomic_write_json(args.archive_manifest, archive_manifest)
                skipped.append({"path": v.path, "branch": v.branch, "reason": str(e)})
            else:
                action["result"] = "archived"
                action["completed_at"] = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
                _atomic_write_json(args.archive_manifest, archive_manifest)
                archived.append(asdict(v))

    report = {
        "reaped": reaped,
        "archived": archived,
        "skipped": skipped,
        "merged": [asdict(v) for v in verdicts if v.cls == "MERGED"],
        "stale": [asdict(v) for v in verdicts if v.cls == "STALE"],
        "stale_dirty": [asdict(v) for v in verdicts if v.cls == "STALE_DIRTY"],
        "archivable": [asdict(v) for v in verdicts if v.cls == "ARCHIVABLE"],
        "active": [asdict(v) for v in verdicts if v.cls == "ACTIVE"],
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    icon = {
        "MERGED": "✓ merged",
        "STALE": "⚠ stale ",
        "STALE_DIRTY": "⚠ dirty ",
        "ARCHIVABLE": "⌛ archive",
        "ACTIVE": "· active",
    }
    for v in verdicts:
        if v.cls == "SKIP":
            continue
        print(
            f"{icon[v.cls]}  {v.branch:42}  ahead {v.ahead:<3} behind {v.behind:<3} "
            f"age {v.age_days:>3.0f}d {'DIRTY' if v.dirty else ''}"
        )
    if report["stale"]:
        print(f"\n{len(report['stale'])} stale — review, then remove if dead:")
        for v in report["stale"]:
            print(f"  git worktree remove {v['path']} && git branch -D {v['branch']}")
    if reaped:
        print(f"\nReaped {len(reaped)} merged worktree(s).")
    if skipped:
        print(f"\nSkipped {len(skipped)} (moved since classification — re-run):")
        for s in skipped:
            print(f"  {s['branch']}: {s['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
