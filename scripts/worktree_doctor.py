#!/usr/bin/env python3
"""worktree-doctor — 体检 git worktree,回收已合并的、报告僵尸的。

确定性部分(脚本做):列出每个 worktree 的分支、相对 main 的 ahead/behind、
最后改动距今天数、有无未提交改动、改动是否已并入 main。
判断救/弃的模糊情况(STALE)只报告、不自动删,留给人/agent 看一眼。

自动回收(--reap)仅作用于 MERGED:分支自分叉点引入改动的文件,在 branch 与
main 上内容已一致(已被 squash 吸收)且无未提交改动 —— 这种删除是无损的。
判断不依赖 commit 祖先关系,所以对 squash-merge 稳健;且保守 —— main 若在
合并后又改了同一文件,会判 False(宁可不删,留给人看)。

多 agent 并发安全:分类时 pin 分支 OID,删除前重验且用 `update-ref -d <oid>`
原子删除 —— 分类后若另一 agent 在该分支提交了新工作,OID 变化会让删除失败,
绝不丢未合并的 commit。

配合 cron:--json 输出供告警管线消费;按运维告警约定,僵尸提醒路由到
#ops-alerts(不发生活助理频道)。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MAIN = "main"
REPO = str(Path.home() / ".hermes" / "hermes-agent")  # 固定操作生产仓库,不随调用方 cwd 漂移
DEFAULT_STALE_DAYS = 14


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


def _age_days(branch: str) -> float:
    ts = int(_git("log", "-1", "--format=%ct", branch))
    return (time.time() - ts) / 86400


def _dirty(path: str) -> bool:
    return bool(_git("status", "--porcelain", cwd=path))


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


def _classify(wt: dict, stale_days: int) -> Verdict:
    path, branch = wt["path"], wt.get("branch")
    if wt.get("is_main") or branch in (None, MAIN):
        # 主 worktree(gateway 生产)/ detached / main 本身:绝不回收
        return Verdict(path, branch, 0, 0, 0.0, False, "SKIP")
    if wt.get("prunable") or not (Path(path) / ".git").exists():
        # Broken admin records are Git's prune domain, not live worktrees. Skip
        # them so one stale metadata entry cannot abort the whole cron run.
        return Verdict(path, branch, 0, 0, 0.0, False, "SKIP")
    ahead, behind = _ahead_behind(branch)
    dirty = _dirty(path)
    age = _age_days(branch)
    head = _git("rev-parse", f"refs/heads/{branch}")  # pin,供 _reap 防 race
    if dirty:
        cls = "ACTIVE"            # 有未提交的活,绝不动
    elif _is_merged(branch):
        cls = "MERGED"            # 改动已进 main,无损可删
    elif age >= stale_days:
        cls = "STALE"             # 久未动 + 有独有改动,交人判断
    else:
        cls = "ACTIVE"
    return Verdict(path, branch, ahead, behind, age, dirty, cls, head)


def _reap(v: Verdict) -> None:
    """删除已合并的 worktree + 分支。删前重验分支 OID 未变,并用
    `update-ref -d <old-oid>` 原子删除 —— 分类后若另一 agent 在该分支提交了新
    工作,OID 会变、删除失败,绝不丢未合并的 commit。"""
    current = _git("rev-parse", f"refs/heads/{v.branch}")
    if current != v.head:
        raise RuntimeError(
            f"{v.branch} 在分类后被改动({v.head[:8]}→{current[:8]}),跳过回收"
        )
    _git("worktree", "remove", v.path)
    _git("update-ref", "-d", f"refs/heads/{v.branch}", v.head)


def main() -> int:
    ap = argparse.ArgumentParser(description="git worktree 体检 / 回收")
    ap.add_argument(
        "--reap", action="store_true",
        help="自动删除 MERGED 的 worktree+分支(无损);STALE 永不自动删",
    )
    ap.add_argument("--stale-days", type=int, default=DEFAULT_STALE_DAYS)
    ap.add_argument("--json", action="store_true", help="JSON 输出(供 cron/告警)")
    args = ap.parse_args()

    verdicts = [_classify(wt, args.stale_days) for wt in _list_worktrees()]

    reaped: list[str] = []
    skipped: list[dict] = []
    if args.reap:
        for v in verdicts:
            if v.cls == "MERGED":
                try:
                    _reap(v)
                    reaped.append(v.path)
                except (subprocess.CalledProcessError, RuntimeError) as e:
                    # race 或 worktree 占用:跳过、记录,绝不强删
                    skipped.append({"path": v.path, "branch": v.branch, "reason": str(e)})

    report = {
        "reaped": reaped,
        "skipped": skipped,
        "merged": [asdict(v) for v in verdicts if v.cls == "MERGED"],
        "stale": [asdict(v) for v in verdicts if v.cls == "STALE"],
        "active": [asdict(v) for v in verdicts if v.cls == "ACTIVE"],
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    icon = {"MERGED": "✓ merged", "STALE": "⚠ stale ", "ACTIVE": "· active"}
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
