"""Deterministic efficiency-review support for the background skill review.

The background review historically looked only for correctness/preference
signals (user corrections, wrong skills, new techniques). It was blind to
*efficiency* waste — e.g. the email-triage cron re-reading a 1100-line runner
script in segments every single run to rediscover CLI flags (~19-36k tokens
per run, measured). Three reasons: long-transcript attention is unreliable,
the routed-digest path strips tool arguments, and single-session judgment
can't tell "one legitimate lookup" from "systematic recurring waste".

This module fixes all three with a strict division of labour — scripts do the
deterministic part, the review model does the judgment:

1. ``build_tool_call_digest`` — pure function over the message snapshot.
   Flags candidate waste patterns:
     • the same file read ≥``REREAD_THRESHOLD`` times via read_file
       (different offsets never trip the tool-level mtime dedup in
       file_tools.py, but "walk a source file in chunks" is the classic
       learn-flags-from-source waste);
     • identical (tool, arguments) calls repeated.
   Re-reads of a path the session itself mutated between reads
   (write_file/patch) are verification, not waste — exempted. A clean
   session returns ``None`` and the review prompt carries zero efficiency
   text (the gate: no cost when there is nothing to look at).

2. The sightings ledger (``.efficiency_observations.jsonl`` next to
   ``.usage.json``) — observe → recur → encode → verify-by-recurrence:
     • 1st sighting: recorded; the review is told to do NOTHING (one
       occurrence may be legitimate);
     • sighted in ≥2 distinct sessions: the review may encode an avoidance
       rule into the governing skill;
     • sighted again AFTER a rule landed: the review must NOT pile on more
       rules; a deterministic escalation line is surfaced to the user
       instead (the rule did not stick — a human needs to look).

3. ``apply_efficiency_outcome`` — post-review bookkeeping: appends
   escalation lines to the user-facing action summary and stamps
   ``encoded`` markers when the review actually wrote to a skill.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

READ_TOOL = "read_file"
WRITE_TOOLS = {"write_file", "patch"}

# Same-path read_file calls (any offsets) at or above this count flag the
# path. 2 segments is a normal way to read a long file once; 3+ is the
# walk-the-whole-source pattern.
REREAD_THRESHOLD = 3

# Identical (tool, args) calls at or above this count flag a duplicate.
DUPLICATE_THRESHOLD = 2

# How many largest tool results to surface as context in the digest.
TOP_RESULTS = 3

# Ledger hygiene: drop sightings older than this and cap total lines so the
# sidecar can't grow without bound.
LEDGER_MAX_AGE_DAYS = 120
LEDGER_MAX_LINES = 1000


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------

def _iter_tool_calls(messages: List[Dict]) -> List[Tuple[int, str, str, Dict]]:
    """Flatten assistant tool_calls into (seq, tool_call_id, name, args)."""
    out: List[Tuple[int, str, str, Dict]] = []
    seq = 0
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            out.append((seq, tc.get("id") or "", name, args))
            seq += 1
    return out


def _result_sizes(messages: List[Dict]) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        content = msg.get("content")
        if tcid and isinstance(content, str):
            sizes[tcid] = len(content)
    return sizes


def _args_hint(name: str, args: Dict) -> str:
    if "path" in args:
        return str(args["path"])
    if "command" in args:
        return str(args["command"]).replace("\n", " ")[:60]
    return json.dumps(args, sort_keys=True)[:60]


def build_tool_call_digest(messages_snapshot: List[Dict]) -> Optional[Dict[str, Any]]:
    """Deterministic tool-usage digest. ``None`` when nothing is suspicious.

    Returns ``{"lines": [...], "patterns": [{"key", "desc"}], "total_calls": n}``.
    """
    calls = _iter_tool_calls(messages_snapshot)
    if not calls:
        return None

    # Path-aggregated read_file grouping + mutation spans for exemption.
    reads_by_path: Dict[str, List[int]] = {}
    writes_by_path: Dict[str, List[int]] = {}
    for seq, _tcid, name, args in calls:
        path = args.get("path")
        if not isinstance(path, str) or not path:
            continue
        if name == READ_TOOL:
            reads_by_path.setdefault(path, []).append(seq)
        elif name in WRITE_TOOLS:
            writes_by_path.setdefault(path, []).append(seq)

    patterns: List[Dict[str, str]] = []
    lines: List[str] = []

    for path, seqs in sorted(reads_by_path.items()):
        if len(seqs) < REREAD_THRESHOLD:
            continue
        first, last = min(seqs), max(seqs)
        mutated_between = any(first < w < last for w in writes_by_path.get(path, []))
        if mutated_between:
            # The session edited this file between reads — re-reading is
            # normal verification. Exempt entirely (no noise).
            continue
        desc = (
            f"read_file '{path}' called {len(seqs)} times (different segments; "
            "no writes to this path in between)"
        )
        patterns.append({"key": f"reread:{path}", "desc": desc})
        lines.append(f"  • {desc}")

    # Identical (tool, canonical-args) duplicates.
    dup_counts: Dict[Tuple[str, str], List[int]] = {}
    canon_args: Dict[Tuple[str, str], str] = {}
    for seq, _tcid, name, args in calls:
        canon = json.dumps(args, sort_keys=True, ensure_ascii=False)
        key = (name, canon)
        dup_counts.setdefault(key, []).append(seq)
        canon_args[key] = canon
    for (name, canon), seqs in sorted(dup_counts.items()):
        if len(seqs) < DUPLICATE_THRESHOLD:
            continue
        # Path-aggregated re-reads already cover duplicated read_file calls.
        if name == READ_TOOL:
            try:
                path = json.loads(canon).get("path")
            except (json.JSONDecodeError, TypeError, AttributeError):
                path = None
            if path in reads_by_path and len(reads_by_path[path]) >= REREAD_THRESHOLD:
                continue
        digest8 = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:8]
        try:
            hint = _args_hint(name, json.loads(canon))
        except (json.JSONDecodeError, TypeError):
            hint = canon[:60]
        desc = f"{name} called {len(seqs)} times with identical arguments ({hint})"
        patterns.append({"key": f"dup:{name}:{digest8}", "desc": desc})
        lines.append(f"  • {desc}")

    if not patterns:
        return None

    sizes = _result_sizes(messages_snapshot)
    by_id = {tcid: (name, args) for _seq, tcid, name, args in calls if tcid}
    top = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:TOP_RESULTS]
    size_bits = []
    for tcid, n in top:
        name, args = by_id.get(tcid, ("?", {}))
        size_bits.append(f"{name} {_args_hint(name, args)} (~{n // 1000}K chars)"
                         if n >= 1000 else f"{name} {_args_hint(name, args)} ({n} chars)")
    header = f"TOOL DIGEST: {len(calls)} tool calls this session. Flagged patterns:"
    all_lines = [header] + lines
    if size_bits:
        all_lines.append("Largest tool results (context): " + "; ".join(size_bits))

    return {"lines": all_lines, "patterns": patterns, "total_calls": len(calls)}


# ---------------------------------------------------------------------------
# Sightings ledger
# ---------------------------------------------------------------------------

def ledger_path() -> Path:
    from tools.skill_usage import _skills_dir
    return _skills_dir() / ".efficiency_observations.jsonl"


def _load_ledger() -> List[Dict[str, Any]]:
    path = ledger_path()
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("efficiency ledger unreadable (%s); treating as empty", e)
        return []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("key") and entry.get("type"):
            entries.append(entry)
    return entries


def _write_ledger(entries: List[Dict[str, Any]]) -> None:
    cutoff = time.time() - LEDGER_MAX_AGE_DAYS * 86400
    kept = [e for e in entries if float(e.get("at", 0)) >= cutoff]
    if len(kept) > LEDGER_MAX_LINES:
        kept = kept[-LEDGER_MAX_LINES:]
    path = ledger_path()
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def record_and_classify(
    patterns: List[Dict[str, str]],
    session_id: str,
    run_family: str,
) -> List[Dict[str, str]]:
    """Record this session's sightings and classify each pattern.

    Returns ``[{"key", "desc", "status"}]`` with status one of
    ``observed`` / ``encode_now`` / ``recurred_after_encode``.
    """
    if not patterns:
        return []
    entries = _load_ledger()
    now = time.time()
    out: List[Dict[str, str]] = []
    for p in patterns:
        key = p["key"]
        history = [e for e in entries if e.get("key") == key]
        sessions = {e.get("session") for e in history if e.get("type") == "sighting"}
        encoded_at = max(
            (float(e.get("at", 0)) for e in history if e.get("type") == "encoded"),
            default=None,
        )
        if session_id not in sessions:
            entries.append({
                "type": "sighting", "key": key, "session": session_id,
                "family": run_family, "at": now, "desc": p.get("desc", ""),
            })
        if encoded_at is not None:
            status = "recurred_after_encode"
        elif len(sessions | {session_id}) >= 2:
            status = "encode_now"
        else:
            status = "observed"
        out.append({"key": key, "desc": p.get("desc", ""), "status": status})
    _write_ledger(entries)
    return out


def mark_encoded(keys: List[str]) -> None:
    """Stamp an ``encoded`` marker so later sightings escalate instead of
    piling on more rules."""
    if not keys:
        return
    entries = _load_ledger()
    now = time.time()
    for key in keys:
        entries.append({"type": "encoded", "key": key, "at": now})
    _write_ledger(entries)


# ---------------------------------------------------------------------------
# Prompt block + outcome
# ---------------------------------------------------------------------------

def run_family(session_id: Optional[str], platform: Optional[str]) -> str:
    """Stable grouping for recurrence counting. Cron sessions group by job id
    (``cron_<jobid>_<ts>`` → ``cron_<jobid>``); everything else by platform."""
    sid = session_id or ""
    if sid.startswith("cron_"):
        parts = sid.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
    return platform or "interactive"


_STATUS_INSTRUCTIONS = (
    "How to act on each status (hard rules):\n"
    "  • observed (1st sighting): do NOTHING about it — it is already "
    "recorded in the recurrence ledger. One occurrence may be legitimate; "
    "only recurring patterns earn a rule.\n"
    "  • encode_now (seen in multiple sessions): encode the avoidance into "
    "the skill that governs this class of task, following the constraints "
    "below.\n"
    "  • recurred_after_encode: do NOT add more rules — a previously "
    "encoded rule did not stick. The escalation is surfaced to the user "
    "automatically; you may mention the recurrence in your reply, nothing "
    "more.\n"
)

_HARD_CONSTRAINTS = (
    "Hard constraints for ANY efficiency-motivated skill edit:\n"
    "  • Every prohibition MUST name the cheaper replacement path "
    "(e.g. \"don't re-read X to learn its CLI flags — the parameter table "
    "is references/pipeline.md, read that one file\"). A ban without an "
    "exit hardens into a refusal.\n"
    "  • Prefer adding to `references/` (loaded on demand) over the "
    "SKILL.md body (always loaded — every added line is paid on every "
    "future run). If you must touch an always-loaded SKILL.md, REPLACE or "
    "MERGE existing text; do not grow it.\n"
    "  • Judge in hindsight, by what was actually USED downstream: if a "
    "file was read in 5 segments and only one flag was extracted, the "
    "waste is real; if the content genuinely drove decisions, it is not.\n"
    "  • Never remove evidence-gathering, verification gates, or "
    "review-quality steps in the name of efficiency.\n"
)


def build_efficiency_block(
    messages_snapshot: List[Dict],
    session_id: Optional[str],
    run_family: str,
) -> Tuple[Optional[str], Optional[Dict[str, List[str]]]]:
    """Assemble the efficiency prompt block and its outcome context.

    Returns ``(None, None)`` when the digest gate finds nothing — the review
    prompt then carries zero efficiency text.
    """
    digest = build_tool_call_digest(messages_snapshot)
    if digest is None:
        return None, None
    classified = record_and_classify(
        digest["patterns"], session_id=session_id or "unknown", run_family=run_family
    )
    ledger_lines = [f"  • {c['key']}: {c['status']}" for c in classified]
    block = (
        "\n\n--- EFFICIENCY REVIEW (deterministic tool-usage digest) ---\n"
        "This session's tool calls contained repeat patterns that MAY be "
        "avoidable waste. The digest below is computed mechanically; your "
        "job is the judgment call.\n"
        + "\n".join(digest["lines"]) + "\n"
        "Recurrence-ledger status per pattern (sightings counted across "
        "sessions):\n"
        + "\n".join(ledger_lines) + "\n"
        + _STATUS_INSTRUCTIONS
        + _HARD_CONSTRAINTS
    )
    ctx = {
        "encode_keys": [c["key"] for c in classified if c["status"] == "encode_now"],
        "escalations": [c["key"] for c in classified if c["status"] == "recurred_after_encode"],
    }
    return block, ctx


def _made_skill_write(review_messages: List[Dict], prior_snapshot: List[Dict]) -> bool:
    """True when the review itself completed a successful skill_manage write
    (create/edit/patch/write_file), ignoring results inherited from the prior
    conversation snapshot."""
    prior_ids = {
        m.get("tool_call_id")
        for m in prior_snapshot or []
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id")
    }
    write_call_ids = set()
    for _seq, tcid, name, args in _iter_tool_calls(review_messages):
        if name == "skill_manage" and args.get("action") in {"create", "edit", "patch", "write_file"}:
            if tcid:
                write_call_ids.add(tcid)
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if not tcid or tcid in prior_ids or tcid not in write_call_ids:
            continue
        try:
            data = json.loads(msg.get("content") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get("success"):
            return True
    return False


def apply_efficiency_outcome(
    efficiency_ctx: Optional[Dict[str, List[str]]],
    review_messages: List[Dict],
    prior_snapshot: List[Dict],
    actions: List[str],
) -> None:
    """Post-review bookkeeping: surface escalations to the user and stamp
    ``encoded`` markers when the review actually wrote a skill rule."""
    if not efficiency_ctx:
        return
    for key in efficiency_ctx.get("escalations") or []:
        actions.append(
            f"⚠️ Efficiency pattern '{key}' recurred AFTER a skill rule was "
            "added — the rule is not sticking; needs human attention"
        )
    encode_keys = efficiency_ctx.get("encode_keys") or []
    if encode_keys and _made_skill_write(review_messages, prior_snapshot):
        mark_encoded(encode_keys)


__all__ = [
    "build_tool_call_digest",
    "build_efficiency_block",
    "apply_efficiency_outcome",
    "record_and_classify",
    "mark_encoded",
    "ledger_path",
    "run_family",
]
