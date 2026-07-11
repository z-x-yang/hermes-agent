"""Aggregate-only, read-only planning for state DB FTS migration and retention.

This module deliberately does not use :class:`hermes_state.SessionDB`: every SQLite
connection is opened with ``mode=ro`` and immediately placed in query-only mode.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from state_db_fts import detect_fts_schema
from state_db_maintenance import (
    TERMINAL_PHASES,
    load_maintenance_journal,
    maintenance_journal_path,
    state_db_file_inventory,
)

_GIB = 1024**3
_FTS_SUFFIXES = ("", "_data", "_idx", "_content", "_docsize", "_config")
_HANDLE_TOKEN_RE = re.compile(r"hermes://session/[^\s\"'<>]+")
_HANDLE_RE = re.compile(r"hermes://session/([^/\s]+)/message/([1-9][0-9]*)")
_TRAILING_TOKEN_PUNCTUATION = ".,;:!?)]}"
_PAYLOAD_FIELDS = (
    "content",
    "tool_calls",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
)


@dataclass(frozen=True)
class MigrationPlan:
    schema_kind: str
    schema_marker: str | None
    db_bytes: int
    wal_bytes: int
    shm_bytes: int
    free_bytes: int
    required_free_bytes: int
    message_count: int
    session_count: int
    archived_session_holds: int
    session_deletion_candidates: int
    fts_object_bytes: tuple[tuple[str, int], ...]
    writer_status: str
    maintenance_status: str
    can_apply: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RetentionEstimate:
    clock_status: str
    rows_by_age_basis: str
    rows_by_age: tuple[tuple[int, int], ...]
    logical_chars_by_age: tuple[tuple[int, int], ...]
    field_logical_chars: tuple[tuple[str, int], ...]
    valid_handle_targets: int
    handle_exemptions: int
    malformed_handles: int
    missing_handles: int
    wrong_handle_targets: int
    missing_or_wrong_targets: int
    archived_session_holds: int
    session_deletion_candidates: int


def _open_read_only(db_path: Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve(strict=False)
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
    except BaseException:
        conn.close()
        raise
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    # table names are module constants, never caller-controlled.
    return frozenset(str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})"))


def _schema_marker(conn: sqlite3.Connection) -> str | None:
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_meta'"
    ).fetchone()
    if table is None:
        return None
    row = conn.execute(
        "SELECT value FROM state_meta WHERE key='fts_schema_version'"
    ).fetchone()
    if row is None:
        return None
    marker = str(row[0])
    return marker if marker in {"1", "2"} else "invalid"


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _maintenance_summary(db_path: Path) -> tuple[str, dict[str, Any]]:
    journal_path = maintenance_journal_path(db_path)
    if not journal_path.exists():
        return "absent", {"journal_status": "absent"}
    try:
        journal = load_maintenance_journal(db_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "malformed", {
            "journal_status": "malformed",
            "journal_phase": None,
            "journal_fingerprints": {},
        }
    if journal is None:
        return "absent", {"journal_status": "absent"}
    terminal = journal.phase in TERMINAL_PHASES
    return ("terminal" if terminal else "active"), {
        "journal_status": "terminal" if terminal else "active",
        "journal_phase": journal.phase.value,
        "journal_fingerprints": {
            name: (
                None
                if value is None
                else {
                    key: item
                    for key, item in value.items()
                    if key in {"size", "mtime_ns", "device", "inode", "sha256"}
                }
            )
            for name, value in journal.fingerprints.items()
            if name in {"db", "wal", "shm", "backup", "work", "candidate"}
        },
    }


def _fts_bytes(conn: sqlite3.Connection) -> tuple[tuple[str, int], ...]:
    families: list[tuple[str, int]] = []
    for family, base in (
        ("unicode", "messages_fts"),
        ("trigram", "messages_fts_trigram"),
    ):
        names = tuple(f"{base}{suffix}" for suffix in _FTS_SUFFIXES)
        placeholders = ",".join("?" for _ in names)
        row = conn.execute(
            f"SELECT coalesce(sum(pgsize),0) FROM dbstat WHERE name IN ({placeholders})",
            names,
        ).fetchone()
        families.append((family, int(row[0] if row else 0)))
    return tuple(families)


def plan_fts_migration(db_path: Path) -> MigrationPlan:
    """Return a conservative migration plan without mutating SQLite or sidecars."""
    path = Path(db_path).expanduser().resolve(strict=False)
    conn = _open_read_only(path)
    try:
        schema_kind = detect_fts_schema(conn)
        marker = _schema_marker(conn)
        message_count = int(conn.execute("SELECT count(*) FROM messages").fetchone()[0])
        session_count = int(conn.execute("SELECT count(*) FROM sessions").fetchone()[0])
        session_columns = _columns(conn, "sessions")
        archived_holds = (
            int(conn.execute("SELECT count(*) FROM sessions WHERE archived=1").fetchone()[0])
            if "archived" in session_columns
            else 0
        )
        fts_bytes = _fts_bytes(conn)
    finally:
        conn.close()

    db_bytes = _file_size(path)
    wal_bytes = _file_size(Path(f"{path}-wal"))
    shm_bytes = _file_size(Path(f"{path}-shm"))
    required = 2 * (db_bytes + wal_bytes + shm_bytes) + 10 * _GIB
    free = shutil.disk_usage(path.parent).free
    maintenance_status, _ = _maintenance_summary(path)
    reasons: list[str] = []
    if schema_kind != "v1_inline":
        reasons.append(f"schema_not_migratable:{schema_kind}")
    if free < required:
        reasons.append("insufficient_free_space")
    if maintenance_status == "active":
        reasons.append("active_maintenance")
    elif maintenance_status == "malformed":
        reasons.append("malformed_maintenance_journal")
    # A read-only planner must not probe locks or infer that writers are stopped.
    reasons.append("writers_not_verified_stopped")
    return MigrationPlan(
        schema_kind=schema_kind,
        schema_marker=marker,
        db_bytes=db_bytes,
        wal_bytes=wal_bytes,
        shm_bytes=shm_bytes,
        free_bytes=free,
        required_free_bytes=required,
        message_count=message_count,
        session_count=session_count,
        archived_session_holds=archived_holds,
        session_deletion_candidates=0,
        fts_object_bytes=fts_bytes,
        writer_status="not_probed_read_only",
        maintenance_status=maintenance_status,
        can_apply=False,
        reasons=tuple(reasons),
    )


def _handle_counts(
    conn: sqlite3.Connection, message_columns: frozenset[str]
) -> tuple[int, int, int, int]:
    searchable_fields = tuple(
        field for field in ("content", "tool_name", "tool_calls") if field in message_columns
    )
    if not searchable_fields:
        return 0, 0, 0, 0
    eligibility = []
    if "active" in message_columns:
        eligibility.append("active=1")
    if "compacted" in message_columns:
        eligibility.append("compacted=1")
    where = " OR ".join(eligibility) or "1"
    targets: set[tuple[str, int]] = set()
    malformed = 0
    for row in conn.execute(f"SELECT {','.join(searchable_fields)} FROM messages WHERE {where}"):
        for value in row:
            if not isinstance(value, str):
                continue
            for token_match in _HANDLE_TOKEN_RE.finditer(value):
                token = token_match.group(0).rstrip(_TRAILING_TOKEN_PUNCTUATION)
                match = _HANDLE_RE.fullmatch(token)
                if match is None:
                    malformed += 1
                    continue
                session_id = unquote(match.group(1))
                if not session_id or "/" in session_id:
                    malformed += 1
                    continue
                targets.add((session_id, int(match.group(2))))

    valid = 0
    missing = 0
    wrong = 0
    has_tool_call_id = "tool_call_id" in message_columns
    for session_id, row_id in targets:
        select = "session_id, role" + (", tool_call_id" if has_tool_call_id else "")
        target = conn.execute(
            f"SELECT {select} FROM messages WHERE id=?", (row_id,)
        ).fetchone()
        target_valid = (
            target is not None
            and target[0] == session_id
            and target[1] == "tool"
            and has_tool_call_id
            and isinstance(target[2], str)
            and bool(target[2])
        )
        if target is None:
            missing += 1
        elif target_valid:
            valid += 1
        else:
            wrong += 1
    return valid, malformed, missing, wrong


def estimate_payload_retention(
    db_path: Path, age_days: tuple[int, ...] = (0, 1, 3, 7, 14)
) -> RetentionEstimate:
    """Estimate aggregate payload sizes; timestamp ages are never actionable."""
    if any(not isinstance(day, int) or day < 0 for day in age_days):
        raise ValueError("age_days must contain non-negative integers")
    path = Path(db_path).expanduser().resolve(strict=False)
    conn = _open_read_only(path)
    try:
        message_columns = _columns(conn, "messages")
        session_columns = _columns(conn, "sessions")
        clock_status = "available" if "compacted_at" in message_columns else "unavailable"
        fields = tuple(field for field in _PAYLOAD_FIELDS if field in message_columns)
        field_chars = tuple(
            (
                field,
                int(
                    conn.execute(
                        f"SELECT coalesce(sum(length({field})),0) FROM messages"
                    ).fetchone()[0]
                ),
            )
            for field in fields
        )
        now = time.time()
        rows_by_age: list[tuple[int, int]] = []
        chars_by_age: list[tuple[int, int]] = []
        payload_expr = "+".join(f"coalesce(length({field}),0)" for field in fields) or "0"
        eligibility = []
        if "active" in message_columns:
            eligibility.append("active=0")
        if "compacted" in message_columns:
            eligibility.append("compacted=1")
        base_where = " AND ".join(eligibility) or "1"
        for days in age_days:
            cutoff = now - days * 86400
            count, chars = conn.execute(
                f"SELECT count(*),coalesce(sum({payload_expr}),0) FROM messages "
                f"WHERE {base_where} AND timestamp<=?",
                (cutoff,),
            ).fetchone()
            rows_by_age.append((days, int(count)))
            chars_by_age.append((days, int(chars)))
        valid, malformed, missing, wrong = _handle_counts(conn, message_columns)
        archived_holds = (
            int(conn.execute("SELECT count(*) FROM sessions WHERE archived=1").fetchone()[0])
            if "archived" in session_columns
            else 0
        )
    finally:
        conn.close()
    return RetentionEstimate(
        clock_status=clock_status,
        rows_by_age_basis="non_actionable_upper_bound",
        rows_by_age=tuple(rows_by_age),
        logical_chars_by_age=tuple(chars_by_age),
        field_logical_chars=field_chars,
        valid_handle_targets=valid,
        handle_exemptions=valid,
        malformed_handles=malformed,
        missing_handles=missing,
        wrong_handle_targets=wrong,
        missing_or_wrong_targets=missing + wrong,
        archived_session_holds=archived_holds,
        session_deletion_candidates=0,
    )


def status_fts_migration(db_path: Path) -> dict[str, Any]:
    """Report safe aggregate schema, file and external-journal status."""
    path = Path(db_path).expanduser().resolve(strict=False)
    conn = _open_read_only(path)
    try:
        schema_kind = detect_fts_schema(conn)
        schema_marker = _schema_marker(conn)
        counts = {
            "messages": int(conn.execute("SELECT count(*) FROM messages").fetchone()[0]),
            "sessions": int(conn.execute("SELECT count(*) FROM sessions").fetchone()[0]),
        }
    finally:
        conn.close()
    maintenance_status, journal = _maintenance_summary(path)
    return {
        "schema_kind": schema_kind,
        "schema_marker": schema_marker,
        "counts": counts,
        "maintenance_status": maintenance_status,
        **journal,
        "file_fingerprints": state_db_file_inventory(path),
        "read_only": True,
    }
