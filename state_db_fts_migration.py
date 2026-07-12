"""Plan and verify explicit state DB FTS migration candidates.

Planning/status paths remain strictly read-only. Candidate construction requires an
issued maintenance permit, snapshots the source through SQLite's backup API, and
writes only isolated files under the caller-provided work directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from state_db_fts import (
    create_fts_v2,
    detect_fts_schema,
    integrity_check_fts_v2,
    rebuild_fts,
)
from state_db_maintenance import (
    TERMINAL_PHASES,
    JournalPhase,
    MaintenanceBlockedError,
    MaintenanceJournal,
    MaintenancePermit,
    assert_state_db_maintenance_access,
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

CONTROLLED_PAIRED_CORPUS_VERSION = "hermes-state-fts-controlled-v1"
_CONTROLLED_DIR_NAME = "controlled-paired-verification"
_CONTROLLED_TERMS = {
    "english": "cpx9d7b4e2a61f38",
    "unicode": "cafécpx7a91d5e3b62",
    "user_cjk": "受控检索甲辰玖",
    "assistant_cjk": "受控检索乙巳捌",
    "tool_calls_cjk": "受控检索丙午柒",
    "tool_mixed_like": "cpx5e81a2d9f工具",
    "tool_cjk_like": "验具",
    "short_cjk": "短验",
    "source": "cpx4c7e19a6b35d",
    "visibility": "cpx8a2f6d1e94b7",
    "lineage": "cpx3b9e71d5a26f",
}


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
    paired_corpus_version: str = CONTROLLED_PAIRED_CORPUS_VERSION


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


@dataclass(frozen=True)
class CandidateReport:
    """Aggregate-only evidence for a compact v2 file, never swap authority."""

    source_message_count: int
    candidate_message_count: int
    source_session_count: int
    candidate_session_count: int
    field_digest_equal: bool
    unicode_integrity: str
    trigram_integrity: str
    trigger_rollback_probe: str
    quick_check: str
    no_inline_content_shadows: bool
    candidate_wal_exists: bool
    candidate_shm_exists: bool
    candidate_sha256: str
    paired_verification_required: bool = True
    eligible_for_live_swap: bool = False


@dataclass(frozen=True)
class SearchCase:
    """One private paired-search input; reports retain only ``case_id``."""

    case_id: str
    category: str
    query: str
    role_filter: tuple[str, ...] = ()
    source_filter: tuple[str, ...] = ()
    exclude_sources: tuple[str, ...] = ()
    include_inactive: bool = False


@dataclass(frozen=True)
class SearchCaseReport:
    case_id: str
    category: str
    source_match_count: int
    candidate_match_count: int
    source_lineage_dedup_count: int
    candidate_lineage_dedup_count: int
    match_sets_equal: bool
    lineage_dedupe_equal: bool
    top10_overlap: float
    snippets_valid: bool
    ordering_difference_allowed: bool
    source_latency_ms: float
    candidate_latency_ms: float


@dataclass(frozen=True)
class LatencyReport:
    category: str
    case_count: int
    source_p50_ms: float
    source_p95_ms: float
    candidate_p50_ms: float
    candidate_p95_ms: float


@dataclass(frozen=True)
class VerificationReport:
    """Aggregate-only acceptance evidence; Task 7 still owns live-swap authority."""

    verification_passed: bool
    candidate_accepted: bool
    files_stable: bool
    field_digest_equal: bool
    row_counts_equal: bool
    all_match_sets_equal: bool
    all_lineage_dedupe_equal: bool
    minimum_top10_overlap: float
    all_snippets_valid: bool
    all_ordering_differences_allowed: bool
    candidate_schema_verified: bool
    unicode_integrity: str
    trigram_integrity: str
    trigger_rollback_probe: str
    quick_check: str
    no_inline_content_shadows: bool
    candidate_wal_exists: bool
    candidate_shm_exists: bool
    source_copy_sha256: str
    candidate_sha256: str
    cases: tuple[SearchCaseReport, ...]
    latency: tuple[LatencyReport, ...]
    ordering_difference_policy: str = "bm25_corpus_statistics_only"
    eligible_for_live_swap: bool = False


@dataclass(frozen=True)
class ControlledVerificationResult:
    """Aggregate-only controlled verification authority for the apply state machine."""

    paired_corpus_version: str
    verification: VerificationReport


def controlled_paired_corpus() -> tuple[SearchCase, ...]:
    """Return the versioned private corpus used only on controlled disposable rows."""
    terms = _CONTROLLED_TERMS
    return (
        SearchCase("english", "english", terms["english"]),
        SearchCase("english-unicode", "english", terms["unicode"]),
        SearchCase("default-user-cjk", "default_cjk", terms["user_cjk"], role_filter=("user", "assistant")),
        SearchCase("default-assistant-cjk", "default_cjk", terms["assistant_cjk"], role_filter=("user", "assistant")),
        SearchCase("assistant-toolcalls-cjk", "default_cjk", terms["tool_calls_cjk"], role_filter=("user", "assistant")),
        SearchCase("explicit-tool-english-like", "tool_cjk", terms["tool_mixed_like"], role_filter=("tool",)),
        SearchCase("explicit-tool-cjk-like", "tool_cjk", terms["tool_cjk_like"], role_filter=("tool",)),
        SearchCase("short-cjk", "default_cjk", terms["short_cjk"], role_filter=("user", "assistant")),
        SearchCase("source-include", "english", terms["source"], source_filter=("hermes-controlled-include",)),
        SearchCase("source-exclude", "english", terms["source"], exclude_sources=("hermes-controlled-exclude",)),
        SearchCase("active-compacted", "english", terms["visibility"]),
        SearchCase("include-inactive", "english", terms["visibility"], include_inactive=True),
        SearchCase("lineage-dedupe", "english", terms["lineage"]),
    )


_FTS_TABLE_NAMES = frozenset(
    f"{base}{suffix}"
    for base in ("messages_fts", "messages_fts_trigram")
    for suffix in _FTS_SUFFIXES
)
_FTS_TRIGGERS = (
    "messages_fts_insert",
    "messages_fts_delete",
    "messages_fts_update",
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
)


def _digest_value(digest: Any, value: object) -> None:
    if value is None:
        payload = b""
        tag = b"n"
    elif isinstance(value, bytes):
        payload = value
        tag = b"b"
    elif isinstance(value, str):
        payload = value.encode("utf-8", "surrogatepass")
        tag = b"s"
    elif isinstance(value, int):
        payload = str(value).encode("ascii")
        tag = b"i"
    elif isinstance(value, float):
        payload = value.hex().encode("ascii")
        tag = b"f"
    else:
        raise TypeError(f"unsupported SQLite value type: {type(value).__name__}")
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def field_digest(conn: sqlite3.Connection) -> str:
    """Digest all non-derived table fields, excluding only the v2 owner marker."""
    digest = hashlib.sha256()
    tables = tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        if str(row[0]) not in _FTS_TABLE_NAMES
    )
    for table in tables:
        quoted_table = f'"{table.replace(chr(34), chr(34) * 2)}"'
        columns = tuple(
            str(row[1]) for row in conn.execute(f"PRAGMA table_info({quoted_table})")
        )
        if not columns:
            raise RuntimeError(f"non-derived table has no visible fields: {table}")
        digest.update(table.encode("utf-8") + b"\0")
        for column in columns:
            _digest_value(digest, column)
        quoted = ",".join(
            f'"{column.replace(chr(34), chr(34) * 2)}"' for column in columns
        )
        where = " WHERE key <> 'fts_schema_version'" if table == "state_meta" else ""
        for row in conn.execute(
            f"SELECT {quoted} FROM {quoted_table}{where} ORDER BY {quoted}"
        ):
            digest.update(b"r")
            for value in row:
                _digest_value(digest, value)
    return digest.hexdigest()


def _source_truth_digest(conn: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    digest.update(field_digest(conn).encode("ascii"))
    for row in conn.execute(
        "SELECT type,name,tbl_name,coalesce(sql,'') FROM sqlite_master ORDER BY type,name"
    ):
        for value in row:
            _digest_value(digest, value)
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_meta'"
    ).fetchone():
        for row in conn.execute("SELECT key,value FROM state_meta ORDER BY key"):
            for value in row:
                _digest_value(digest, value)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_build_access(
    source: Path,
    journal: MaintenanceJournal,
    permit: MaintenancePermit | None,
) -> None:
    if permit is None:
        raise MaintenanceBlockedError(
            "fts-status: candidate build requires an issued maintenance permit"
        )
    assert_state_db_maintenance_access(source, write_capable=True, permit=permit)
    current = load_maintenance_journal(source)
    if current is None or current.to_dict() != journal.to_dict():
        raise MaintenanceBlockedError(
            "fts-status: candidate build journal does not match active journal"
        )
    if current.phase is not JournalPhase.BACKUP_READY:
        raise MaintenanceBlockedError(
            "fts-status: candidate build requires backup_ready phase"
        )


def _prepare_build_paths(work_dir: Path) -> tuple[Path, Path, Path]:
    raw = Path(work_dir).expanduser()
    try:
        info = raw.lstat()
    except FileNotFoundError as exc:
        raise ValueError("work_dir must already exist") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("work_dir must be a real directory, not a symlink")
    root = raw.resolve(strict=True)
    build_dir = root / "candidate-build"
    if build_dir.exists() or build_dir.is_symlink():
        raise FileExistsError("candidate build path already exists")
    build_dir.mkdir(mode=0o700)
    os.chmod(build_dir, 0o700)
    work = build_dir / "work.db"
    candidate = build_dir / "candidate.db"
    for path in (work, candidate):
        if path.exists() or path.is_symlink() or path.parent.resolve(strict=True) != build_dir:
            raise ValueError("candidate path escapes work_dir or collides")
    return build_dir, work, candidate


def _convert_work_copy_to_v2(conn: sqlite3.Connection) -> None:
    if detect_fts_schema(conn) != "v1_inline":
        raise RuntimeError("candidate source copy must be verified v1_inline")
    conn.execute("BEGIN IMMEDIATE")
    try:
        for trigger in _FTS_TRIGGERS:
            conn.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
        conn.execute("DROP TABLE messages_fts_trigram")
        conn.execute("DROP TABLE messages_fts")
        create_fts_v2(conn)
        conn.execute(
            "INSERT INTO state_meta(key,value) VALUES('fts_schema_version','2') "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
        )
        for table in ("messages_fts", "messages_fts_trigram"):
            conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
        integrity_check_fts_v2(conn)
        if detect_fts_schema(conn) != "v2_external":
            raise RuntimeError("converted work copy is not verified v2_external")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _verify_candidate_file(path: Path, expected_digest: str) -> tuple[int, int, str, bool]:
    conn = sqlite3.connect(path)
    try:
        if detect_fts_schema(conn) != "v2_external":
            raise RuntimeError("candidate schema is not verified v2_external")
        message_count = int(conn.execute("SELECT count(*) FROM messages").fetchone()[0])
        session_count = int(conn.execute("SELECT count(*) FROM sessions").fetchone()[0])
        if field_digest(conn) != expected_digest:
            raise RuntimeError("candidate field digest differs from source copy")
        shadows = int(
            conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name IN "
                "('messages_fts_content','messages_fts_trigram_content')"
            ).fetchone()[0]
        )
        if shadows:
            raise RuntimeError("candidate contains inline FTS content shadows")
        original_digest = field_digest(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT id FROM messages ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                raise RuntimeError("trigger rollback probe requires at least one message")
            conn.execute(
                "UPDATE messages SET content=coalesce(content,'') || ? WHERE id=?",
                (" candidate-trigger-probe", row[0]),
            )
            integrity_check_fts_v2(conn)
        finally:
            conn.rollback()
        if field_digest(conn) != original_digest:
            raise RuntimeError("trigger rollback probe changed candidate fields")
        quick = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        if quick != "ok":
            raise RuntimeError("candidate quick_check failed")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return message_count, session_count, quick, not bool(shadows)
    finally:
        conn.close()


def build_v2_candidate(
    source: Path,
    work_dir: Path,
    journal: MaintenanceJournal,
    permit: MaintenancePermit,
) -> CandidateReport:
    """Build an isolated compact v2 candidate from a consistent SQLite backup."""
    source_path = Path(source).expanduser()
    if source_path.is_symlink():
        raise ValueError("source must not be a symlink")
    source_path = source_path.resolve(strict=True)
    if not source_path.is_file():
        raise ValueError("source must be a regular database file")
    _validate_build_access(source_path, journal, permit)
    before_inventory = state_db_file_inventory(source_path)
    source_conn = _open_read_only(source_path)
    try:
        before_truth = _source_truth_digest(source_conn)
        source_messages = int(source_conn.execute("SELECT count(*) FROM messages").fetchone()[0])
        source_sessions = int(source_conn.execute("SELECT count(*) FROM sessions").fetchone()[0])
        source_fields = field_digest(source_conn)
        _, work_path, candidate_path = _prepare_build_paths(work_dir)
        work_fd = os.open(work_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(work_fd)
        work_conn = sqlite3.connect(work_path)
        try:
            source_conn.backup(work_conn)
        finally:
            work_conn.close()
    finally:
        source_conn.close()

    os.chmod(work_path, 0o600)
    candidate_fd = os.open(candidate_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(candidate_fd)
    work_conn = sqlite3.connect(work_path)
    try:
        _convert_work_copy_to_v2(work_conn)
        work_conn.execute("VACUUM INTO ?", (str(candidate_path),))
    finally:
        work_conn.close()
    os.chmod(candidate_path, 0o600)
    candidate_messages, candidate_sessions, quick, no_shadows = _verify_candidate_file(
        candidate_path, source_fields
    )
    with candidate_path.open("rb") as stream:
        os.fsync(stream.fileno())

    after_conn = _open_read_only(source_path)
    try:
        after_truth = _source_truth_digest(after_conn)
    finally:
        after_conn.close()
    if before_truth != after_truth or before_inventory != state_db_file_inventory(source_path):
        raise RuntimeError("source database or sidecars changed during candidate build")
    if (source_messages, source_sessions) != (candidate_messages, candidate_sessions):
        raise RuntimeError("candidate row counts differ from source copy")
    candidate_wal = Path(f"{candidate_path}-wal").exists()
    candidate_shm = Path(f"{candidate_path}-shm").exists()
    if candidate_wal or candidate_shm:
        raise RuntimeError("candidate sidecars remain after all handles closed")
    return CandidateReport(
        source_message_count=source_messages,
        candidate_message_count=candidate_messages,
        source_session_count=source_sessions,
        candidate_session_count=candidate_sessions,
        field_digest_equal=True,
        unicode_integrity="passed_rank1",
        trigram_integrity="passed_rank1",
        trigger_rollback_probe="passed",
        quick_check=quick,
        no_inline_content_shadows=no_shadows,
        candidate_wal_exists=False,
        candidate_shm_exists=False,
        candidate_sha256=_sha256_file(candidate_path),
    )


@dataclass(frozen=True)
class _SearchCopyResult:
    matches: list[dict]
    latency_ms: float
    lineage_survivors: tuple[tuple[str, str, int], ...]


def _lineage_survivor_identities(
    db: Any, raw_results: list[dict]
) -> tuple[tuple[str, str, int], ...]:
    """Apply canonical recall ordering and first-hit-per-lineage selection."""
    from tools.session_search_tool import _order_for_recall, _resolve_to_parent

    seen: set[str] = set()
    survivors: list[tuple[str, str, int]] = []
    for match in _order_for_recall(raw_results):
        session_id = str(match["session_id"])
        lineage_root = _resolve_to_parent(db, session_id)
        if lineage_root in seen:
            continue
        seen.add(lineage_root)
        survivors.append((lineage_root, session_id, int(match["id"])))
    return tuple(survivors)


def _search_copy(path: Path, cases: Sequence[SearchCase]) -> list[_SearchCopyResult]:
    from hermes_state import SessionDB

    conn = _open_read_only(path)
    conn.row_factory = sqlite3.Row
    db = object.__new__(SessionDB)
    db._conn = conn
    db._lock = threading.RLock()
    db._fts_enabled = True
    db._trigram_available = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages_fts_trigram'"
    ).fetchone() is not None
    total = int(conn.execute("SELECT count(*) FROM messages").fetchone()[0])
    results: list[_SearchCopyResult] = []
    try:
        for case in cases:
            started = time.perf_counter_ns()
            matches = db.search_messages(
                case.query,
                source_filter=list(case.source_filter) or None,
                exclude_sources=list(case.exclude_sources) or None,
                role_filter=list(case.role_filter or ("user", "assistant")),
                limit=total + 1,
                include_inactive=case.include_inactive,
            )
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            results.append(
                _SearchCopyResult(
                    matches=matches,
                    latency_ms=elapsed_ms,
                    lineage_survivors=_lineage_survivor_identities(db, matches),
                )
            )
    finally:
        conn.close()
    return results


def _top10_overlap(source_ids: list[int], candidate_ids: list[int]) -> float:
    source_top = source_ids[:10]
    candidate_top = candidate_ids[:10]
    denominator = min(10, max(len(source_top), len(candidate_top)))
    if denominator == 0:
        return 1.0
    return len(set(source_top).intersection(candidate_top)) / denominator


def _snippet_matches_query(snippet: object, query: str) -> bool:
    if not isinstance(snippet, str):
        return False
    marked = [value for value in re.findall(r">>>(.*?)<<<", snippet, re.DOTALL) if value]
    if not marked:
        return False
    tokens = [
        token.strip('"').rstrip("*")
        for token in query.split()
        if token.upper() not in {"AND", "OR", "NOT"}
    ]
    folded_tokens = [token.casefold() for token in tokens if token]
    return any(
        marked_value.casefold() in token or token in marked_value.casefold()
        for marked_value in marked
        for token in folded_tokens
    )


def _bm25_ordering_applies(case: SearchCase) -> bool:
    if case.category == "english":
        return True
    roles = set(case.role_filter or ("user", "assistant"))
    if case.category != "default_cjk" or not roles.issubset(
        {"user", "assistant"}
    ):
        return False
    tokens = [
        token
        for token in case.query.split()
        if token.upper() not in {"AND", "OR", "NOT"}
    ]
    counts = [
        sum(
            1
            for char in token
            if "\u3400" <= char <= "\u9fff"
            or "\u3040" <= char <= "\u30ff"
            or "\uac00" <= char <= "\ud7af"
        )
        for token in tokens
    ]
    return bool(counts) and all(count >= 3 for count in counts)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(
        0,
        min(len(ordered) - 1, int((len(ordered) * fraction) + 0.999999) - 1),
    )
    return round(ordered[index], 6)


def verify_v2_candidate(
    source_copy: Path,
    candidate: Path,
    corpus: Sequence[SearchCase],
) -> VerificationReport:
    """Compare complete old/new search results without serializing private values."""
    source_path = Path(source_copy).expanduser()
    candidate_path = Path(candidate).expanduser()
    if source_path.is_symlink() or candidate_path.is_symlink():
        raise ValueError("paired verification paths must not be symlinks")
    source_path = source_path.resolve(strict=True)
    candidate_path = candidate_path.resolve(strict=True)
    if source_path == candidate_path:
        raise ValueError("source_copy and candidate must be distinct files")
    cases = tuple(corpus)
    if not cases:
        raise ValueError("paired search corpus must not be empty")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)) or any(
        re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", case_id) is None for case_id in case_ids
    ):
        raise ValueError("case IDs must be unique safe identifiers")
    allowed_categories = {"english", "default_cjk", "tool_cjk"}
    if any(case.category not in allowed_categories for case in cases):
        raise ValueError("unsupported paired search latency category")
    if any(not isinstance(case.query, str) or not case.query.strip() for case in cases):
        raise ValueError("paired search queries must be non-empty strings")

    initial_source_sha256 = _sha256_file(source_path)
    initial_candidate_sha256 = _sha256_file(candidate_path)
    source_conn = _open_read_only(source_path)
    candidate_conn = _open_read_only(candidate_path)
    try:
        source_schema = detect_fts_schema(source_conn)
        candidate_schema = detect_fts_schema(candidate_conn)
        source_counts = (
            int(source_conn.execute("SELECT count(*) FROM messages").fetchone()[0]),
            int(source_conn.execute("SELECT count(*) FROM sessions").fetchone()[0]),
        )
        candidate_counts = (
            int(candidate_conn.execute("SELECT count(*) FROM messages").fetchone()[0]),
            int(candidate_conn.execute("SELECT count(*) FROM sessions").fetchone()[0]),
        )
        source_fields = field_digest(source_conn)
        candidate_fields = field_digest(candidate_conn)
    finally:
        source_conn.close()
        candidate_conn.close()
    if source_schema != "v1_inline":
        raise RuntimeError("paired source copy must be verified v1_inline")
    if candidate_schema != "v2_external":
        raise RuntimeError("paired candidate must be verified v2_external")

    candidate_messages, candidate_sessions, quick, no_shadows = _verify_candidate_file(
        candidate_path, candidate_fields
    )
    if candidate_counts != (candidate_messages, candidate_sessions):
        raise RuntimeError("candidate row counts changed during verification")
    source_results = _search_copy(source_path, cases)
    candidate_results = _search_copy(candidate_path, cases)
    candidate_wal = Path(f"{candidate_path}-wal").exists()
    candidate_shm = Path(f"{candidate_path}-shm").exists()
    final_source_sha256 = _sha256_file(source_path)
    final_candidate_sha256 = _sha256_file(candidate_path)
    files_stable = (
        initial_source_sha256 == final_source_sha256
        and initial_candidate_sha256 == final_candidate_sha256
    )
    case_reports: list[SearchCaseReport] = []
    for case, source_result, candidate_result in zip(
        cases, source_results, candidate_results
    ):
        source_matches = source_result.matches
        candidate_matches = candidate_result.matches
        source_ids = [int(match["id"]) for match in source_matches]
        candidate_ids = [int(match["id"]) for match in candidate_matches]
        snippets = [
            match.get("snippet") for match in (*source_matches, *candidate_matches)
        ]
        snippets_valid = all(
            _snippet_matches_query(snippet, case.query) for snippet in snippets
        )
        case_reports.append(
            SearchCaseReport(
                case_id=case.case_id,
                category=case.category,
                source_match_count=len(source_ids),
                candidate_match_count=len(candidate_ids),
                source_lineage_dedup_count=len(source_result.lineage_survivors),
                candidate_lineage_dedup_count=len(candidate_result.lineage_survivors),
                match_sets_equal=set(source_ids) == set(candidate_ids),
                lineage_dedupe_equal=(
                    source_result.lineage_survivors
                    == candidate_result.lineage_survivors
                ),
                top10_overlap=_top10_overlap(source_ids, candidate_ids),
                snippets_valid=snippets_valid,
                ordering_difference_allowed=(
                    source_ids == candidate_ids or _bm25_ordering_applies(case)
                ),
                source_latency_ms=round(source_result.latency_ms, 6),
                candidate_latency_ms=round(candidate_result.latency_ms, 6),
            )
        )

    latency: list[LatencyReport] = []
    for category in sorted(allowed_categories):
        category_reports = [item for item in case_reports if item.category == category]
        if not category_reports:
            continue
        latency.append(
            LatencyReport(
                category=category,
                case_count=len(category_reports),
                source_p50_ms=_percentile(
                    [item.source_latency_ms for item in category_reports], 0.50
                ),
                source_p95_ms=_percentile(
                    [item.source_latency_ms for item in category_reports], 0.95
                ),
                candidate_p50_ms=_percentile(
                    [item.candidate_latency_ms for item in category_reports], 0.50
                ),
                candidate_p95_ms=_percentile(
                    [item.candidate_latency_ms for item in category_reports], 0.95
                ),
            )
        )
    match_sets_equal = all(item.match_sets_equal for item in case_reports)
    lineage_dedupe_equal = all(item.lineage_dedupe_equal for item in case_reports)
    minimum_overlap = min(item.top10_overlap for item in case_reports)
    snippets_valid = all(item.snippets_valid for item in case_reports)
    ordering_valid = all(item.ordering_difference_allowed for item in case_reports)
    fields_equal = source_fields == candidate_fields
    row_counts_equal = source_counts == candidate_counts
    verified = (
        files_stable
        and fields_equal
        and row_counts_equal
        and match_sets_equal
        and lineage_dedupe_equal
        and minimum_overlap >= 0.9
        and snippets_valid
        and ordering_valid
        and no_shadows
        and quick == "ok"
        and not candidate_wal
        and not candidate_shm
    )
    return VerificationReport(
        verification_passed=verified,
        candidate_accepted=verified,
        files_stable=files_stable,
        field_digest_equal=fields_equal,
        row_counts_equal=row_counts_equal,
        all_match_sets_equal=match_sets_equal,
        all_lineage_dedupe_equal=lineage_dedupe_equal,
        minimum_top10_overlap=minimum_overlap,
        all_snippets_valid=snippets_valid,
        all_ordering_differences_allowed=ordering_valid,
        candidate_schema_verified=True,
        unicode_integrity="passed_rank1",
        trigram_integrity="passed_rank1",
        trigger_rollback_probe="passed",
        quick_check=quick,
        no_inline_content_shadows=no_shadows,
        candidate_wal_exists=candidate_wal,
        candidate_shm_exists=candidate_shm,
        source_copy_sha256=final_source_sha256,
        candidate_sha256=final_candidate_sha256,
        cases=tuple(case_reports),
        latency=tuple(latency),
    )


@dataclass(frozen=True)
class _OriginalInvariant:
    sha256: str
    truth_digest: str
    inventory: dict[str, Any]


def _snapshot_original(path: Path) -> _OriginalInvariant:
    conn = _open_read_only(path)
    try:
        truth = _source_truth_digest(conn)
    finally:
        conn.close()
    return _OriginalInvariant(
        sha256=_sha256_file(path),
        truth_digest=truth,
        inventory=state_db_file_inventory(path),
    )


def _assert_original_unchanged(path: Path, expected: _OriginalInvariant) -> None:
    if _snapshot_original(path) != expected:
        raise RuntimeError("original verification database changed")


def _prepare_controlled_dir(work_dir: Path) -> tuple[Path, Path, Path]:
    raw = Path(work_dir).expanduser()
    try:
        info = raw.lstat()
    except FileNotFoundError as exc:
        raise ValueError("controlled verification work_dir must already exist") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("controlled verification work_dir must be a real directory")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise PermissionError("controlled verification work_dir must have mode 0700")
    root = raw.resolve(strict=True)
    owned = root / _CONTROLLED_DIR_NAME
    try:
        owned.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("controlled verification path already exists")
    owned.mkdir(mode=0o700)
    os.chmod(owned, 0o700)
    source_copy = owned / "source.db"
    candidate_copy = owned / "candidate.db"
    return owned, source_copy, candidate_copy


def _backup_to_private_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("controlled verification copy path already exists")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    source_conn = _open_read_only(source)
    destination_conn = sqlite3.connect(destination)
    try:
        source_conn.backup(destination_conn)
    finally:
        destination_conn.close()
        source_conn.close()
    os.chmod(destination, 0o600)


def _insert_controlled_rows(
    source_copy: Path,
    candidate_copy: Path,
    session_ids: tuple[str, str, str],
) -> None:
    connections = [sqlite3.connect(source_copy), sqlite3.connect(candidate_copy)]
    try:
        schemas = [detect_fts_schema(conn) for conn in connections]
        if schemas != ["v1_inline", "v2_external"]:
            raise RuntimeError("controlled copies do not have the required v1/v2 schemas")
        required_sessions = {
            "id", "source", "model", "started_at", "ended_at", "archived", "parent_session_id"
        }
        required_messages = {
            "id", "session_id", "role", "content", "tool_call_id", "tool_calls",
            "tool_name", "timestamp", "active", "compacted"
        }
        for conn in connections:
            if not required_sessions.issubset(_columns(conn, "sessions")):
                raise RuntimeError("sessions schema cannot host controlled verification rows")
            if not required_messages.issubset(_columns(conn, "messages")):
                raise RuntimeError("messages schema cannot host controlled verification rows")
            projection = (
                "coalesce(content,'') || ' ' || coalesce(tool_name,'') || ' ' || "
                "coalesce(tool_calls,'')"
            )
            if any(
                conn.execute(
                    f"SELECT 1 FROM messages WHERE instr({projection}, ?) > 0 LIMIT 1",
                    (term,),
                ).fetchone()
                is not None
                for term in _CONTROLLED_TERMS.values()
            ):
                raise RuntimeError("controlled verification term collides with existing data")
            if any(
                conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
                is not None
                for session_id in session_ids
            ):
                raise RuntimeError("controlled verification session ID collision")
        next_ids = [
            int(conn.execute("SELECT coalesce(max(id),0) FROM messages").fetchone()[0]) + 1
            for conn in connections
        ]
        if next_ids[0] != next_ids[1]:
            raise RuntimeError("controlled copies do not share an aligned message ID space")
        base_id = next_ids[0]
        parent_id, child_id, excluded_id = session_ids
        session_rows = (
            (parent_id, "hermes-controlled-include", "controlled", 1.0, None, 0, None),
            (child_id, "hermes-controlled-include", "controlled", 2.0, None, 0, parent_id),
            (excluded_id, "hermes-controlled-exclude", "controlled", 3.0, None, 0, None),
        )
        terms = _CONTROLLED_TERMS
        message_rows = (
            (base_id, parent_id, "user", terms["english"], None, None, None, 10.0, 1, 0),
            (base_id + 1, parent_id, "assistant", terms["unicode"], None, None, None, 11.0, 1, 0),
            (base_id + 2, parent_id, "user", terms["user_cjk"], None, None, None, 12.0, 1, 0),
            (base_id + 3, parent_id, "assistant", terms["assistant_cjk"], None, None, None, 13.0, 1, 0),
            (base_id + 4, parent_id, "assistant", "", None, json.dumps({"name": terms["tool_calls_cjk"]}, ensure_ascii=False), None, 14.0, 1, 0),
            (base_id + 5, parent_id, "tool", terms["tool_mixed_like"], "controlled-call-1", None, "terminal", 15.0, 1, 0),
            (base_id + 6, parent_id, "tool", terms["tool_cjk_like"], "controlled-call-2", None, "terminal", 16.0, 1, 0),
            (base_id + 7, parent_id, "user", terms["short_cjk"], None, None, None, 17.0, 1, 0),
            (base_id + 8, parent_id, "user", terms["source"], None, None, None, 18.0, 1, 0),
            (base_id + 9, excluded_id, "user", terms["source"], None, None, None, 19.0, 1, 0),
            (base_id + 10, parent_id, "user", terms["visibility"], None, None, None, 20.0, 1, 0),
            (base_id + 11, parent_id, "assistant", terms["visibility"], None, None, None, 21.0, 0, 1),
            (base_id + 12, parent_id, "user", terms["visibility"], None, None, None, 22.0, 0, 0),
            (base_id + 13, parent_id, "user", terms["lineage"], None, None, None, 23.0, 1, 0),
            (base_id + 14, child_id, "assistant", terms["lineage"], None, None, None, 24.0, 1, 0),
        )
        for conn, schema in zip(connections, schemas):
            conn.executemany(
                "INSERT INTO sessions(id,source,model,started_at,ended_at,archived,parent_session_id) "
                "VALUES(?,?,?,?,?,?,?)",
                session_rows,
            )
            conn.executemany(
                "INSERT INTO messages(id,session_id,role,content,tool_call_id,tool_calls,"
                "tool_name,timestamp,active,compacted) VALUES(?,?,?,?,?,?,?,?,?,?)",
                message_rows,
            )
            if schema == "v1_inline":
                rebuild_fts(conn, "v1_inline")
            else:
                rebuild_fts(conn, "v2_external")
            conn.commit()
    except BaseException:
        for conn in connections:
            conn.rollback()
        raise
    finally:
        for conn in connections:
            conn.close()


def _controlled_semantics_passed(report: VerificationReport) -> bool:
    cases = {case.case_id: case for case in report.cases}
    expected_ids = {case.case_id for case in controlled_paired_corpus()}
    if set(cases) != expected_ids:
        return False
    if any(
        case.source_match_count <= 0 or case.candidate_match_count <= 0
        for case in cases.values()
    ):
        return False
    exact_counts = {
        "source-include": 1,
        "source-exclude": 1,
        "active-compacted": 2,
        "include-inactive": 3,
        "lineage-dedupe": 2,
    }
    if any(
        cases[case_id].source_match_count != count
        or cases[case_id].candidate_match_count != count
        for case_id, count in exact_counts.items()
    ):
        return False
    lineage = cases["lineage-dedupe"]
    return (
        lineage.source_lineage_dedup_count == 1
        and lineage.candidate_lineage_dedup_count == 1
    )


def _cleanup_controlled_dir(owned: Path) -> None:
    allowed = {
        "source.db", "source.db-journal", "source.db-wal", "source.db-shm",
        "candidate.db", "candidate.db-journal", "candidate.db-wal", "candidate.db-shm",
    }
    entries = list(os.scandir(owned))
    unexpected = [entry.name for entry in entries if entry.name not in allowed]
    if unexpected:
        raise RuntimeError("controlled verification directory contains unexpected paths")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise RuntimeError("controlled verification cleanup encountered unsafe path")
        os.unlink(entry.path)
    owned.rmdir()


def verify_v2_candidate_with_controlled_corpus(
    source_copy: Path,
    candidate: Path,
    work_dir: Path,
) -> ControlledVerificationResult:
    """Verify paired behavior on synthetic disposable copies without touching originals."""
    source_path = Path(source_copy).expanduser()
    candidate_path = Path(candidate).expanduser()
    if source_path.is_symlink() or candidate_path.is_symlink():
        raise ValueError("controlled verification inputs must not be symlinks")
    source_path = source_path.resolve(strict=True)
    candidate_path = candidate_path.resolve(strict=True)
    if source_path == candidate_path:
        raise ValueError("controlled verification inputs must be distinct")
    if not source_path.is_file() or not candidate_path.is_file():
        raise ValueError("controlled verification inputs must be regular files")
    source_before = _snapshot_original(source_path)
    candidate_before = _snapshot_original(candidate_path)
    owned, disposable_source, disposable_candidate = _prepare_controlled_dir(work_dir)
    try:
        _backup_to_private_copy(source_path, disposable_source)
        _backup_to_private_copy(candidate_path, disposable_candidate)
        session_ids = tuple(
            f"hermes-controlled-{secrets.token_hex(24)}" for _ in range(3)
        )
        _insert_controlled_rows(
            disposable_source,
            disposable_candidate,
            (session_ids[0], session_ids[1], session_ids[2]),
        )
        corpus = controlled_paired_corpus()
        report = verify_v2_candidate(disposable_source, disposable_candidate, corpus)
        if not _controlled_semantics_passed(report):
            report = replace(
                report,
                verification_passed=False,
                candidate_accepted=False,
            )
        return ControlledVerificationResult(
            paired_corpus_version=CONTROLLED_PAIRED_CORPUS_VERSION,
            verification=report,
        )
    finally:
        _cleanup_controlled_dir(owned)
        _assert_original_unchanged(source_path, source_before)
        _assert_original_unchanged(candidate_path, candidate_before)


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
