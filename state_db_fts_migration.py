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
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
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
    fingerprint_path,
    issue_maintenance_permit,
    load_maintenance_journal,
    maintenance_journal_path,
    state_db_file_inventory,
    write_maintenance_journal,
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

CONTROLLED_PAIRED_CORPUS_VERSION = "hermes-state-fts-controlled-v2"
_CONTROLLED_DIR_NAME = "controlled-paired-verification"
_CONTROLLED_TERM_KEYS = (
    "english",
    "unicode",
    "user_cjk",
    "assistant_cjk",
    "tool_calls_cjk",
    "tool_mixed_like",
    "tool_cjk_like",
    "short_cjk",
    "source",
    "visibility",
    "lineage",
)


def _random_cjk(length: int) -> str:
    return "".join(chr(0x4E00 + secrets.randbelow(0x9FFF - 0x4E00 + 1)) for _ in range(length))


def _new_controlled_terms() -> dict[str, str]:
    """Generate a payload-independent namespace that cannot become stale corpus data."""
    nonce = secrets.token_hex(16)
    terms = {
        "english": f"cpx{nonce}a",
        "unicode": f"café{nonce}b",
        "user_cjk": _random_cjk(8),
        "assistant_cjk": _random_cjk(8),
        "tool_calls_cjk": _random_cjk(8),
        "tool_mixed_like": f"cpx{nonce}c{_random_cjk(2)}",
        "tool_cjk_like": _random_cjk(2),
        "short_cjk": _random_cjk(2),
        "source": f"cpx{nonce}d",
        "visibility": f"cpx{nonce}e",
        "lineage": f"cpx{nonce}f",
    }
    if set(terms) != set(_CONTROLLED_TERM_KEYS) or len(set(terms.values())) != len(terms):
        raise RuntimeError("controlled verification namespace generation failed")
    return terms


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


@dataclass(frozen=True)
class LivenessReport:
    """Aggregate-only proof that no process has a state DB bundle open."""

    status: str
    holder_count: int


@dataclass(frozen=True)
class MigrationResult:
    """Aggregate-only terminal result; paths and operation identifiers stay private."""

    phase: str
    completed: bool
    paired_corpus_version: str = CONTROLLED_PAIRED_CORPUS_VERSION


def controlled_paired_corpus(
    terms: Mapping[str, str] | None = None,
) -> tuple[SearchCase, ...]:
    """Return one fresh, versioned corpus used only on controlled disposable rows."""
    terms = dict(terms) if terms is not None else _new_controlled_terms()
    if set(terms) != set(_CONTROLLED_TERM_KEYS):
        raise ValueError("controlled verification terms do not match the corpus contract")
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
        mode_row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if mode_row is None or str(mode_row[0]).lower() != "wal":
            raise RuntimeError("candidate could not persist WAL journal mode")
        return message_count, session_count, quick, not bool(shadows)
    finally:
        conn.close()


def _checkpoint_candidate_file(path: Path) -> None:
    """Checkpoint only after the verifier's SQLite handle is fully closed."""
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        checkpoint = (
            tuple(int(value) for value in row)
            if row is not None and len(row) == 3
            else None
        )
        if checkpoint != (0, 0, 0):
            raise RuntimeError("candidate checkpoint did not return exact (0,0,0)")
    finally:
        conn.close()


_WAL_INDEX_READ_MARK_START = 100
_WAL_INDEX_READ_MARK_END = 120


def _read_optional_file(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _same_candidate_build_source_inventory(
    before: Mapping[str, object],
    after: Mapping[str, object],
    before_shm: bytes | None,
    after_shm: bytes | None,
) -> bool:
    """Compare source stability around known read-only SQLite handles.

    SQLite may update the five ``aReadMark`` reader slots (bytes 100:120 in
    the WAL-index header) and SHM mtime even for a ``mode=ro`` connection.
    Main is exact; WAL identity, size, and bytes are exact; SHM may differ only
    in those reader slots while preserving presence, device, inode, and size.
    This tolerance is intentionally local to known read-only verification
    handles; rename/swap/rollback classifiers keep exact fingerprints.
    """

    def same_keys(name: str, keys: tuple[str, ...]) -> bool:
        left = before.get(name)
        right = after.get(name)
        if left is None or right is None:
            return left is None and right is None
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return all(left.get(key) == right.get(key) for key in keys)

    if not (
        same_keys("db", ("size", "mtime_ns", "device", "inode", "sha256"))
        and same_keys("wal", ("size", "device", "inode", "sha256"))
        and same_keys("shm", ("size", "device", "inode"))
    ):
        return False
    before_fp = before.get("shm")
    after_fp = after.get("shm")
    if before_fp is None and after_fp is None:
        return True
    if not isinstance(before_fp, Mapping) or not isinstance(after_fp, Mapping):
        return False
    if before_fp.get("sha256") == after_fp.get("sha256"):
        return True
    if before_shm is None or after_shm is None or len(before_shm) != len(after_shm):
        return False
    return all(
        _WAL_INDEX_READ_MARK_START <= offset < _WAL_INDEX_READ_MARK_END
        for offset, (left, right) in enumerate(zip(before_shm, after_shm))
        if left != right
    )


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
    source_shm = Path(f"{source_path}-shm")
    before_shm = _read_optional_file(source_shm)
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
    _checkpoint_candidate_file(candidate_path)
    with candidate_path.open("rb") as stream:
        os.fsync(stream.fileno())

    after_conn = _open_read_only(source_path)
    try:
        after_truth = _source_truth_digest(after_conn)
    finally:
        after_conn.close()
    after_inventory = state_db_file_inventory(source_path)
    after_shm = _read_optional_file(source_shm)
    if before_truth != after_truth or not _same_candidate_build_source_inventory(
        before_inventory, after_inventory, before_shm, after_shm
    ):
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
    _checkpoint_candidate_file(candidate_path)
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
    shm_bytes: bytes | None


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
        shm_bytes=_read_optional_file(Path(f"{path}-shm")),
    )


def _assert_original_unchanged(path: Path, expected: _OriginalInvariant) -> None:
    actual = _snapshot_original(path)
    if (
        actual.sha256 != expected.sha256
        or actual.truth_digest != expected.truth_digest
        or not _same_candidate_build_source_inventory(
            expected.inventory,
            actual.inventory,
            expected.shm_bytes,
            actual.shm_bytes,
        )
    ):
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
    terms: Mapping[str, str],
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
                for term in terms.values()
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


def _controlled_semantics_passed(
    report: VerificationReport,
    corpus: Sequence[SearchCase],
) -> bool:
    cases = {case.case_id: case for case in report.cases}
    expected_ids = {case.case_id for case in corpus}
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
        terms = _new_controlled_terms()
        session_ids = tuple(
            f"hermes-controlled-{secrets.token_hex(24)}" for _ in range(3)
        )
        _insert_controlled_rows(
            disposable_source,
            disposable_candidate,
            (session_ids[0], session_ids[1], session_ids[2]),
            terms,
        )
        corpus = controlled_paired_corpus(terms)
        report = verify_v2_candidate(disposable_source, disposable_candidate, corpus)
        if not _controlled_semantics_passed(report, corpus):
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


def _run_lsof(paths: tuple[Path, ...]):
    """Private deterministic seam for the production lsof invocation."""
    return subprocess.run(
        ["lsof", "-Fpf", "--", *(str(path) for path in paths)],
        capture_output=True,
        text=True,
        check=False,
    )


def _find_live_state_db_users_for_paths(paths: Sequence[Path]) -> LivenessReport:
    """Prove aggregate liveness for one deduplicated extant recovery path set."""
    extant_paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in paths:
        normalized = Path(candidate).expanduser().resolve(strict=False)
        if normalized.exists() and normalized not in seen:
            seen.add(normalized)
            extant_paths.append(normalized)
    paths = tuple(extant_paths)
    if not paths:
        raise RuntimeError("lsof cannot prove liveness for a missing database bundle")
    try:
        result = _run_lsof(paths)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("lsof unavailable; writer liveness is unknown") from exc
    output = result.stdout
    if result.returncode == 1 and not output.strip():
        return LivenessReport(status="clear", holder_count=0)
    if result.returncode != 0:
        raise RuntimeError("lsof failed; writer liveness is unknown")
    if not output.strip():
        raise RuntimeError("lsof returned ambiguous empty success")
    holders: set[int] = set()
    active_pid: int | None = None
    for line in output.splitlines():
        if not line:
            continue
        field, value = line[0], line[1:]
        if field == "p" and value.isdigit():
            active_pid = int(value)
            holders.add(active_pid)
        elif field == "f" and active_pid is not None and value:
            continue
        else:
            raise RuntimeError("lsof returned ambiguous machine output")
    if not holders:
        raise RuntimeError("lsof returned no process records")
    return LivenessReport(status="live", holder_count=len(holders))


def find_live_state_db_users(db_path: Path) -> LivenessReport:
    """Return aggregate liveness for the normal live database bundle."""
    path = Path(db_path).expanduser().resolve(strict=False)
    return _find_live_state_db_users_for_paths(tuple(_bundle_paths(path).values()))


def _require_no_live_users(db_path: Path) -> None:
    if find_live_state_db_users(db_path).status != "clear":
        raise RuntimeError("state DB writers are still live")


def _require_no_live_users_for_paths(paths: Sequence[Path]) -> None:
    if _find_live_state_db_users_for_paths(paths).status != "clear":
        raise RuntimeError("state DB writers are still live")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _transition(
    db_path: Path,
    journal: MaintenanceJournal,
    phase: JournalPhase,
    *,
    fingerprints: dict[str, dict[str, int | str] | None] | None = None,
    backup_path: Path | None = None,
    work_path: Path | None = None,
    candidate_path: Path | None = None,
    expected_row_counts: dict[str, int] | None = None,
) -> MaintenanceJournal:
    merged = dict(journal.fingerprints)
    if fingerprints:
        merged.update(fingerprints)
    updated = replace(
        journal,
        phase=phase,
        backup_path=str(backup_path) if backup_path is not None else journal.backup_path,
        work_path=str(work_path) if work_path is not None else journal.work_path,
        candidate_path=str(candidate_path) if candidate_path is not None else journal.candidate_path,
        fingerprints=merged,
        expected_row_counts=(
            expected_row_counts
            if expected_row_counts is not None
            else dict(journal.expected_row_counts)
        ),
        updated_at=str(time.time_ns()),
    )
    write_maintenance_journal(db_path, updated)
    return updated


def _permit(db_path: Path, journal: MaintenanceJournal) -> MaintenancePermit:
    return issue_maintenance_permit(
        db_path, journal.operation_id, frozenset({journal.phase})
    )


def _checkpoint_source(
    db_path: Path, journal: MaintenanceJournal, permit: MaintenancePermit
) -> tuple[int, int, int]:
    assert_state_db_maintenance_access(db_path, write_capable=True, permit=permit)
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if row is None or len(row) != 3:
            raise RuntimeError("checkpoint returned an invalid result")
        return (int(row[0]), int(row[1]), int(row[2]))
    finally:
        conn.close()


def _same_fingerprint(
    actual: dict[str, int | str] | None,
    expected: object,
) -> bool:
    if actual is None or not isinstance(expected, Mapping):
        return actual is None and expected is None
    return all(
        actual.get(key) == expected.get(key)
        for key in ("size", "mtime_ns", "device", "inode", "sha256")
    )


def _require_fingerprint(path: Path, expected: object, label: str) -> None:
    if not _same_fingerprint(fingerprint_path(path), expected):
        raise RuntimeError(f"unknown {label} fingerprint; no files changed")


def _exact_two_location_state(
    source: Path,
    destination: Path,
    expected: object,
    label: str,
) -> str:
    """Classify an idempotent rename only when the other location is absent."""
    source_fp = fingerprint_path(source)
    destination_fp = fingerprint_path(destination)
    if expected is None:
        if source_fp is None and destination_fp is None:
            return "absent"
    elif _same_fingerprint(source_fp, expected) and destination_fp is None:
        return "source"
    elif _same_fingerprint(destination_fp, expected) and source_fp is None:
        return "destination"
    raise RuntimeError(f"unknown {label} state; no files changed")


def _status_fingerprint(status: str) -> dict[str, int | str]:
    """Store a schema-compatible explicit durable status marker."""
    return {
        "size": 0,
        "mtime_ns": 0,
        "device": 0,
        "inode": 0,
        "sha256": status,
    }


def _bundle_paths(db_path: Path) -> dict[str, Path]:
    return {
        "db": db_path,
        "wal": Path(f"{db_path}-wal"),
        "shm": Path(f"{db_path}-shm"),
    }


def _candidate_bundle_paths(candidate: Path) -> dict[str, Path]:
    return {
        "db": candidate,
        "wal": Path(f"{candidate}-wal"),
        "shm": Path(f"{candidate}-shm"),
    }


def _candidate_inventory_fingerprints(
    candidate: Path,
) -> dict[str, dict[str, int | str] | None]:
    return {
        f"candidate_{name}": fingerprint_path(item)
        for name, item in _candidate_bundle_paths(candidate).items()
    }


def _validate_candidate_bundle(candidate: Path, journal: MaintenanceJournal) -> None:
    for name, item in _candidate_bundle_paths(candidate).items():
        key = f"candidate_{name}"
        if key not in journal.fingerprints:
            raise RuntimeError(f"candidate {name} inventory missing from journal")
        _require_fingerprint(item, journal.fingerprints[key], f"candidate {name}")


def _original_paths(db_path: Path) -> dict[str, Path]:
    return {
        "db": db_path.with_name(f"{db_path.name}.pre-v2.original"),
        "wal": db_path.with_name(f"{db_path.name}.pre-v2.original-wal"),
        "shm": db_path.with_name(f"{db_path.name}.pre-v2.original-shm"),
    }


def _quarantine_paths(db_path: Path) -> dict[str, Path]:
    return {
        "db": db_path.with_name(f"{db_path.name}.v2.quarantine"),
        "wal": db_path.with_name(f"{db_path.name}.v2.quarantine-wal"),
        "shm": db_path.with_name(f"{db_path.name}.v2.quarantine-shm"),
    }


def _source_inventory_fingerprints(db_path: Path) -> dict[str, dict[str, int | str] | None]:
    inventory = state_db_file_inventory(db_path)
    return {name: inventory[name] for name in ("db", "wal", "shm")}


def _validate_live_source(db_path: Path, journal: MaintenanceJournal) -> None:
    for name, path in _bundle_paths(db_path).items():
        _require_fingerprint(path, journal.fingerprints.get(name), f"source {name}")


_REQUIRED_SESSION_COLUMNS = frozenset(
    {"id", "source", "model", "started_at", "ended_at", "archived", "parent_session_id"}
)
_REQUIRED_MESSAGE_COLUMNS = frozenset(
    {
        "id", "session_id", "role", "content", "tool_call_id", "tool_calls",
        "tool_name", "timestamp", "active", "compacted",
    }
)


def _require_expected_base_schema(conn: sqlite3.Connection) -> None:
    if not _REQUIRED_SESSION_COLUMNS.issubset(_columns(conn, "sessions")):
        raise RuntimeError("live database has an unexpected sessions schema")
    if not _REQUIRED_MESSAGE_COLUMNS.issubset(_columns(conn, "messages")):
        raise RuntimeError("live database has an unexpected messages schema")


def _require_live_main_lineage(db_path: Path, journal: MaintenanceJournal) -> None:
    """Allow content drift only while retaining the accepted live main inode."""
    expected = journal.fingerprints.get("db")
    actual = fingerprint_path(db_path)
    if not isinstance(expected, Mapping) or actual is None or any(
        actual.get(key) != expected.get(key) for key in ("device", "inode")
    ):
        raise RuntimeError("unknown live candidate lineage; no files changed")


def _verify_v2_rollback_database(
    db_path: Path, permit: MaintenancePermit
) -> None:
    assert_state_db_maintenance_access(db_path, write_capable=True, permit=permit)
    conn = sqlite3.connect(db_path)
    try:
        marker = conn.execute(
            "SELECT value FROM state_meta WHERE key='fts_schema_version'"
        ).fetchone()
        if marker != ("2",) or detect_fts_schema(conn) != "v2_external":
            raise RuntimeError("live rollback candidate is not verified v2_external")
        _require_expected_base_schema(conn)
        if conn.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise RuntimeError("live rollback candidate quick_check failed")
        integrity_check_fts_v2(conn)
    finally:
        conn.close()


def _verify_active_v2_for_rollback(
    db_path: Path, journal: MaintenanceJournal
) -> MaintenanceJournal:
    """Recapture an ordinary-use v2 bundle only after rollback is write-blocked."""
    _require_live_main_lineage(db_path, journal)
    permit = _permit(db_path, journal)
    # Reject wrong ownership/base schema/corruption before a checkpoint can
    # modify the current bundle, then repeat against checkpointed truth.
    _verify_v2_rollback_database(db_path, permit)
    checkpoint = _checkpoint_source(db_path, journal, permit)
    if checkpoint != (0, 0, 0):
        raise RuntimeError("rollback checkpoint did not return exact (0,0,0)")

    # The checkpoint handle is closed by _checkpoint_source before this fresh
    # verification handle is opened. Any sidecars materialized by verification
    # are deliberately included in the exact inventory captured below.
    _verify_v2_rollback_database(db_path, permit)

    return _transition(
        db_path,
        journal,
        journal.phase,
        fingerprints={
            f"rollback_{name}": fingerprint_path(item)
            for name, item in _bundle_paths(db_path).items()
        },
    )


def _create_sqlite_backup(
    source: Path,
    destination: Path,
    permit: MaintenancePermit,
) -> None:
    assert_state_db_maintenance_access(source, write_capable=True, permit=permit)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("backup path already exists")
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
    check = _open_read_only(destination)
    try:
        if check.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise RuntimeError("rollback backup quick_check failed")
    finally:
        check.close()
    _fsync_file(destination)
    _fsync_directory(destination.parent)


def _migration_paths(db_path: Path) -> tuple[Path, Path, Path]:
    backup = db_path.with_name(f"{db_path.name}.pre-v2.backup")
    work = db_path.parent / f".{db_path.name}.fts-v2-work"
    candidate = work / "candidate-build" / "candidate.db"
    return backup, work, candidate


def _move_recorded_bundle_to_original(
    db_path: Path, journal: MaintenanceJournal
) -> MaintenanceJournal:
    live = _bundle_paths(db_path)
    original = _original_paths(db_path)
    actions: list[tuple[Path, Path]] = []
    for name in ("db", "wal", "shm"):
        expected = journal.fingerprints.get(name)
        state = _exact_two_location_state(
            live[name], original[name], expected, f"source {name} rename"
        )
        if state == "source":
            actions.append((live[name], original[name]))
    for source, destination in actions:
        os.replace(source, destination)
        _fsync_directory(db_path.parent)
    originals = {
        f"original_{name}": fingerprint_path(path) for name, path in original.items()
    }
    return _transition(db_path, journal, JournalPhase.OLD_MOVED, fingerprints=originals)


def _install_recorded_candidate(
    db_path: Path, journal: MaintenanceJournal
) -> MaintenanceJournal:
    if journal.candidate_path is None:
        raise RuntimeError("candidate path missing from journal")
    candidate = Path(journal.candidate_path)
    expected = journal.fingerprints.get("candidate_db")
    for name in ("wal", "shm"):
        key = f"candidate_{name}"
        if key not in journal.fingerprints:
            raise RuntimeError(f"candidate {name} inventory missing from journal")
        _require_fingerprint(
            _candidate_bundle_paths(candidate)[name],
            journal.fingerprints[key],
            f"candidate {name}",
        )
    live = _bundle_paths(db_path)
    for name in ("wal", "shm"):
        if live[name].exists():
            raise RuntimeError("stale sidecar remains at live basename")
    state = _exact_two_location_state(
        candidate, db_path, expected, "candidate rename"
    )
    if state == "source":
        os.replace(candidate, db_path)
        _fsync_directory(db_path.parent)
    return _transition(
        db_path,
        journal,
        JournalPhase.CANDIDATE_LIVE,
        fingerprints=_source_inventory_fingerprints(db_path),
    )


def _run_v2_canary(db_path: Path, journal: MaintenanceJournal) -> None:
    permit = _permit(db_path, journal)
    assert_state_db_maintenance_access(db_path, write_capable=True, permit=permit)
    conn = sqlite3.connect(db_path)
    token = f"hermescanary{secrets.token_hex(12)}"
    try:
        if detect_fts_schema(conn) != "v2_external":
            raise RuntimeError("live candidate is not v2_external")
        row = conn.execute("SELECT id FROM messages ORDER BY id LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("canary requires a message row")
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "UPDATE messages SET content=coalesce(content,'') || ? WHERE id=?",
                (f" {token}", row[0]),
            )
            if conn.execute("SELECT content FROM messages WHERE id=?", (row[0],)).fetchone() is None:
                raise RuntimeError("canary read failed")
            found = conn.execute(
                "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
                (f'"{token}"',),
            ).fetchone()
            if found is None:
                raise RuntimeError("canary search failed")
        finally:
            conn.rollback()
    finally:
        conn.close()
    permit = _permit(db_path, journal)
    checkpoint = _checkpoint_source(db_path, journal, permit)
    if checkpoint != (0, 0, 0):
        raise RuntimeError("canary checkpoint did not return exact (0,0,0)")


def apply_fts_migration(db_path: Path) -> MigrationResult:
    """Start and fully execute the explicit recoverable v1-to-v2 migration."""
    path = Path(db_path).expanduser()
    if path.is_symlink():
        raise ValueError("database must not be a symlink")
    path = path.resolve(strict=True)
    if load_maintenance_journal(path) is not None:
        raise RuntimeError("maintenance journal already exists; use resume")
    _require_no_live_users(path)
    plan = plan_fts_migration(path)
    if plan.schema_kind != "v1_inline":
        raise RuntimeError("only a verified v1_inline database can be migrated")
    if plan.free_bytes < plan.required_free_bytes:
        raise RuntimeError("insufficient free space for migration")
    operation_id = secrets.token_hex(32)
    inventory = _source_inventory_fingerprints(path)
    journal = replace(
        MaintenanceJournal.new(operation_id, path),
        fingerprints=inventory,
        expected_row_counts={"messages": plan.message_count, "sessions": plan.session_count},
    )
    write_maintenance_journal(path, journal)
    return resume_fts_migration(path)


def resume_fts_migration(db_path: Path) -> MigrationResult:
    """Resume exactly one journaled action at each crash-durable phase."""
    path = Path(db_path).expanduser().resolve(strict=False)
    journal = load_maintenance_journal(path)
    if journal is None:
        raise RuntimeError("no maintenance journal to resume")
    if Path(journal.db_path).resolve(strict=False) != path:
        raise RuntimeError("journal database path mismatch")
    while True:
        phase = journal.phase
        if phase in TERMINAL_PHASES:
            return MigrationResult(phase=phase.value, completed=phase is JournalPhase.COMPLETE)
        if phase is JournalPhase.PLANNED:
            _validate_live_source(path, journal)
            _require_no_live_users(path)
            journal = _transition(path, journal, JournalPhase.WRITERS_STOPPED)
        elif phase is JournalPhase.WRITERS_STOPPED:
            _validate_live_source(path, journal)
            checkpoint = _checkpoint_source(path, journal, _permit(path, journal))
            if checkpoint != (0, 0, 0):
                raise RuntimeError("checkpoint did not return exact (0,0,0)")
            backup, work, candidate = _migration_paths(path)
            journal = _transition(
                path,
                journal,
                JournalPhase.CHECKPOINTED,
                backup_path=backup,
                work_path=work,
                candidate_path=candidate,
                fingerprints=_source_inventory_fingerprints(path),
            )
        elif phase is JournalPhase.CHECKPOINTED:
            _validate_live_source(path, journal)
            if journal.backup_path is None or journal.work_path is None or journal.candidate_path is None:
                raise RuntimeError("checkpointed paths missing from journal")
            backup = Path(journal.backup_path)
            work = Path(journal.work_path)
            candidate = Path(journal.candidate_path)
            if work.is_symlink():
                raise RuntimeError("migration work path is unsafe")
            if work.exists():
                if not work.is_dir() or any(work.iterdir()):
                    raise FileExistsError("migration work path contains unknown files")
            else:
                work.mkdir(mode=0o700)
                os.chmod(work, 0o700)
                _fsync_directory(work.parent)
            expected_backup = journal.fingerprints.get("backup")
            pending_backup = backup.with_name(f".{backup.name}.pending")
            if expected_backup is None:
                if backup.exists() or backup.is_symlink() or pending_backup.exists() or pending_backup.is_symlink():
                    raise FileExistsError("unrecorded backup path already exists")
                _create_sqlite_backup(path, pending_backup, _permit(path, journal))
                journal = _transition(
                    path,
                    journal,
                    JournalPhase.CHECKPOINTED,
                    fingerprints={
                        **_source_inventory_fingerprints(path),
                        "backup": fingerprint_path(pending_backup),
                    },
                )
                expected_backup = journal.fingerprints.get("backup")
            backup_state = _exact_two_location_state(
                pending_backup,
                backup,
                expected_backup,
                "rollback backup install",
            )
            if backup_state == "source":
                os.replace(pending_backup, backup)
                _fsync_directory(backup.parent)
            journal = _transition(
                path,
                journal,
                JournalPhase.BACKUP_READY,
                backup_path=backup,
                work_path=work,
                candidate_path=candidate,
                fingerprints={
                    **_source_inventory_fingerprints(path),
                    "backup": fingerprint_path(backup),
                },
            )
        elif phase is JournalPhase.BACKUP_READY:
            _validate_live_source(path, journal)
            if journal.backup_path is None or journal.work_path is None or journal.candidate_path is None:
                raise RuntimeError("backup-ready paths missing from journal")
            _require_fingerprint(Path(journal.backup_path), journal.fingerprints.get("backup"), "backup")
            report = build_v2_candidate(path, Path(journal.work_path), journal, _permit(path, journal))
            candidate = Path(journal.candidate_path)
            if report.candidate_sha256 != _sha256_file(candidate):
                raise RuntimeError("candidate builder hash mismatch")
            controlled = verify_v2_candidate_with_controlled_corpus(
                Path(journal.backup_path), candidate, Path(journal.work_path)
            )
            if (
                controlled.paired_corpus_version != CONTROLLED_PAIRED_CORPUS_VERSION
                or not controlled.verification.verification_passed
                or not controlled.verification.candidate_accepted
            ):
                raise RuntimeError("controlled paired candidate verification failed")
            # Controlled verification opens the accepted WAL-mode candidate
            # read-only. Close those handles, then restore the required exact
            # empty-checkpoint/no-sidecar candidate bundle before recording it.
            _checkpoint_candidate_file(candidate)
            candidate_inventory = _candidate_inventory_fingerprints(candidate)
            if candidate_inventory["candidate_db"] is None:
                raise RuntimeError("verified candidate main file is missing")
            if (
                candidate_inventory["candidate_wal"] is not None
                or candidate_inventory["candidate_shm"] is not None
            ):
                raise RuntimeError("verified candidate has an unexpected sidecar")
            work_db = Path(journal.work_path) / "candidate-build" / "work.db"
            journal = _transition(
                path,
                journal,
                JournalPhase.CANDIDATE_READY,
                fingerprints={
                    **_source_inventory_fingerprints(path),
                    "candidate": candidate_inventory["candidate_db"],
                    **candidate_inventory,
                    "work": fingerprint_path(work_db),
                },
            )
        elif phase is JournalPhase.CANDIDATE_READY:
            _validate_live_source(path, journal)
            if journal.candidate_path is None:
                raise RuntimeError("candidate path missing from journal")
            _validate_candidate_bundle(Path(journal.candidate_path), journal)
            journal = _transition(path, journal, JournalPhase.SWAPPING)
        elif phase is JournalPhase.SWAPPING:
            journal = _move_recorded_bundle_to_original(path, journal)
        elif phase is JournalPhase.OLD_MOVED:
            journal = _install_recorded_candidate(path, journal)
        elif phase is JournalPhase.CANDIDATE_LIVE:
            _run_v2_canary(path, journal)
            journal = _transition(
                path,
                journal,
                JournalPhase.CANARY_PASSED,
                fingerprints=_source_inventory_fingerprints(path),
            )
        elif phase is JournalPhase.CANARY_PASSED:
            _require_fingerprint(path, journal.fingerprints.get("db"), "live candidate")
            _fsync_file(path)
            _fsync_directory(path.parent)
            journal = _transition(path, journal, JournalPhase.COMPLETE)
        else:
            raise RuntimeError("unsupported migration phase")


def _remove_owned_work(journal: MaintenanceJournal) -> None:
    if journal.work_path is None:
        return
    root = Path(journal.work_path)
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("owned work path is unsafe")
    allowed = {
        root / "candidate-build" / "candidate.db": journal.fingerprints.get("candidate"),
        root / "candidate-build" / "work.db": journal.fingerprints.get("work"),
    }
    files = [path for path in root.rglob("*") if path.is_file() or path.is_symlink()]
    if any(path not in allowed or path.is_symlink() for path in files):
        raise RuntimeError("owned work directory contains unknown files")
    for path in files:
        expected = allowed[path]
        if expected is None:
            raise RuntimeError("owned work file lacks a recorded fingerprint")
        _require_fingerprint(path, expected, "owned work file")
        path.unlink()
    build = root / "candidate-build"
    if build.exists():
        build.rmdir()
    root.rmdir()
    _fsync_directory(root.parent)


def abort_fts_migration(db_path: Path) -> MigrationResult:
    """Abort only while the exact recorded v1 bundle is still live."""
    path = Path(db_path).expanduser().resolve(strict=False)
    journal = load_maintenance_journal(path)
    if journal is None:
        raise RuntimeError("no maintenance journal to abort")
    safe = {
        JournalPhase.PLANNED,
        JournalPhase.WRITERS_STOPPED,
        JournalPhase.CHECKPOINTED,
        JournalPhase.BACKUP_READY,
        JournalPhase.CANDIDATE_READY,
    }
    if journal.phase not in safe:
        if journal.phase is JournalPhase.ABORTED:
            return MigrationResult(phase="aborted", completed=False)
        raise RuntimeError("post-swap migration requires rollback")
    _validate_live_source(path, journal)
    _remove_owned_work(journal)
    journal = _transition(path, journal, JournalPhase.ABORTED)
    return MigrationResult(phase=journal.phase.value, completed=False)


def _restore_original_bundle(path: Path, journal: MaintenanceJournal) -> None:
    live = _bundle_paths(path)
    original = _original_paths(path)
    actions: list[tuple[Path, Path]] = []
    for name in ("db", "wal", "shm"):
        original_key = f"original_{name}"
        expected = (
            journal.fingerprints[original_key]
            if original_key in journal.fingerprints
            else journal.fingerprints.get(name)
        )
        state = _exact_two_location_state(
            original[name], live[name], expected, f"original {name} restore"
        )
        if state == "source":
            actions.append((original[name], live[name]))
    for source, destination in actions:
        os.replace(source, destination)
        _fsync_directory(path.parent)


def _run_v1_rollback_canary(path: Path, journal: MaintenanceJournal) -> None:
    """Prove restored v1 write/read/search while leaving no durable canary row."""
    permit = _permit(path, journal)
    assert_state_db_maintenance_access(path, write_capable=True, permit=permit)
    conn = sqlite3.connect(path)
    token = f"hermesrollbackcanary{secrets.token_hex(12)}"
    try:
        if detect_fts_schema(conn) != "v1_inline":
            raise RuntimeError("restored database is not v1_inline")
        _require_expected_base_schema(conn)
        row = conn.execute("SELECT id FROM messages ORDER BY id LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("rollback canary requires a message row")
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "UPDATE messages SET content=coalesce(content,'') || ? WHERE id=?",
                (f" {token}", row[0]),
            )
            read = conn.execute(
                "SELECT content FROM messages WHERE id=?", (row[0],)
            ).fetchone()
            if read is None or token not in str(read[0]):
                raise RuntimeError("restored v1 canary read failed")
            found = conn.execute(
                "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
                (f'"{token}"',),
            ).fetchone()
            if found is None:
                raise RuntimeError("restored v1 canary search failed")
        finally:
            conn.rollback()
    finally:
        conn.close()

    checkpoint = _checkpoint_source(path, journal, _permit(path, journal))
    if checkpoint != (0, 0, 0):
        raise RuntimeError("restored v1 checkpoint did not return exact (0,0,0)")
    verify = sqlite3.connect(path)
    try:
        if detect_fts_schema(verify) != "v1_inline":
            raise RuntimeError("restored database lost v1_inline ownership")
        _require_expected_base_schema(verify)
        if verify.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise RuntimeError("restored v1 quick_check failed")
        # Both v1 indexes are FTS5 owners and support the rank-1 integrity command.
        integrity_check_fts_v2(verify)
    finally:
        verify.close()


def _rollback_liveness_paths(path: Path) -> tuple[Path, ...]:
    return tuple(
        item
        for bundle in (
            _bundle_paths(path),
            _original_paths(path),
            _quarantine_paths(path),
        )
        for item in bundle.values()
    )


def rollback_fts_migration(
    db_path: Path, backup: Path | None = None
) -> MigrationResult:
    """Quarantine a recorded live v2 bundle and restore the exact v1 bundle."""
    path = Path(db_path).expanduser().resolve(strict=False)
    journal = load_maintenance_journal(path)
    if journal is None:
        raise RuntimeError("no maintenance journal to rollback")
    if journal.phase is JournalPhase.ROLLED_BACK:
        return MigrationResult(phase="rolled_back", completed=False)
    if journal.phase in {
        JournalPhase.PLANNED,
        JournalPhase.WRITERS_STOPPED,
        JournalPhase.CHECKPOINTED,
        JournalPhase.BACKUP_READY,
        JournalPhase.CANDIDATE_READY,
    }:
        raise RuntimeError("pre-swap migration must use abort")
    if backup is not None:
        if journal.backup_path is None or Path(backup).resolve(strict=False) != Path(journal.backup_path).resolve(strict=False):
            raise RuntimeError("rollback backup does not match journal")
        _require_fingerprint(Path(backup), journal.fingerprints.get("backup"), "rollback backup")

    activation = journal.fingerprints.get("rollback_activation")
    if activation is None:
        # Proof one occurs while a terminal journal may still allow ordinary writers.
        _require_no_live_users_for_paths(_rollback_liveness_paths(path))
        next_phase = (
            JournalPhase.CANDIDATE_LIVE
            if journal.phase in {JournalPhase.CANARY_PASSED, JournalPhase.COMPLETE}
            else journal.phase
        )
        journal = _transition(
            path,
            journal,
            next_phase,
            fingerprints={"rollback_activation": _status_fingerprint("planned")},
        )
    elif not _same_fingerprint(
        dict(activation) if isinstance(activation, Mapping) else None,
        _status_fingerprint("planned"),
    ):
        raise RuntimeError("unknown rollback activation marker")

    # Proof two always runs after the durable nonterminal activation. A crash or
    # failed proof leaves this marker in place, so every retry repeats this proof.
    _require_no_live_users_for_paths(_rollback_liveness_paths(path))

    if journal.phase is JournalPhase.CANDIDATE_LIVE and "rollback_db" not in journal.fingerprints:
        # Ordinary post-complete use may legitimately change size, mtime, hash,
        # and WAL contents. After both liveness proofs, accept only the same live
        # main device+inode lineage, then checkpoint and fully verify v2 before
        # freezing a fresh exact quarantine inventory.
        journal = _verify_active_v2_for_rollback(path, journal)

    if journal.phase is JournalPhase.SWAPPING:
        journal = _move_recorded_bundle_to_original(path, journal)
    if journal.phase is JournalPhase.OLD_MOVED:
        _restore_original_bundle(path, journal)
    else:
        live = _bundle_paths(path)
        quarantine = _quarantine_paths(path)
        actions: list[tuple[Path, Path]] = []
        for name in ("db", "wal", "shm"):
            key = f"rollback_{name}"
            if key not in journal.fingerprints:
                raise RuntimeError(f"candidate {name} rollback inventory missing")
            state = _exact_two_location_state(
                live[name],
                quarantine[name],
                journal.fingerprints[key],
                f"candidate {name} quarantine",
            )
            if state == "source":
                actions.append((live[name], quarantine[name]))
        for source, destination in actions:
            os.replace(source, destination)
            _fsync_directory(path.parent)
        _restore_original_bundle(path, journal)
    _fsync_file(path)
    _fsync_directory(path.parent)
    _run_v1_rollback_canary(path, journal)
    journal = _transition(
        path,
        journal,
        JournalPhase.ROLLED_BACK,
        fingerprints={
            **_source_inventory_fingerprints(path),
            "rollback_v1_canary": _status_fingerprint("passed"),
        },
    )
    return MigrationResult(phase=journal.phase.value, completed=False)


def _open_read_only(db_path: Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve(strict=False)
    # A closed WAL database can have no sidecars. Plain mode=ro may recreate
    # empty WAL/SHM files merely to register a reader, violating the estimator's
    # no-side-effect contract. With no sidecars there are no uncheckpointed WAL
    # frames to observe, so immutable=1 is both accurate and artifact-free.
    # When either sidecar exists, keep normal mode=ro so committed WAL frames
    # remain visible; query_only still forbids SQL writes.
    sidecars_exist = Path(f"{path}-wal").exists() or Path(f"{path}-shm").exists()
    query = "mode=ro" if sidecars_exist else "mode=ro&immutable=1"
    conn = sqlite3.connect(f"{path.as_uri()}?{query}", uri=True)
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
