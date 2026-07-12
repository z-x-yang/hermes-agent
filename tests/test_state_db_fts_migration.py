from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

import state_db_fts_migration as migration
from state_db_fts import create_fts_v1, rebuild_fts
from state_db_maintenance import (
    JournalPhase,
    MaintenanceBlockedError,
    MaintenanceJournal,
    issue_maintenance_permit,
    state_db_file_inventory,
    write_maintenance_journal,
)
from state_db_fts_migration import (
    SearchCase,
    build_v2_candidate,
    estimate_payload_retention,
    field_digest,
    plan_fts_migration,
    status_fts_migration,
    verify_v2_candidate,
)


def _make_v1_fixture(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            model TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            archived INTEGER NOT NULL DEFAULT 0,
            parent_session_id TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            reasoning TEXT,
            reasoning_content TEXT,
            reasoning_details TEXT,
            codex_reasoning_items TEXT,
            codex_message_items TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            compacted INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE durable_records (
            record_key TEXT PRIMARY KEY,
            payload BLOB,
            ordinal INTEGER NOT NULL
        );
        INSERT INTO durable_records(record_key,payload,ordinal)
        VALUES('record-a',x'0001ff',1);
        """
    )
    conn.executemany(
        "INSERT INTO sessions(id, source, model, started_at, ended_at, archived, parent_session_id) VALUES(?,?,?,?,?,?,?)",
        [
            ("active-session", "cli", "model-a", 100.0, None, 0, None),
            ("held-archive", "cron", "model-b", 50.0, 60.0, 1, "active-session"),
        ],
    )
    conn.executemany(
        """INSERT INTO messages(
               id, session_id, role, content, tool_call_id, tool_calls, tool_name,
               timestamp, active, compacted
           ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        [
            (1, "active-session", "tool", "persisted body", "call-1", None, "terminal", 100.0, 0, 1),
            (2, "active-session", "assistant", "hermes://session/active-session/message/1", None, None, None, 101.0, 1, 0),
            (3, "active-session", "assistant", "hermes://session/active-session/message/999", None, None, None, 102.0, 1, 0),
            (4, "active-session", "assistant", "hermes://session//message/nope", None, None, None, 103.0, 1, 0),
            (5, "held-archive", "tool", "archived payload", "call-2", None, "terminal", 50.0, 0, 1),
            (6, "active-session", "assistant", "hermes://session/active-session/message/2", None, None, None, 104.0, 1, 0),
        ],
    )
    create_fts_v1(conn)
    rebuild_fts(conn, "v1_inline")
    conn.commit()
    conn.close()


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _db_digest(path: Path) -> dict[str, object]:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        schema = conn.execute(
            "SELECT type,name,tbl_name,coalesce(sql,'') FROM sqlite_master ORDER BY type,name"
        ).fetchall()
        meta = conn.execute("SELECT key,value FROM state_meta ORDER BY key").fetchall()
        data = {
            "sessions": conn.execute("SELECT * FROM sessions ORDER BY id").fetchall(),
            "messages": conn.execute("SELECT * FROM messages ORDER BY id").fetchall(),
        }
    finally:
        conn.close()
    return {
        "main_bytes": path.read_bytes(),
        "main_sha256": _sha256(path),
        "schema": schema,
        "meta": meta,
        "data": data,
    }


def _fts_family_bytes(path: Path) -> dict[str, int]:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        unicode_bytes = conn.execute(
            "SELECT coalesce(sum(pgsize),0) FROM dbstat WHERE name IN "
            "('messages_fts','messages_fts_data','messages_fts_idx',"
            "'messages_fts_content','messages_fts_docsize','messages_fts_config')"
        ).fetchone()[0]
        trigram_bytes = conn.execute(
            "SELECT coalesce(sum(pgsize),0) FROM dbstat WHERE name IN "
            "('messages_fts_trigram','messages_fts_trigram_data',"
            "'messages_fts_trigram_idx','messages_fts_trigram_content',"
            "'messages_fts_trigram_docsize','messages_fts_trigram_config')"
        ).fetchone()[0]
        return {"unicode": unicode_bytes, "trigram": trigram_bytes}
    finally:
        conn.close()


def test_plan_status_and_estimator_are_file_schema_meta_data_immutable(tmp_path):
    path = tmp_path / "state.db"
    _make_v1_fixture(path)
    journal = tmp_path / "state-db-maintenance.json"
    journal.write_text("{malformed", encoding="utf-8")
    before_digest = _db_digest(path)
    expected_fts_bytes = _fts_family_bytes(path)
    before_artifacts = sorted(item.name for item in tmp_path.iterdir())
    before_journal = journal.read_bytes()

    plan = plan_fts_migration(path)
    estimate = estimate_payload_retention(path)
    status = status_fts_migration(path)

    assert plan.schema_kind == "v1_inline"
    assert plan.required_free_bytes == 2 * (
        plan.db_bytes + plan.wal_bytes + plan.shm_bytes
    ) + 10 * 1024**3
    assert plan.archived_session_holds == 1
    assert plan.session_deletion_candidates == 0
    assert plan.maintenance_status == "malformed"
    assert plan.writer_status == "not_probed_read_only"

    assert dict(plan.fts_object_bytes) == expected_fts_bytes

    assert estimate.clock_status == "unavailable"
    assert estimate.rows_by_age_basis == "non_actionable_upper_bound"
    assert estimate.valid_handle_targets == 1
    assert estimate.handle_exemptions == 1
    assert estimate.malformed_handles == 1
    assert estimate.missing_handles == 1
    assert estimate.wrong_handle_targets == 1
    assert estimate.missing_or_wrong_targets == 2
    assert estimate.archived_session_holds == 1
    assert estimate.session_deletion_candidates == 0
    serialized = json.dumps(asdict(estimate), sort_keys=True)
    representation = repr(estimate)
    for secret in ("hermes://", "active-session", "held-archive", "call-1", "persisted body"):
        assert secret not in serialized
        assert secret not in representation

    assert status["schema_kind"] == "v1_inline"
    assert status["journal_status"] == "malformed"
    assert status["journal_phase"] is None
    assert status["read_only"] is True
    assert _db_digest(path) == before_digest
    assert journal.read_bytes() == before_journal
    assert sorted(item.name for item in tmp_path.iterdir()) == before_artifacts
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()


def test_status_exposes_phase_and_only_safe_aggregate_fingerprints(tmp_path):
    path = tmp_path / "state.db"
    _make_v1_fixture(path)
    journal = MaintenanceJournal.new("private-operation", path)
    object.__setattr__(
        journal,
        "fingerprints",
        {
            "db": {
                "path": "/private/session-id/state.db",
                "size": 123,
                "sha256": "abc",
            },
            "private-session-id": {
                "path": "/private/message/99",
                "size": 456,
                "sha256": "def",
            },
        },
    )
    write_maintenance_journal(path, journal)

    result = status_fts_migration(path)

    assert result["journal_status"] == "active"
    assert result["journal_phase"] == "planned"
    assert result["journal_fingerprints"] == {
        "db": {"size": 123, "sha256": "abc"}
    }
    serialized = json.dumps(result, sort_keys=True)
    assert "private-operation" not in serialized
    assert "private-session-id" not in serialized
    assert "/private/message/99" not in serialized


def test_invalid_schema_marker_is_reported_without_raw_value(tmp_path):
    path = tmp_path / "state.db"
    _make_v1_fixture(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO state_meta(key,value) VALUES('fts_schema_version',?)",
            ("private-session-id",),
        )
        conn.commit()
    finally:
        conn.close()

    plan = plan_fts_migration(path)
    status = status_fts_migration(path)

    assert plan.schema_marker == "invalid"
    assert status["schema_marker"] == "invalid"
    assert "private-session-id" not in repr(plan)
    assert "private-session-id" not in json.dumps(status, sort_keys=True)


def _candidate_fixture(path: Path) -> None:
    _make_v1_fixture(path)
    conn = sqlite3.connect(path)
    try:
        conn.executemany(
            """INSERT INTO messages(
                   id, session_id, role, content, tool_call_id, tool_calls, tool_name,
                   timestamp, active, compacted
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            [
                (7, "active-session", "user", "English alpha café", None, None, None, 105.0, 1, 0),
                (8, "active-session", "assistant", "默认中文检索", None, None, None, 106.0, 1, 0),
                (9, "active-session", "assistant", "", None, '[{"name":"工具调用中文"}]', None, 107.0, 1, 0),
                (10, "active-session", "tool", "显式工具中文", "call-3", None, "terminal", 108.0, 1, 0),
                (11, "active-session", "user", "短词中文", None, None, None, 109.0, 0, 1),
                (12, "held-archive", "assistant", "lineage duplicate English alpha", None, None, None, 61.0, 0, 0),
            ],
        )
        rebuild_fts(conn, "v1_inline")
        conn.commit()
    finally:
        conn.close()


def _candidate_access(path: Path):
    journal = replace(
        MaintenanceJournal.new("candidate-operation", path),
        phase=JournalPhase.BACKUP_READY,
    )
    write_maintenance_journal(path, journal)
    permit = issue_maintenance_permit(
        path,
        journal.operation_id,
        frozenset({JournalPhase.BACKUP_READY}),
    )
    return journal, permit


def test_candidate_copy_preserves_source_and_builds_verified_compact_v2(tmp_path):
    source = tmp_path / "source" / "state.db"
    source.parent.mkdir()
    _candidate_fixture(source)
    journal, permit = _candidate_access(source)
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)
    before_db = _db_digest(source)
    before_files = state_db_file_inventory(source)
    before_names = sorted(item.name for item in source.parent.iterdir())

    report = build_v2_candidate(source, work_dir, journal, permit)

    assert report.source_message_count == report.candidate_message_count == 12
    assert report.source_session_count == report.candidate_session_count == 2
    assert report.field_digest_equal
    assert report.unicode_integrity == "passed_rank1"
    assert report.trigram_integrity == "passed_rank1"
    assert report.trigger_rollback_probe == "passed"
    assert report.quick_check == "ok"
    assert report.no_inline_content_shadows
    assert not report.candidate_wal_exists
    assert not report.candidate_shm_exists
    assert len(report.candidate_sha256) == 64
    assert report.paired_verification_required
    assert not report.eligible_for_live_swap
    assert _db_digest(source) == before_db
    assert state_db_file_inventory(source) == before_files
    assert sorted(item.name for item in source.parent.iterdir()) == before_names

    build_dir = work_dir / "candidate-build"
    candidate = build_dir / "candidate.db"
    assert report.candidate_sha256 == _sha256(candidate)
    assert (os.stat(build_dir).st_mode & 0o777) == 0o700
    assert (os.stat(candidate).st_mode & 0o777) == 0o600
    assert not Path(f"{candidate}-wal").exists()
    assert not Path(f"{candidate}-shm").exists()
    source_conn = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    candidate_conn = sqlite3.connect(f"{candidate.resolve().as_uri()}?mode=ro", uri=True)
    try:
        assert field_digest(source_conn) == field_digest(candidate_conn)
        assert candidate_conn.execute(
            "SELECT value FROM state_meta WHERE key='fts_schema_version'"
        ).fetchone() == ("2",)
        assert candidate_conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE name IN "
            "('messages_fts_content','messages_fts_trigram_content')"
        ).fetchone() == (0,)
        assert candidate_conn.execute(
            "SELECT record_key,payload,ordinal FROM durable_records"
        ).fetchone() == ("record-a", b"\x00\x01\xff", 1)
    finally:
        source_conn.close()
        candidate_conn.close()


def test_candidate_copy_requires_bound_permit_and_rejects_unsafe_paths(tmp_path):
    source = tmp_path / "source" / "state.db"
    source.parent.mkdir()
    _candidate_fixture(source)
    journal, permit = _candidate_access(source)
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)
    before_db = _db_digest(source)
    before_files = state_db_file_inventory(source)

    try:
        build_v2_candidate(source, work_dir, journal, None)
    except MaintenanceBlockedError:
        pass
    else:
        raise AssertionError("ordinary caller bypassed the maintenance permit")

    other = tmp_path / "other.db"
    _candidate_fixture(other)
    other_journal, other_permit = _candidate_access(other)
    try:
        build_v2_candidate(source, work_dir, other_journal, other_permit)
    except MaintenanceBlockedError:
        pass
    else:
        raise AssertionError("permit for another journal/database was accepted")

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_dir = tmp_path / "symlink-work"
    symlink_dir.symlink_to(outside, target_is_directory=True)
    try:
        build_v2_candidate(source, symlink_dir, journal, permit)
    except ValueError:
        pass
    else:
        raise AssertionError("symlink work_dir was accepted")

    collision = work_dir / "candidate-build"
    collision.mkdir()
    sentinel = collision / "unknown"
    sentinel.write_text("keep", encoding="utf-8")
    try:
        build_v2_candidate(source, work_dir, journal, permit)
    except FileExistsError:
        pass
    else:
        raise AssertionError("pre-existing candidate path was overwritten")
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert _db_digest(source) == before_db
    assert state_db_file_inventory(source) == before_files


def _paired_corpus():
    return (
        SearchCase("c01", "english", "English alpha", role_filter=("user", "assistant")),
        SearchCase("c02", "english", "café", role_filter=("user", "assistant")),
        SearchCase("c03", "default_cjk", "默认中文", role_filter=("user", "assistant")),
        SearchCase("c04", "default_cjk", "工具调用中文", role_filter=("user", "assistant")),
        SearchCase("c05", "tool_cjk", "显式工具中文", role_filter=("tool",)),
        SearchCase("c06", "default_cjk", "短词", role_filter=("user", "assistant")),
        SearchCase(
            "c07",
            "english",
            "English alpha",
            role_filter=("user", "assistant"),
            source_filter=("cli",),
        ),
        SearchCase(
            "c08",
            "english",
            "English alpha",
            role_filter=("user", "assistant"),
            exclude_sources=("cron",),
        ),
        SearchCase(
            "c09",
            "english",
            "English alpha",
            role_filter=("user", "assistant"),
            include_inactive=True,
        ),
    )


def test_paired_search_accepts_complete_parity_and_emits_aggregate_only_report(tmp_path):
    source = tmp_path / "source" / "state.db"
    source.parent.mkdir()
    _candidate_fixture(source)
    journal, permit = _candidate_access(source)
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)
    build_v2_candidate(source, work_dir, journal, permit)
    candidate = work_dir / "candidate-build" / "candidate.db"

    report = verify_v2_candidate(source, candidate, _paired_corpus())

    assert report.verification_passed
    assert report.candidate_accepted
    assert report.files_stable
    assert report.source_copy_sha256 == _sha256(source)
    assert report.candidate_sha256 == _sha256(candidate)
    assert report.field_digest_equal
    assert report.row_counts_equal
    assert report.all_match_sets_equal
    assert report.all_lineage_dedupe_equal
    assert report.minimum_top10_overlap >= 0.9
    assert report.all_snippets_valid
    assert report.all_ordering_differences_allowed
    assert not report.eligible_for_live_swap
    assert {case.case_id for case in report.cases} == {
        "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09"
    }
    assert {item.category for item in report.latency} == {
        "english", "default_cjk", "tool_cjk"
    }
    assert all(case.match_sets_equal for case in report.cases)
    lineage_case = next(case for case in report.cases if case.case_id == "c09")
    assert lineage_case.source_match_count == 2
    assert lineage_case.candidate_match_count == 2
    assert lineage_case.source_lineage_dedup_count == 1
    assert lineage_case.candidate_lineage_dedup_count == 1
    assert lineage_case.lineage_dedupe_equal
    assert all(case.snippets_valid for case in report.cases)
    serialized = json.dumps(asdict(report), sort_keys=True)
    representation = repr(report)
    for secret in (
        "English alpha",
        "默认中文",
        "工具调用中文",
        "显式工具中文",
        "短词",
        "active-session",
        "held-archive",
        "candidate-operation",
        "hermes://",
        '"name":"工具调用中文"',
        ">>>",
        "<<<",
    ):
        assert secret not in serialized
        assert secret not in representation


def test_paired_search_rejects_different_lineage_survivor_identity_without_leaking(
    tmp_path, monkeypatch
):
    source = tmp_path / "source" / "state.db"
    source.parent.mkdir()
    _candidate_fixture(source)
    journal, permit = _candidate_access(source)
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)
    build_v2_candidate(source, work_dir, journal, permit)
    candidate = work_dir / "candidate-build" / "candidate.db"
    corpus = _paired_corpus()
    canonical = migration._lineage_survivor_identities
    calls = 0

    def changed_candidate_survivor(db, raw_results):
        nonlocal calls
        calls += 1
        survivors = canonical(db, raw_results)
        if calls <= len(corpus):
            return survivors
        return tuple(
            (root, session_id, message_id + 1000)
            for root, session_id, message_id in survivors
        )

    monkeypatch.setattr(
        migration, "_lineage_survivor_identities", changed_candidate_survivor
    )

    report = verify_v2_candidate(source, candidate, corpus)

    assert report.all_match_sets_equal
    assert report.minimum_top10_overlap >= 0.9
    assert report.all_ordering_differences_allowed
    assert not report.all_lineage_dedupe_equal
    assert not report.verification_passed
    assert not report.candidate_accepted
    assert any(
        case.source_lineage_dedup_count == case.candidate_lineage_dedup_count
        and not case.lineage_dedupe_equal
        for case in report.cases
    )
    serialized = json.dumps(asdict(report), sort_keys=True)
    representation = repr(report)
    for secret in (
        "English alpha",
        "active-session",
        "held-archive",
        "candidate-operation",
        ">>>",
        "<<<",
    ):
        assert secret not in serialized
        assert secret not in representation


def test_paired_search_rejects_marker_two_candidate_with_changed_base_fields(tmp_path):
    source = tmp_path / "source" / "state.db"
    source.parent.mkdir()
    _candidate_fixture(source)
    journal, permit = _candidate_access(source)
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)
    build_v2_candidate(source, work_dir, journal, permit)
    candidate = work_dir / "candidate-build" / "candidate.db"
    conn = sqlite3.connect(candidate)
    try:
        conn.execute("UPDATE messages SET content=? WHERE id=7", ("private mutated payload",))
        conn.commit()
    finally:
        conn.close()

    report = verify_v2_candidate(source, candidate, _paired_corpus())

    assert not report.verification_passed
    assert not report.candidate_accepted
    assert not report.field_digest_equal
    assert not report.eligible_for_live_swap
    assert "private mutated payload" not in repr(report)
    assert "private mutated payload" not in json.dumps(asdict(report), sort_keys=True)
