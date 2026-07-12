from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import state_db_fts_migration as migration
import pytest
from state_db_fts import create_fts_v1, rebuild_fts
from state_db_maintenance import (
    JournalPhase,
    MaintenanceBlockedError,
    MaintenanceJournal,
    assert_state_db_maintenance_access,
    issue_maintenance_permit,
    state_db_file_inventory,
    write_maintenance_journal,
)
from state_db_fts_migration import (
    CONTROLLED_PAIRED_CORPUS_VERSION,
    SearchCase,
    abort_fts_migration,
    apply_fts_migration,
    build_v2_candidate,
    controlled_paired_corpus,
    estimate_payload_retention,
    field_digest,
    find_live_state_db_users,
    plan_fts_migration,
    resume_fts_migration,
    rollback_fts_migration,
    status_fts_migration,
    verify_v2_candidate,
    verify_v2_candidate_with_controlled_corpus,
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


def _lsof_result(returncode: int = 0, stdout: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def test_liveness_report_parses_machine_output_and_fails_closed(monkeypatch, tmp_path):
    path = tmp_path / "state.db"
    _make_v1_fixture(path)
    monkeypatch.setattr(
        migration, "_run_lsof", lambda paths: _lsof_result(stdout="p123\nf9\n")
    )

    report = find_live_state_db_users(path)

    assert report.status == "live"
    assert report.holder_count == 1
    assert "123" not in repr(report)

    monkeypatch.setattr(migration, "_run_lsof", lambda paths: _lsof_result(1, ""))
    assert find_live_state_db_users(path).status == "clear"
    monkeypatch.setattr(migration, "_run_lsof", lambda paths: _lsof_result(2, ""))
    try:
        find_live_state_db_users(path)
    except RuntimeError as exc:
        assert "lsof" in str(exc)
    else:
        raise AssertionError("unknown lsof failure was accepted")


def test_apply_refuses_live_holder_before_planned(monkeypatch, tmp_path):
    path = tmp_path / "state.db"
    _make_v1_fixture(path)
    monkeypatch.setattr(
        migration, "_run_lsof", lambda paths: _lsof_result(stdout="p123\nf9\n")
    )

    try:
        apply_fts_migration(path)
    except RuntimeError as exc:
        assert "writers" in str(exc)
    else:
        raise AssertionError("apply accepted a live writer")

    assert not (tmp_path / "state-db-maintenance.json").exists()


def test_apply_rechecks_liveness_after_planned(monkeypatch, tmp_path):
    path = tmp_path / "state.db"
    _make_v1_fixture(path)
    calls = 0

    def lsof_after_planned(paths):
        nonlocal calls
        calls += 1
        return _lsof_result(1, "") if calls == 1 else _lsof_result(stdout="p456\nf9\n")

    monkeypatch.setattr(migration, "_run_lsof", lsof_after_planned)
    monkeypatch.setattr(
        migration.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=20 * 1024**3, used=0, free=20 * 1024**3),
    )

    try:
        apply_fts_migration(path)
    except RuntimeError as exc:
        assert "writers" in str(exc)
    else:
        raise AssertionError("second liveness proof was skipped")

    journal = migration.load_maintenance_journal(path)
    assert journal is not None
    assert journal.phase is JournalPhase.PLANNED


def test_apply_requires_exact_empty_checkpoint_before_backup(monkeypatch, tmp_path):
    path = tmp_path / "state.db"
    _make_v1_fixture(path)
    monkeypatch.setattr(migration, "_run_lsof", lambda paths: _lsof_result(1, ""))
    monkeypatch.setattr(
        migration.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=20 * 1024**3, used=0, free=20 * 1024**3),
    )
    monkeypatch.setattr(migration, "_checkpoint_source", lambda *args: (0, 1, 1))

    try:
        apply_fts_migration(path)
    except RuntimeError as exc:
        assert "checkpoint" in str(exc)
    else:
        raise AssertionError("non-empty checkpoint was accepted")

    journal = migration.load_maintenance_journal(path)
    assert journal is not None
    assert journal.phase is JournalPhase.WRITERS_STOPPED
    assert journal.backup_path is None


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
    assert plan.paired_corpus_version == CONTROLLED_PAIRED_CORPUS_VERSION

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


def _built_candidate(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source" / "state.db"
    source.parent.mkdir()
    _candidate_fixture(source)
    journal, permit = _candidate_access(source)
    work_dir = tmp_path / "work"
    work_dir.mkdir(mode=0o700)
    build_v2_candidate(source, work_dir, journal, permit)
    return source, work_dir / "candidate-build" / "candidate.db", work_dir


def test_controlled_provider_exercises_all_required_semantics_without_mutating_originals(
    tmp_path,
):
    source, candidate, work_dir = _built_candidate(tmp_path)
    source_before = _db_digest(source)
    candidate_before = _db_digest(candidate)
    source_inventory = state_db_file_inventory(source)
    candidate_inventory = state_db_file_inventory(candidate)

    corpus = controlled_paired_corpus()
    result = verify_v2_candidate_with_controlled_corpus(source, candidate, work_dir)

    assert corpus
    assert result.paired_corpus_version == CONTROLLED_PAIRED_CORPUS_VERSION
    assert result.verification.verification_passed
    assert result.verification.candidate_accepted
    assert all(case.source_match_count > 0 for case in result.verification.cases)
    assert all(case.candidate_match_count > 0 for case in result.verification.cases)
    assert {case.case_id for case in result.verification.cases} == {
        case.case_id for case in corpus
    }
    assert {case.category for case in result.verification.cases} == {
        "english",
        "default_cjk",
        "tool_cjk",
    }
    lineage = next(
        case for case in result.verification.cases if case.case_id == "lineage-dedupe"
    )
    assert lineage.source_match_count == lineage.candidate_match_count == 2
    assert lineage.source_lineage_dedup_count == lineage.candidate_lineage_dedup_count == 1
    default_visibility = next(
        case for case in result.verification.cases if case.case_id == "active-compacted"
    )
    all_visibility = next(
        case for case in result.verification.cases if case.case_id == "include-inactive"
    )
    assert default_visibility.source_match_count == 2
    assert all_visibility.source_match_count == 3
    assert _db_digest(source) == source_before
    assert _db_digest(candidate) == candidate_before
    assert state_db_file_inventory(source) == source_inventory
    assert state_db_file_inventory(candidate) == candidate_inventory
    assert not (work_dir / "controlled-paired-verification").exists()


def test_controlled_verifier_cleans_owned_copies_after_injected_failure(
    tmp_path, monkeypatch
):
    source, candidate, work_dir = _built_candidate(tmp_path)
    source_before = _db_digest(source)
    candidate_before = _db_digest(candidate)

    def injected_failure(*args, **kwargs):
        raise RuntimeError("injected controlled verification failure")

    monkeypatch.setattr(migration, "verify_v2_candidate", injected_failure)
    try:
        verify_v2_candidate_with_controlled_corpus(source, candidate, work_dir)
    except RuntimeError as exc:
        assert str(exc) == "injected controlled verification failure"
    else:
        raise AssertionError("injected verifier failure was swallowed")

    assert _db_digest(source) == source_before
    assert _db_digest(candidate) == candidate_before
    assert not (work_dir / "controlled-paired-verification").exists()


def test_controlled_verifier_report_is_private_and_rejects_candidate_search_divergence(
    tmp_path, monkeypatch
):
    source, candidate, work_dir = _built_candidate(tmp_path)
    canonical = migration._search_copy
    calls = 0

    def divergent_candidate_search(path, corpus):
        nonlocal calls
        calls += 1
        results = canonical(path, corpus)
        if calls == 2:
            first = results[0]
            results[0] = replace(first, matches=[])
        return results

    monkeypatch.setattr(migration, "_search_copy", divergent_candidate_search)
    result = verify_v2_candidate_with_controlled_corpus(source, candidate, work_dir)

    assert not result.verification.verification_passed
    assert not result.verification.candidate_accepted
    assert not result.verification.all_match_sets_equal
    serialized = json.dumps(asdict(result), sort_keys=True)
    representation = repr(result)
    for secret in (
        "controlled-paired-verification",
        "state.db",
        "candidate.db",
        "hermes-controlled-",
        "cpv1_",
        "受控检索",
        "tool_calls",
    ):
        assert secret not in serialized
        assert secret not in representation
    assert not (work_dir / "controlled-paired-verification").exists()


def test_controlled_verifier_fails_closed_on_preexisting_or_symlink_owned_path(tmp_path):
    source, candidate, work_dir = _built_candidate(tmp_path)
    owned = work_dir / "controlled-paired-verification"
    owned.mkdir()
    sentinel = owned / "keep"
    sentinel.write_text("do not delete", encoding="utf-8")
    try:
        verify_v2_candidate_with_controlled_corpus(source, candidate, work_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError("pre-existing controlled verification path was accepted")
    assert sentinel.read_text(encoding="utf-8") == "do not delete"

    sentinel.unlink()
    owned.rmdir()
    outside = tmp_path / "outside-controlled"
    outside.mkdir()
    owned.symlink_to(outside, target_is_directory=True)
    try:
        verify_v2_candidate_with_controlled_corpus(source, candidate, work_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError("symlink controlled verification path was accepted")
    assert owned.is_symlink()


def _allow_apply(monkeypatch):
    monkeypatch.setattr(migration, "_run_lsof", lambda paths: _lsof_result(1, ""))
    monkeypatch.setattr(migration, "_checkpoint_source", lambda *args: (0, 0, 0))
    monkeypatch.setattr(
        migration.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=20 * 1024**3, used=0, free=20 * 1024**3),
    )


def test_apply_runs_exact_phase_sequence_and_installs_verified_candidate(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    phases = []
    canonical_write = migration.write_maintenance_journal

    def recording_write(db_path, journal):
        canonical_write(db_path, journal)
        phases.append(journal.phase.value)

    monkeypatch.setattr(migration, "write_maintenance_journal", recording_write)

    result = apply_fts_migration(path)

    assert result.phase == "complete"
    assert result.completed
    phase_transitions = [
        phase for index, phase in enumerate(phases) if index == 0 or phase != phases[index - 1]
    ]
    assert phase_transitions == [
        "planned",
        "writers_stopped",
        "checkpointed",
        "backup_ready",
        "candidate_ready",
        "swapping",
        "old_moved",
        "candidate_live",
        "canary_passed",
        "complete",
    ]
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        assert migration.detect_fts_schema(conn) == "v2_external"
        assert conn.execute(
            "SELECT value FROM state_meta WHERE key='fts_schema_version'"
        ).fetchone() == ("2",)
    finally:
        conn.close()
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()
    assert (tmp_path / "state.db.pre-v2.original").exists()
    assert (tmp_path / "state.db.pre-v2.backup").exists()
    serialized = json.dumps(asdict(result), sort_keys=True)
    assert "state.db" not in serialized
    assert "candidate" not in serialized


def test_candidate_ready_abort_preserves_backup_and_unblocks_writers(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    before = _db_digest(path)
    canonical_transition = migration._transition
    stopped = False

    def stop_after_candidate(*args, **kwargs):
        nonlocal stopped
        journal = canonical_transition(*args, **kwargs)
        if journal.phase is JournalPhase.CANDIDATE_READY and not stopped:
            stopped = True
            raise RuntimeError("injected candidate-ready crash")
        return journal

    monkeypatch.setattr(migration, "_transition", stop_after_candidate)
    try:
        apply_fts_migration(path)
    except RuntimeError as exc:
        assert str(exc) == "injected candidate-ready crash"
    else:
        raise AssertionError("crash injection did not stop apply")

    result = abort_fts_migration(path)

    assert result.phase == "aborted"
    assert _db_digest(path) == before
    assert (tmp_path / "state.db.pre-v2.backup").exists()
    assert not (tmp_path / ".state.db.fts-v2-work").exists()
    assert abort_fts_migration(path) == result
    assert_state_db_maintenance_access(path, write_capable=True)


def test_complete_rollback_quarantines_candidate_and_restores_exact_v1(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    before = _db_digest(path)
    apply_fts_migration(path)

    result = rollback_fts_migration(path)

    assert result.phase == "rolled_back"
    assert _db_digest(path) == before
    assert (tmp_path / "state.db.v2.quarantine").exists()
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()
    assert rollback_fts_migration(path) == result
    assert_state_db_maintenance_access(path, write_capable=True)


def test_resume_unknown_planned_fingerprint_preserves_every_file(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    journal = replace(
        MaintenanceJournal.new("unknown-fingerprint-operation", path),
        fingerprints=state_db_file_inventory(path),
    )
    write_maintenance_journal(path, journal)
    with path.open("ab") as stream:
        stream.write(b"unexpected")
    file_before = path.read_bytes()
    journal_path = tmp_path / "state-db-maintenance.json"
    journal_before = journal_path.read_bytes()

    try:
        resume_fts_migration(path)
    except RuntimeError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown fingerprint was accepted")

    assert path.read_bytes() == file_before
    assert journal_path.read_bytes() == journal_before


@pytest.mark.parametrize(
    "crash_phase",
    [
        JournalPhase.WRITERS_STOPPED,
        JournalPhase.CHECKPOINTED,
        JournalPhase.BACKUP_READY,
        JournalPhase.CANDIDATE_READY,
        JournalPhase.SWAPPING,
        JournalPhase.OLD_MOVED,
        JournalPhase.CANDIDATE_LIVE,
        JournalPhase.CANARY_PASSED,
    ],
)
def test_resume_after_each_recorded_phase_boundary(tmp_path, monkeypatch, crash_phase):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    canonical_transition = migration._transition
    injected = False

    def crash_after_transition(*args, **kwargs):
        nonlocal injected
        journal = canonical_transition(*args, **kwargs)
        if journal.phase is crash_phase and not injected:
            injected = True
            raise RuntimeError("injected phase crash")
        return journal

    monkeypatch.setattr(migration, "_transition", crash_after_transition)
    with pytest.raises(RuntimeError, match="injected phase crash"):
        apply_fts_migration(path)

    result = resume_fts_migration(path)

    assert result.phase == "complete"
    assert result.completed


def test_resume_completes_source_bundle_after_crash_between_renames(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    canonical_replace = migration.os.replace
    crashed = False
    original_main = tmp_path / "state.db.pre-v2.original"

    def crash_after_main(source, destination):
        nonlocal crashed
        canonical_replace(source, destination)
        if Path(source) == path and Path(destination) == original_main and not crashed:
            crashed = True
            raise RuntimeError("injected source rename crash")

    monkeypatch.setattr(migration.os, "replace", crash_after_main)
    with pytest.raises(RuntimeError, match="injected source rename crash"):
        apply_fts_migration(path)
    assert migration.load_maintenance_journal(path).phase is JournalPhase.SWAPPING
    monkeypatch.setattr(migration.os, "replace", canonical_replace)

    assert resume_fts_migration(path).phase == "complete"
    assert original_main.exists()


def test_resume_completes_candidate_install_after_rename_before_journal(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    canonical_replace = migration.os.replace
    crashed = False

    def crash_after_candidate(source, destination):
        nonlocal crashed
        canonical_replace(source, destination)
        if Path(source).name == "candidate.db" and Path(destination) == path and not crashed:
            crashed = True
            raise RuntimeError("injected candidate rename crash")

    monkeypatch.setattr(migration.os, "replace", crash_after_candidate)
    with pytest.raises(RuntimeError, match="injected candidate rename crash"):
        apply_fts_migration(path)
    assert migration.load_maintenance_journal(path).phase is JournalPhase.OLD_MOVED
    monkeypatch.setattr(migration.os, "replace", canonical_replace)

    assert resume_fts_migration(path).phase == "complete"


def test_zero_frame_sidecars_move_with_original_and_never_remain_live(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    canonical_transition = migration._transition

    def add_zero_sidecars(*args, **kwargs):
        journal = canonical_transition(*args, **kwargs)
        if journal.phase is JournalPhase.CANDIDATE_READY:
            Path(f"{path}-wal").touch()
            Path(f"{path}-shm").touch()
            journal = canonical_transition(
                path,
                journal,
                JournalPhase.CANDIDATE_READY,
                fingerprints=migration._source_inventory_fingerprints(path),
            )
        return journal

    monkeypatch.setattr(migration, "_transition", add_zero_sidecars)

    assert apply_fts_migration(path).phase == "complete"
    assert (tmp_path / "state.db.pre-v2.original-wal").read_bytes() == b""
    assert (tmp_path / "state.db.pre-v2.original-shm").read_bytes() == b""
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()


def test_rollback_resumes_after_candidate_main_quarantine_crash(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    before = _db_digest(path)
    apply_fts_migration(path)
    canonical_replace = migration.os.replace
    quarantine = tmp_path / "state.db.v2.quarantine"
    crashed = False

    def crash_after_quarantine(source, destination):
        nonlocal crashed
        canonical_replace(source, destination)
        if Path(source) == path and Path(destination) == quarantine and not crashed:
            crashed = True
            raise RuntimeError("injected quarantine crash")

    monkeypatch.setattr(migration.os, "replace", crash_after_quarantine)
    with pytest.raises(RuntimeError, match="injected quarantine crash"):
        rollback_fts_migration(path)
    monkeypatch.setattr(migration.os, "replace", canonical_replace)

    assert rollback_fts_migration(path).phase == "rolled_back"
    assert _db_digest(path) == before


def test_abort_rejects_swap_phase_and_requires_rollback(tmp_path):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    journal = replace(
        MaintenanceJournal.new("swap-operation", path),
        phase=JournalPhase.SWAPPING,
        fingerprints=state_db_file_inventory(path),
    )
    write_maintenance_journal(path, journal)

    with pytest.raises(RuntimeError, match="requires rollback"):
        abort_fts_migration(path)


def test_production_checkpoint_returns_exact_empty_wal_tuple(tmp_path):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    finally:
        conn.close()
    journal = replace(
        MaintenanceJournal.new("checkpoint-operation", path),
        phase=JournalPhase.WRITERS_STOPPED,
        fingerprints=state_db_file_inventory(path),
    )
    write_maintenance_journal(path, journal)
    permit = issue_maintenance_permit(
        path, journal.operation_id, frozenset({JournalPhase.WRITERS_STOPPED})
    )

    assert migration._checkpoint_source(path, journal, permit) == (0, 0, 0)


def test_lsof_unavailable_empty_success_and_malformed_output_fail_closed(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)

    def unavailable(paths):
        raise FileNotFoundError("lsof missing")

    monkeypatch.setattr(migration, "_run_lsof", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        find_live_state_db_users(path)
    monkeypatch.setattr(migration, "_run_lsof", lambda paths: _lsof_result(0, ""))
    with pytest.raises(RuntimeError, match="ambiguous"):
        find_live_state_db_users(path)
    monkeypatch.setattr(
        migration, "_run_lsof", lambda paths: _lsof_result(0, "not-machine-output\n")
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        find_live_state_db_users(path)


def test_unknown_sidecar_fingerprint_does_not_partially_move_main(tmp_path):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    wal = Path(f"{path}-wal")
    wal.write_bytes(b"recorded-zero-frame")
    journal = replace(
        MaintenanceJournal.new("unknown-sidecar-operation", path),
        phase=JournalPhase.SWAPPING,
        fingerprints=state_db_file_inventory(path),
    )
    write_maintenance_journal(path, journal)
    wal.write_bytes(b"changed-after-record")
    main_before = path.read_bytes()
    journal_path = tmp_path / "state-db-maintenance.json"
    journal_before = journal_path.read_bytes()

    with pytest.raises(RuntimeError, match="unknown source wal"):
        resume_fts_migration(path)

    assert path.read_bytes() == main_before
    assert not (tmp_path / "state.db.pre-v2.original").exists()
    assert journal_path.read_bytes() == journal_before


def _snapshot_paths(paths):
    return {
        item: (item.read_bytes() if item.exists() and not item.is_symlink() else None)
        for item in paths
    }


def _assert_path_snapshot(snapshot):
    assert _snapshot_paths(snapshot) == snapshot


def _collision_journal(path, phase, fingerprints, **paths):
    journal = replace(
        MaintenanceJournal.new(f"collision-{phase.value}", path),
        phase=phase,
        fingerprints=fingerprints,
        backup_path=str(paths["backup"]) if "backup" in paths else None,
        work_path=str(paths["work"]) if "work" in paths else None,
        candidate_path=str(paths["candidate"]) if "candidate" in paths else None,
    )
    write_maintenance_journal(path, journal)
    return journal


@pytest.mark.parametrize("expected_at", ["pending", "final"])
def test_backup_install_collision_preserves_both_locations_and_journal(
    tmp_path, monkeypatch, expected_at
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    backup = tmp_path / "state.db.pre-v2.backup"
    pending = tmp_path / ".state.db.pre-v2.backup.pending"
    work = tmp_path / ".state.db.fts-v2-work"
    work.mkdir()
    candidate = work / "candidate-build" / "candidate.db"
    expected_path = pending if expected_at == "pending" else backup
    unknown_path = backup if expected_at == "pending" else pending
    expected_path.write_bytes(b"recorded-backup")
    fingerprints = {
        **state_db_file_inventory(path),
        "backup": migration.fingerprint_path(expected_path),
    }
    _collision_journal(
        path,
        JournalPhase.CHECKPOINTED,
        fingerprints,
        backup=backup,
        work=work,
        candidate=candidate,
    )
    unknown_path.write_bytes(b"unknown-backup")
    journal_path = migration.maintenance_journal_path(path)
    snapshot = _snapshot_paths((pending, backup, journal_path))

    with pytest.raises(RuntimeError, match="unknown rollback backup install state"):
        resume_fts_migration(path)

    _assert_path_snapshot(snapshot)


@pytest.mark.parametrize("expected_at", ["live", "original"])
def test_source_to_original_collision_preserves_both_locations_and_journal(
    tmp_path, expected_at
):
    path = tmp_path / "state.db"
    live = path
    original = tmp_path / "state.db.pre-v2.original"
    expected_path = live if expected_at == "live" else original
    unknown_path = original if expected_at == "live" else live
    expected_path.write_bytes(b"recorded-source")
    fingerprints = {"db": migration.fingerprint_path(expected_path), "wal": None, "shm": None}
    journal = _collision_journal(path, JournalPhase.SWAPPING, fingerprints)
    unknown_path.write_bytes(b"unknown-source")
    journal_path = migration.maintenance_journal_path(path)
    snapshot = _snapshot_paths((live, original, journal_path))

    with pytest.raises(RuntimeError, match="unknown source db rename state"):
        migration._move_recorded_bundle_to_original(path, journal)

    _assert_path_snapshot(snapshot)


@pytest.mark.parametrize("expected_at", ["candidate", "live"])
def test_candidate_to_live_collision_preserves_both_locations_and_journal(
    tmp_path, expected_at
):
    path = tmp_path / "state.db"
    candidate = tmp_path / "candidate.db"
    expected_path = candidate if expected_at == "candidate" else path
    unknown_path = path if expected_at == "candidate" else candidate
    expected_path.write_bytes(b"recorded-candidate")
    fingerprints = {
        "candidate": migration.fingerprint_path(expected_path),
        "candidate_db": migration.fingerprint_path(expected_path),
        "candidate_wal": None,
        "candidate_shm": None,
    }
    journal = _collision_journal(
        path, JournalPhase.OLD_MOVED, fingerprints, candidate=candidate
    )
    unknown_path.write_bytes(b"unknown-candidate")
    journal_path = migration.maintenance_journal_path(path)
    snapshot = _snapshot_paths((candidate, path, journal_path))

    with pytest.raises(RuntimeError, match="unknown candidate rename state"):
        migration._install_recorded_candidate(path, journal)

    _assert_path_snapshot(snapshot)


@pytest.mark.parametrize("expected_at", ["original", "live"])
def test_original_to_live_collision_preserves_both_locations_and_journal(
    tmp_path, expected_at
):
    path = tmp_path / "state.db"
    original = tmp_path / "state.db.pre-v2.original"
    expected_path = original if expected_at == "original" else path
    unknown_path = path if expected_at == "original" else original
    expected_path.write_bytes(b"recorded-original")
    fingerprints = {
        "original_db": migration.fingerprint_path(expected_path),
        "original_wal": None,
        "original_shm": None,
    }
    journal = _collision_journal(path, JournalPhase.OLD_MOVED, fingerprints)
    unknown_path.write_bytes(b"unknown-original")
    journal_path = migration.maintenance_journal_path(path)
    snapshot = _snapshot_paths((original, path, journal_path))

    with pytest.raises(RuntimeError, match="unknown original db restore state"):
        migration._restore_original_bundle(path, journal)

    _assert_path_snapshot(snapshot)


@pytest.mark.parametrize("expected_at", ["live", "quarantine"])
def test_live_to_quarantine_collision_preserves_both_locations_and_journal(
    tmp_path, monkeypatch, expected_at
):
    path = tmp_path / "state.db"
    quarantine = tmp_path / "state.db.v2.quarantine"
    original = tmp_path / "state.db.pre-v2.original"
    expected_path = path if expected_at == "live" else quarantine
    unknown_path = quarantine if expected_at == "live" else path
    expected_path.write_bytes(b"recorded-live-v2")
    original.write_bytes(b"recorded-original-v1")
    fingerprints = {
        "db": migration.fingerprint_path(expected_path),
        "wal": None,
        "shm": None,
        "original_db": migration.fingerprint_path(original),
        "original_wal": None,
        "original_shm": None,
        "rollback_activation": migration._status_fingerprint("planned"),
        "rollback_db": migration.fingerprint_path(expected_path),
        "rollback_wal": None,
        "rollback_shm": None,
    }
    journal = _collision_journal(path, JournalPhase.CANDIDATE_LIVE, fingerprints)
    unknown_path.write_bytes(b"unknown-live-v2")
    monkeypatch.setattr(migration, "_run_lsof", lambda paths: _lsof_result(1, ""))
    journal_path = migration.maintenance_journal_path(path)
    snapshot = _snapshot_paths((path, quarantine, original, journal_path))

    with pytest.raises(RuntimeError, match="unknown candidate db quarantine state"):
        rollback_fts_migration(path)

    _assert_path_snapshot(snapshot)


def test_candidate_sidecar_appearing_after_candidate_ready_fails_without_changes(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    canonical_transition = migration._transition
    stopped = False

    def stop_after_candidate_ready(*args, **kwargs):
        nonlocal stopped
        journal = canonical_transition(*args, **kwargs)
        if journal.phase is JournalPhase.CANDIDATE_READY and not stopped:
            stopped = True
            raise RuntimeError("stop after candidate ready")
        return journal

    monkeypatch.setattr(migration, "_transition", stop_after_candidate_ready)
    with pytest.raises(RuntimeError, match="stop after candidate ready"):
        apply_fts_migration(path)
    journal = migration.load_maintenance_journal(path)
    candidate = Path(journal.candidate_path)
    candidate_wal = Path(f"{candidate}-wal")
    candidate_wal.write_bytes(b"unknown-candidate-wal")
    journal_path = migration.maintenance_journal_path(path)
    snapshot = _snapshot_paths((path, candidate, candidate_wal, journal_path))

    with pytest.raises(RuntimeError, match="unknown candidate wal"):
        resume_fts_migration(path)

    _assert_path_snapshot(snapshot)


def test_candidate_sidecar_appearing_immediately_before_install_fails_without_changes(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    canonical_transition = migration._transition
    stopped = False

    def stop_after_old_moved(*args, **kwargs):
        nonlocal stopped
        journal = canonical_transition(*args, **kwargs)
        if journal.phase is JournalPhase.OLD_MOVED and not stopped:
            stopped = True
            raise RuntimeError("stop before candidate install")
        return journal

    monkeypatch.setattr(migration, "_transition", stop_after_old_moved)
    with pytest.raises(RuntimeError, match="stop before candidate install"):
        apply_fts_migration(path)
    journal = migration.load_maintenance_journal(path)
    candidate = Path(journal.candidate_path)
    candidate_shm = Path(f"{candidate}-shm")
    candidate_shm.write_bytes(b"unknown-candidate-shm")
    original = tmp_path / "state.db.pre-v2.original"
    journal_path = migration.maintenance_journal_path(path)
    snapshot = _snapshot_paths((path, original, candidate, candidate_shm, journal_path))

    with pytest.raises(RuntimeError, match="unknown candidate shm"):
        resume_fts_migration(path)

    _assert_path_snapshot(snapshot)


def test_terminal_rollback_requires_durable_two_proof_liveness_activation(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    before = _db_digest(path)
    apply_fts_migration(path)
    quarantine = tmp_path / "state.db.v2.quarantine"
    calls = []
    outcomes = [_lsof_result(1, ""), _lsof_result(0, "p4242\nf1\n")]

    def raced_lsof(paths):
        calls.append(tuple(paths))
        return outcomes.pop(0)

    monkeypatch.setattr(migration, "_run_lsof", raced_lsof)
    with pytest.raises(RuntimeError, match="writers are still live"):
        rollback_fts_migration(path)

    activated = migration.load_maintenance_journal(path)
    assert activated.phase is JournalPhase.CANDIDATE_LIVE
    assert activated.fingerprints["rollback_activation"]["sha256"] == "planned"
    assert len(calls) == 2
    assert _db_digest(path) != before
    assert not quarantine.exists()
    with pytest.raises(MaintenanceBlockedError, match="write blocked"):
        assert_state_db_maintenance_access(path, write_capable=True, permit=None)
    monkeypatch.setattr(migration, "_run_lsof", lambda paths: _lsof_result(1, ""))
    assert rollback_fts_migration(path).phase == "rolled_back"
    assert _db_digest(path) == before


def test_terminal_rollback_proofs_cover_every_extant_split_bundle_path_atomically(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    apply_fts_migration(path)
    original = migration._original_paths(path)
    quarantine = migration._quarantine_paths(path)
    original["wal"].write_bytes(b"split-original-wal")
    quarantine["shm"].write_bytes(b"split-quarantine-shm")
    expected_paths = (
        path,
        original["db"],
        original["wal"],
        quarantine["shm"],
    )
    journal_path = migration.maintenance_journal_path(path)
    file_snapshot = _snapshot_paths(expected_paths)
    journal_before = journal_path.read_bytes()
    calls = []

    def holder_on_nonfirst_root(paths):
        calls.append(tuple(paths))
        assert tuple(paths) == expected_paths
        return _lsof_result(1, "") if len(calls) == 1 else _lsof_result(0, "p5150\nf1\n")

    monkeypatch.setattr(migration, "_run_lsof", holder_on_nonfirst_root)
    with pytest.raises(RuntimeError, match="writers are still live"):
        rollback_fts_migration(path)

    assert calls == [expected_paths, expected_paths]
    _assert_path_snapshot(file_snapshot)
    activated = migration.load_maintenance_journal(path)
    assert activated is not None
    assert activated.phase is JournalPhase.CANDIDATE_LIVE
    assert activated.fingerprints["rollback_activation"]["sha256"] == "planned"
    assert journal_path.read_bytes() != journal_before


def test_activated_rollback_retry_recomputes_complete_split_bundle_path_set(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _candidate_fixture(path)
    _allow_apply(monkeypatch)
    apply_fts_migration(path)
    outcomes = [_lsof_result(1, ""), _lsof_result(0, "p5250\nf1\n")]
    monkeypatch.setattr(migration, "_run_lsof", lambda paths: outcomes.pop(0))
    with pytest.raises(RuntimeError, match="writers are still live"):
        rollback_fts_migration(path)

    original = migration._original_paths(path)
    quarantine = migration._quarantine_paths(path)
    live = migration._bundle_paths(path)
    live["wal"].write_bytes(b"retry-live-wal")
    original["shm"].write_bytes(b"retry-original-shm")
    quarantine["db"].write_bytes(b"retry-quarantine-main")
    expected_paths = (
        live["db"],
        live["wal"],
        original["db"],
        original["shm"],
        quarantine["db"],
    )
    journal_path = migration.maintenance_journal_path(path)
    snapshot = _snapshot_paths((*expected_paths, journal_path))
    calls = []

    def retry_holder_on_nonfirst_root(paths):
        calls.append(tuple(paths))
        assert tuple(paths) == expected_paths
        return _lsof_result(0, "p5350\nf1\n")

    monkeypatch.setattr(migration, "_run_lsof", retry_holder_on_nonfirst_root)
    with pytest.raises(RuntimeError, match="writers are still live"):
        rollback_fts_migration(path)

    assert calls == [expected_paths]
    _assert_path_snapshot(snapshot)
