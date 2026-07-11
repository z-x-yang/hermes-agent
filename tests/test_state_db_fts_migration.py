from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from state_db_fts import create_fts_v1, rebuild_fts
from state_db_maintenance import MaintenanceJournal, write_maintenance_journal
from state_db_fts_migration import (
    estimate_payload_retention,
    plan_fts_migration,
    status_fts_migration,
)


def _make_v1_fixture(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            started_at REAL NOT NULL,
            ended_at REAL,
            archived INTEGER NOT NULL DEFAULT 0
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
        """
    )
    conn.executemany(
        "INSERT INTO sessions(id, source, started_at, ended_at, archived) VALUES(?,?,?,?,?)",
        [
            ("active-session", "cli", 100.0, None, 0),
            ("held-archive", "cli", 50.0, 60.0, 1),
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
