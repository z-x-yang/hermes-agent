import sqlite3

import pytest

import hermes_state
from hermes_state import SessionDB
from state_db_fts import (
    create_fts_v1,
    detect_fts_schema,
    integrity_check_fts_v2,
    rebuild_fts,
)


TRIGGERS = (
    "messages_fts_insert",
    "messages_fts_delete",
    "messages_fts_update",
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
)


def _schema_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'messages_fts%'"
        )
    }


def _make_v1_fixture(path) -> None:
    db = SessionDB(path)
    db.create_session(session_id="legacy", source="cli")
    db.append_message("legacy", role="user", content="legacy needle")
    for trigger in TRIGGERS:
        db._conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    db._conn.execute("DROP TABLE messages_fts_trigram")
    db._conn.execute("DROP TABLE messages_fts")
    db._conn.execute("DROP VIEW messages_fts_trigram_content_v2")
    db._conn.execute("DROP VIEW messages_fts_unicode_content_v2")
    db._conn.execute("DELETE FROM state_meta WHERE key='fts_schema_version'")
    create_fts_v1(db._conn)
    rebuild_fts(db._conn, "v1_inline")
    db._conn.commit()
    db.close()


def _schema_and_meta(path) -> tuple[list[tuple], list[tuple]]:
    conn = sqlite3.connect(path)
    try:
        schema = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        meta = conn.execute("SELECT key, value FROM state_meta ORDER BY key").fetchall()
        return schema, meta
    finally:
        conn.close()


def _replace_v2_table(path, table: str, sql: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"DROP TABLE {table}")
        conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


def test_new_empty_db_uses_v2_without_content_shadows(tmp_path):
    db = SessionDB(tmp_path / "new.db")
    try:
        names = _schema_names(db._conn)
        assert "messages_fts_unicode_content_v2" in names
        assert "messages_fts_trigram_content_v2" in names
        assert "messages_fts_content" not in names
        assert "messages_fts_trigram_content" not in names
        assert db.get_meta("fts_schema_version") == "2"
        assert detect_fts_schema(db._conn) == "v2_external"
    finally:
        db.close()


def test_legacy_inline_db_opens_without_schema_or_meta_write(tmp_path):
    path = tmp_path / "legacy.db"
    _make_v1_fixture(path)
    before = _schema_and_meta(path)
    db = SessionDB(path)
    try:
        assert db._fts_effective_schema == "v1_inline"
        assert db.get_meta("fts_schema_version") is None
    finally:
        db.close()
    assert _schema_and_meta(path) == before


def test_external_schema_without_marker_fails_closed(tmp_path):
    path = tmp_path / "external-no-marker.db"
    db = SessionDB(path)
    db._conn.execute("DELETE FROM state_meta WHERE key='fts_schema_version'")
    db._conn.commit()
    db.close()

    with pytest.raises(RuntimeError, match="inconsistent FTS schema"):
        SessionDB(path)


def test_v1_missing_trigger_repairs_with_inline_owner(tmp_path):
    path = tmp_path / "v1-repair.db"
    _make_v1_fixture(path)
    conn = sqlite3.connect(path)
    conn.execute("DROP TRIGGER messages_fts_update")
    conn.commit()
    conn.close()

    db = SessionDB(path)
    try:
        assert detect_fts_schema(db._conn) == "v1_inline"
        sql = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='messages_fts_update'"
        ).fetchone()[0]
        assert "DELETE FROM messages_fts" in sql
        assert db.get_meta("fts_schema_version") is None
    finally:
        db.close()


def test_v2_trigger_projection_handles_role_transitions_and_integrity(tmp_path):
    db = SessionDB(tmp_path / "v2-transitions.db")
    try:
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1",
            role="assistant",
            content="visible",
            tool_name="runner",
            tool_calls='{"query":"投影针"}',
        )
        rowid = db._conn.execute(
            "SELECT id FROM messages WHERE session_id='s1'"
        ).fetchone()[0]
        # Store a legacy raw JSON projection exactly as old databases can contain
        # it; the update trigger must index tool_calls, not only visible content.
        db._conn.execute(
            "UPDATE messages SET tool_calls=? WHERE id=?",
            ('{"query":"投影针"}', rowid),
        )
        assert db._conn.execute(
            "SELECT count(*) FROM messages_fts_trigram WHERE messages_fts_trigram MATCH '投影针'"
        ).fetchone()[0] == 1

        db._conn.execute("UPDATE messages SET role='tool' WHERE id=?", (rowid,))
        assert db._conn.execute(
            "SELECT count(*) FROM messages_fts_trigram WHERE messages_fts_trigram MATCH '投影针'"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'runner'"
        ).fetchone()[0] == 1

        db._conn.execute("UPDATE messages SET role='user' WHERE id=?", (rowid,))
        assert db._conn.execute(
            "SELECT count(*) FROM messages_fts_trigram WHERE messages_fts_trigram MATCH '投影针'"
        ).fetchone()[0] == 1
        integrity_check_fts_v2(db._conn)
        rebuild_fts(db._conn, "v2_external")
        db._conn.execute("DELETE FROM messages WHERE id=?", (rowid,))
        assert db._conn.execute(
            "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'runner'"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT count(*) FROM messages_fts_trigram WHERE messages_fts_trigram MATCH '投影针'"
        ).fetchone()[0] == 0
        integrity_check_fts_v2(db._conn)
    finally:
        db.close()


def test_v2_missing_trigger_repairs_with_external_owner(tmp_path):
    path = tmp_path / "v2-repair.db"
    db = SessionDB(path)
    db.create_session(session_id="s1", source="cli")
    db.append_message("s1", role="assistant", content="before repair")
    db._conn.execute("DROP TRIGGER messages_fts_update")
    db._conn.commit()
    db.close()

    repaired = SessionDB(path)
    try:
        sql = repaired._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='messages_fts_update'"
        ).fetchone()[0]
        assert "VALUES ('delete', old.id" in sql
        assert detect_fts_schema(repaired._conn) == "v2_external"
        integrity_check_fts_v2(repaired._conn)
    finally:
        repaired.close()


def test_new_v2_schema_and_marker_roll_back_together(tmp_path, monkeypatch):
    path = tmp_path / "atomic.db"
    real_create = hermes_state.create_fts_v2

    def create_then_fail(conn):
        real_create(conn)
        raise RuntimeError("injected after v2 DDL")

    monkeypatch.setattr(hermes_state, "create_fts_v2", create_then_fail)
    with pytest.raises(RuntimeError, match="injected after v2 DDL"):
        SessionDB(path)

    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE name LIKE 'messages_fts%'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='state_meta'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_v2_wrong_owner_trigger_is_inconsistent(tmp_path):
    path = tmp_path / "wrong-owner.db"
    db = SessionDB(path)
    try:
        db._conn.execute("DROP TRIGGER messages_fts_delete")
        db._conn.execute(
            "CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN "
            "DELETE FROM messages_fts WHERE rowid=old.id; END"
        )
        assert detect_fts_schema(db._conn) == "inconsistent"
    finally:
        db.close()


@pytest.mark.parametrize(
    ("table", "sql"),
    [
        (
            "messages_fts",
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "content, content='messages_fts_unicode_content_v2')",
        ),
        (
            "messages_fts",
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "content, content='messages_fts_unicode_content_v2', "
            "content_rowid='wrong_id')",
        ),
        (
            "messages_fts",
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "content, content_rowid='id')",
        ),
        (
            "messages_fts",
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "content, content='wrong_unicode_owner', content_rowid='id')",
        ),
        (
            "messages_fts_trigram",
            "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5("
            "content, content='messages_fts_trigram_content_v2', tokenize='trigram')",
        ),
        (
            "messages_fts_trigram",
            "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5("
            "content, content='messages_fts_trigram_content_v2', "
            "content_rowid='wrong_id', tokenize='trigram')",
        ),
        (
            "messages_fts_trigram",
            "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5("
            "content, content='messages_fts_trigram_content_v2', content_rowid='id')",
        ),
        (
            "messages_fts_trigram",
            "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5("
            "content, content='messages_fts_trigram_content_v2', "
            "content_rowid='id', tokenize='unicode61')",
        ),
        (
            "messages_fts_trigram",
            "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5("
            "content, content_rowid='id', tokenize='trigram')",
        ),
        (
            "messages_fts_trigram",
            "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5("
            "content, content='wrong_trigram_owner', "
            "content_rowid='id', tokenize='trigram')",
        ),
    ],
    ids=[
        "unicode-missing-content-rowid",
        "unicode-wrong-content-rowid",
        "unicode-missing-owner",
        "unicode-wrong-owner",
        "trigram-missing-content-rowid",
        "trigram-wrong-content-rowid",
        "trigram-missing-tokenizer",
        "trigram-wrong-tokenizer",
        "trigram-missing-owner",
        "trigram-wrong-owner",
    ],
)
def test_malformed_v2_table_fails_closed_without_startup_writes(
    tmp_path, table, sql
):
    path = tmp_path / "malformed-v2-table.db"
    SessionDB(path).close()
    _replace_v2_table(path, table, sql)

    conn = sqlite3.connect(path)
    try:
        assert detect_fts_schema(conn) == "inconsistent"
    finally:
        conn.close()

    before = _schema_and_meta(path)
    with pytest.raises(RuntimeError, match="inconsistent FTS schema"):
        SessionDB(path)
    assert _schema_and_meta(path) == before


def test_v2_table_detection_accepts_normalized_quoting_and_whitespace(tmp_path):
    path = tmp_path / "quoted-v2-table.db"
    SessionDB(path).close()
    _replace_v2_table(
        path,
        "messages_fts",
        '''CREATE VIRTUAL TABLE messages_fts USING fts5(
               content,
               content = "messages_fts_unicode_content_v2",
               content_rowid = "id"
           )''',
    )
    _replace_v2_table(
        path,
        "messages_fts_trigram",
        '''CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
               content,
               content = "messages_fts_trigram_content_v2",
               content_rowid = "id",
               tokenize = "trigram"
           )''',
    )

    conn = sqlite3.connect(path)
    try:
        assert detect_fts_schema(conn) == "v2_external"
    finally:
        conn.close()
