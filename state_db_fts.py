"""Schema ownership helpers for Hermes state DB full-text indexes."""

from __future__ import annotations

import re
import sqlite3
from typing import Literal

FtsSchemaKind = Literal["missing", "v1_inline", "v2_external", "inconsistent"]

FULL_PROJECTION_SQL = (
    "coalesce(content,'') || ' ' || "
    "coalesce(tool_name,'') || ' ' || "
    "coalesce(tool_calls,'')"
)


def _projection(row: str) -> str:
    return (
        f"coalesce({row}.content,'') || ' ' || "
        f"coalesce({row}.tool_name,'') || ' ' || "
        f"coalesce({row}.tool_calls,'')"
    )

FTS_V2_VIEW_SQL = """
CREATE VIEW IF NOT EXISTS messages_fts_unicode_content_v2 AS
SELECT id, coalesce(content,'') || ' ' || coalesce(tool_name,'') || ' ' || coalesce(tool_calls,'') AS content
FROM messages;
CREATE VIEW IF NOT EXISTS messages_fts_trigram_content_v2 AS
SELECT id, coalesce(content,'') || ' ' || coalesce(tool_name,'') || ' ' || coalesce(tool_calls,'') AS content
FROM messages WHERE role IN ('user','assistant');
"""

FTS_V2_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  content,
  content='messages_fts_unicode_content_v2',
  content_rowid='id'
);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
  content,
  content='messages_fts_trigram_content_v2',
  content_rowid='id',
  tokenize='trigram'
);
"""

_UNICODE_NEW = _projection("new")
_UNICODE_OLD = _projection("old")

FTS_V2_TRIGGER_SQL = f"""
CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, {_UNICODE_NEW});
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content)
  VALUES ('delete', old.id, {_UNICODE_OLD});
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content)
  VALUES ('delete', old.id, {_UNICODE_OLD});
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, {_UNICODE_NEW});
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages
WHEN new.role IN ('user','assistant') BEGIN
  INSERT INTO messages_fts_trigram(rowid, content) VALUES (new.id, {_UNICODE_NEW});
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages
WHEN old.role IN ('user','assistant') BEGIN
  INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content)
  VALUES ('delete', old.id, {_UNICODE_OLD});
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content)
  SELECT 'delete', old.id, {_UNICODE_OLD}
  WHERE old.role IN ('user','assistant');
  INSERT INTO messages_fts_trigram(rowid, content)
  SELECT new.id, {_UNICODE_NEW}
  WHERE new.role IN ('user','assistant');
END;
"""

FTS_V1_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content);
CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, {_UNICODE_NEW});
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
  DELETE FROM messages_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
  DELETE FROM messages_fts WHERE rowid = old.id;
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, {_UNICODE_NEW});
END;
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(content, tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts_trigram(rowid, content) VALUES (new.id, {_UNICODE_NEW});
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
  DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
  DELETE FROM messages_fts_trigram WHERE rowid = old.id;
  INSERT INTO messages_fts_trigram(rowid, content) VALUES (new.id, {_UNICODE_NEW});
END;
"""


def execute_ddl(conn: sqlite3.Connection, sql: str) -> None:
    """Execute a SQLite script without the implicit COMMIT of executescript()."""
    cursor = conn.cursor()
    statement = ""
    for line in sql.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            if statement.strip():
                cursor.execute(statement)
            statement = ""
    if statement.strip():
        raise ValueError("incomplete SQL statement")


def _master(conn: sqlite3.Connection) -> dict[str, tuple[str, str]]:
    return {
        row[0]: (row[1], row[2] or "")
        for row in conn.execute(
            "SELECT name, type, sql FROM sqlite_master WHERE name LIKE 'messages_fts%'"
        )
    }


def _marker(conn: sqlite3.Connection) -> str | None:
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_meta'"
    ).fetchone() is None:
        return None
    row = conn.execute(
        "SELECT value FROM state_meta WHERE key='fts_schema_version'"
    ).fetchone()
    return None if row is None else str(row[0])


_FTS5_OPTION_RE = re.compile(
    r"\b(?P<key>content_rowid|content|tokenize)\s*=\s*"
    r"(?P<value>'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|"
    r"\[(?:]]|[^]])*\]|[^,\s)]+)",
    re.IGNORECASE,
)


def _fts5_options(sql: str) -> dict[str, str] | None:
    """Return normalized FTS5 options, independent of SQL quoting/spacing."""
    if re.search(r"\busing\s+fts5\s*\(", sql, re.IGNORECASE) is None:
        return None
    options: dict[str, str] = {}
    for match in _FTS5_OPTION_RE.finditer(sql):
        key = match.group("key").lower()
        if key in options:
            return None
        value = match.group("value").strip()
        if value[:1] in {"'", '"', "`", "["}:
            closing = "]" if value[0] == "[" else value[0]
            if not value.endswith(closing):
                return None
            value = value[1:-1]
            if closing != "]":
                value = value.replace(closing * 2, closing)
        options[key] = value.lower()
    return options


def detect_fts_schema(conn: sqlite3.Connection) -> FtsSchemaKind:
    """Purely classify FTS ownership; contradictions are never guessed."""
    objects = _master(conn)
    marker = _marker(conn)
    core_names = {
        "messages_fts",
        "messages_fts_trigram",
        "messages_fts_unicode_content_v2",
        "messages_fts_trigram_content_v2",
    }
    present = core_names.intersection(objects)
    if not present:
        return "missing" if marker is None else "inconsistent"

    base = objects.get("messages_fts")
    trigram = objects.get("messages_fts_trigram")
    unicode_view = objects.get("messages_fts_unicode_content_v2")
    trigram_view = objects.get("messages_fts_trigram_content_v2")
    if base is None or base[0] != "table":
        return "inconsistent"
    if trigram is not None and trigram[0] != "table":
        return "inconsistent"

    base_options = _fts5_options(base[1])
    trigram_options = None if trigram is None else _fts5_options(trigram[1])
    if base_options is None or (trigram is not None and trigram_options is None):
        return "inconsistent"
    base_has_external = "content" in base_options
    trigram_has_external = (
        trigram_options is not None and "content" in trigram_options
    )
    base_external = (
        base_options.get("content") == "messages_fts_unicode_content_v2"
        and base_options.get("content_rowid") == "id"
    )
    trigram_external = (
        trigram_options is not None
        and trigram_options.get("content") == "messages_fts_trigram_content_v2"
        and trigram_options.get("content_rowid") == "id"
        and trigram_options.get("tokenize") == "trigram"
    )
    views_valid = (
        unicode_view is not None
        and unicode_view[0] == "view"
        and trigram_view is not None
        and trigram_view[0] == "view"
    )
    if views_valid:
        unicode_sql = " ".join(unicode_view[1].lower().split())
        trigram_view_sql = " ".join(trigram_view[1].lower().split())
        projection_fields = ("coalesce(content", "coalesce(tool_name", "coalesce(tool_calls")
        views_valid = (
            all(field in unicode_sql for field in projection_fields)
            and " where " not in unicode_sql
            and all(field in trigram_view_sql for field in projection_fields)
            and "whererolein('user','assistant')" in trigram_view_sql.replace(" ", "")
        )

    trigger_sql = {
        name: " ".join(sql.lower().split())
        for name, (type_, sql) in objects.items()
        if type_ == "trigger" and name in {
            "messages_fts_insert",
            "messages_fts_delete",
            "messages_fts_update",
            "messages_fts_trigram_insert",
            "messages_fts_trigram_delete",
            "messages_fts_trigram_update",
        }
    }

    def triggers_match(external: bool) -> bool:
        for name, sql in trigger_sql.items():
            is_delete_path = name.endswith("_delete") or name.endswith("_update")
            if is_delete_path:
                has_external_delete = "'delete'" in sql and "insert into messages_fts" in sql
                if has_external_delete != external:
                    return False
            if name.startswith("messages_fts_trigram") and external:
                if "rolein('user','assistant')" not in sql.replace(" ", ""):
                    return False
            if not name.endswith("_delete") and (
                "tool_name" not in sql or "tool_calls" not in sql
            ):
                return False
        return True

    if not base_has_external and not trigram_has_external and not views_valid:
        return "v1_inline" if marker in (None, "1") and triggers_match(False) else "inconsistent"
    if base_external and trigram_external and views_valid:
        shadows = {"messages_fts_content", "messages_fts_trigram_content"}
        return (
            "v2_external"
            if marker == "2" and not shadows.intersection(objects) and triggers_match(True)
            else "inconsistent"
        )
    return "inconsistent"


def create_fts_v1(conn: sqlite3.Connection) -> None:
    execute_ddl(conn, FTS_V1_SQL)


def create_fts_v2(conn: sqlite3.Connection) -> None:
    execute_ddl(conn, FTS_V2_VIEW_SQL)
    execute_ddl(conn, FTS_V2_TABLE_SQL)
    execute_ddl(conn, FTS_V2_TRIGGER_SQL)


def integrity_check_fts_v2(conn: sqlite3.Connection) -> None:
    for table in ("messages_fts", "messages_fts_trigram"):
        conn.execute(
            f"INSERT INTO {table}({table}, rank) VALUES('integrity-check', 1)"
        )


def rebuild_fts(conn: sqlite3.Connection, kind: FtsSchemaKind) -> None:
    if kind == "v2_external":
        for table in ("messages_fts", "messages_fts_trigram"):
            conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
        integrity_check_fts_v2(conn)
        return
    if kind == "v1_inline":
        conn.execute("DELETE FROM messages_fts")
        conn.execute(
            "INSERT INTO messages_fts(rowid, content) "
            "SELECT id, coalesce(content,'') || ' ' || coalesce(tool_name,'') || ' ' || coalesce(tool_calls,'') FROM messages"
        )
        conn.execute("DELETE FROM messages_fts_trigram")
        conn.execute(
            "INSERT INTO messages_fts_trigram(rowid, content) "
            "SELECT id, coalesce(content,'') || ' ' || coalesce(tool_name,'') || ' ' || coalesce(tool_calls,'') FROM messages"
        )
        return
    raise RuntimeError(f"cannot rebuild inconsistent FTS schema: {kind}")
