from __future__ import annotations

import sqlite3

from hermes_state import SessionDB


def test_compression_replay_anchor_round_trips_across_db_reopen(tmp_path):
    db_path = tmp_path / "state.db"
    session_id = "session-1"
    provider_request = {
        "model": "gpt-5.6-sol",
        "instructions": "stable system",
        "input": [
            {"role": "user", "content": "hello"},
            {
                "id": "rs_1",
                "type": "reasoning",
                "encrypted_content": "opaque-replay-payload",
            },
        ],
        "tools": [{"type": "function", "name": "demo"}],
        "prompt_cache_key": "session-cache-scope",
    }

    db = SessionDB(db_path)
    db.create_session(session_id, "discord")
    db.upsert_compression_replay_anchor(
        session_id,
        source_message_count=2,
        provider_input_tokens=84_000,
        target_tokens=80_000,
        prefix_fingerprint="prefix-fingerprint",
        provider_request=provider_request,
    )
    db.close()

    reopened = SessionDB(db_path)
    anchor = reopened.get_compression_replay_anchor(session_id)

    assert anchor is not None
    assert anchor["source_message_count"] == 2
    assert anchor["provider_input_tokens"] == 84_000
    assert anchor["target_tokens"] == 80_000
    assert anchor["prefix_fingerprint"] == "prefix-fingerprint"
    assert anchor["provider_request"] == provider_request
    assert anchor["payload_sha256"]


def test_compression_replay_anchor_corruption_fails_closed(tmp_path):
    db_path = tmp_path / "state.db"
    session_id = "session-corrupt"
    db = SessionDB(db_path)
    db.create_session(session_id, "discord")
    db.upsert_compression_replay_anchor(
        session_id,
        source_message_count=1,
        provider_input_tokens=84_000,
        target_tokens=80_000,
        prefix_fingerprint="prefix-fingerprint",
        provider_request={"messages": [{"role": "user", "content": "hello"}]},
    )
    db.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE compression_replay_anchors SET provider_request_zlib = ? "
            "WHERE session_id = ?",
            (b"not-zlib", session_id),
        )

    reopened = SessionDB(db_path)
    assert reopened.get_compression_replay_anchor(session_id) is None


def test_compression_replay_anchor_unknown_schema_version_fails_closed(tmp_path):
    db_path = tmp_path / "state.db"
    session_id = "session-future-schema"
    db = SessionDB(db_path)
    db.create_session(session_id, "discord")
    db.upsert_compression_replay_anchor(
        session_id,
        source_message_count=1,
        provider_input_tokens=84_000,
        target_tokens=80_000,
        prefix_fingerprint="prefix-fingerprint",
        provider_request={"messages": [{"role": "user", "content": "hello"}]},
    )
    db.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE compression_replay_anchors SET schema_version = 2 "
            "WHERE session_id = ?",
            (session_id,),
        )

    reopened = SessionDB(db_path)
    assert reopened.get_compression_replay_anchor(session_id) is None


def test_compression_replay_anchor_is_immutable_until_deleted(tmp_path):
    db_path = tmp_path / "state.db"
    session_id = "session-immutable"
    db = SessionDB(db_path)
    db.create_session(session_id, "discord")
    db.upsert_compression_replay_anchor(
        session_id,
        source_message_count=2,
        provider_input_tokens=79_000,
        target_tokens=80_000,
        prefix_fingerprint="first-fingerprint",
        provider_request={"messages": [{"role": "user", "content": "first"}]},
    )
    db.upsert_compression_replay_anchor(
        session_id,
        source_message_count=9,
        provider_input_tokens=99_000,
        target_tokens=80_000,
        prefix_fingerprint="later-fingerprint",
        provider_request={"messages": [{"role": "user", "content": "later"}]},
    )

    anchor = db.get_compression_replay_anchor(session_id)
    assert anchor is not None
    assert anchor["source_message_count"] == 2
    assert anchor["provider_input_tokens"] == 79_000
    assert anchor["prefix_fingerprint"] == "first-fingerprint"
    assert anchor["provider_request"] == {
        "messages": [{"role": "user", "content": "first"}]
    }
