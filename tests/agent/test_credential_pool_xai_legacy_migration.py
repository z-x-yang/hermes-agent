"""Regression tests for the pre-0.18 xAI OAuth ``loopback_pkce`` → ``device_code``
pool migration.

0.17 surfaced the xAI OAuth singleton in the credential pool with
``source="loopback_pkce"``; 0.18 unified every device-code singleton onto
``source="device_code"``.  Without an explicit load-time migration, a pool
persisted by 0.17 ends up with *two* entries for the same single-use OAuth
refresh token — the legacy ``loopback_pkce`` row plus a freshly seeded
``device_code`` row — because the singleton seed always upserts ``device_code``
and ``_upsert_entry`` matches by exact source.  Rotation across that pair
replays a consumed refresh token and terminally quarantines the lineage.

``_canonicalize_legacy_singleton_sources`` rewrites the legacy row to
``device_code`` before seed/prune so the lineage stays a single credential.
"""

from __future__ import annotations

import json

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    PooledCredential,
    _canonicalize_legacy_singleton_sources,
    load_pool,
)


def _entry(source: str, *, id: str, access_token: str, refresh_token: str) -> PooledCredential:
    return PooledCredential(
        provider="xai-oauth",
        id=id,
        label="cred",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=source,
        access_token=access_token,
        refresh_token=refresh_token,
    )


# ── pure-function behaviour ─────────────────────────────────────────────────


def test_canonicalize_rewrites_legacy_source_in_place_preserving_tokens():
    entries = [_entry("loopback_pkce", id="e1", access_token="at", refresh_token="rt")]
    changed = _canonicalize_legacy_singleton_sources("xai-oauth", entries)
    assert changed is True
    assert len(entries) == 1
    assert entries[0].source == "device_code"
    # Tokens (and id) survive the rewrite — this is the same credential lineage.
    assert entries[0].access_token == "at"
    assert entries[0].refresh_token == "rt"
    assert entries[0].id == "e1"


def test_canonicalize_dedupes_when_device_code_already_present():
    entries = [
        _entry("device_code", id="e1", access_token="at1", refresh_token="rt1"),
        _entry("loopback_pkce", id="e2", access_token="at1", refresh_token="rt1"),
    ]
    changed = _canonicalize_legacy_singleton_sources("xai-oauth", entries)
    assert changed is True
    # The legacy duplicate is dropped, not merged into a second device_code row.
    assert len(entries) == 1
    assert entries[0].source == "device_code"
    assert entries[0].id == "e1"


def test_canonicalize_is_noop_when_already_device_code():
    entries = [_entry("device_code", id="e1", access_token="at", refresh_token="rt")]
    changed = _canonicalize_legacy_singleton_sources("xai-oauth", entries)
    assert changed is False
    assert len(entries) == 1
    assert entries[0].source == "device_code"


def test_canonicalize_ignores_manual_and_other_sources():
    entries = [
        _entry("manual:key", id="e1", access_token="at", refresh_token="rt"),
        _entry("device_code", id="e2", access_token="at2", refresh_token="rt2"),
    ]
    changed = _canonicalize_legacy_singleton_sources("xai-oauth", entries)
    assert changed is False
    assert {e.source for e in entries} == {"manual:key", "device_code"}


def test_canonicalize_only_touches_xai_oauth():
    entries = [_entry("loopback_pkce", id="e1", access_token="at", refresh_token="rt")]
    # A non-xai provider must not have its sources rewritten.
    changed = _canonicalize_legacy_singleton_sources("nous", entries)
    assert changed is False
    assert entries[0].source == "loopback_pkce"


# ── partial-migration token safety (codex re-review Finding 1) ──────────────


def test_canonicalize_keeps_valid_legacy_over_empty_device_code():
    """Stale/empty canonical row + token-bearing legacy row: the surviving
    device_code row MUST carry the legacy row's tokens, not the empty ones.
    Blindly keeping device_code here would delete the only usable credential."""
    entries = [
        _entry("device_code", id="canon", access_token="", refresh_token=""),
        _entry("loopback_pkce", id="legacy", access_token="valid-at", refresh_token="valid-rt"),
    ]
    changed = _canonicalize_legacy_singleton_sources("xai-oauth", entries)
    assert changed is True
    assert len(entries) == 1
    assert entries[0].source == "device_code"
    assert entries[0].refresh_token == "valid-rt"
    assert entries[0].access_token == "valid-at"


def test_canonicalize_multiple_legacy_rows_keeps_the_token_bearing_one():
    """First legacy row empty, second holds the tokens: the tokens survive,
    not the row that happens to come first in list order."""
    entries = [
        _entry("loopback_pkce", id="empty", access_token="", refresh_token=""),
        _entry("loopback_pkce", id="valid", access_token="a", refresh_token="r"),
    ]
    changed = _canonicalize_legacy_singleton_sources("xai-oauth", entries)
    assert changed is True
    assert len(entries) == 1
    assert entries[0].source == "device_code"
    assert entries[0].refresh_token == "r"


def test_canonicalize_conflicting_tokens_keeps_canonical_and_warns(caplog):
    """Two rows with DISTINCT valid refresh tokens is an unarbitrable conflict:
    keep the canonical row's token (never lose it silently) and log a warning."""
    import logging

    entries = [
        _entry("device_code", id="canon", access_token="a1", refresh_token="r1"),
        _entry("loopback_pkce", id="legacy", access_token="a2", refresh_token="r2"),
    ]
    with caplog.at_level(logging.WARNING):
        changed = _canonicalize_legacy_singleton_sources("xai-oauth", entries)
    assert changed is True
    assert len(entries) == 1
    assert entries[0].source == "device_code"
    # Canonical row's token is retained (not silently swapped for the legacy one).
    assert entries[0].refresh_token == "r1"
    assert any("conflicting refresh tokens" in rec.message for rec in caplog.records)


# ── end-to-end load_pool (the exact regression codex flagged) ───────────────


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def test_load_pool_migrates_legacy_row_without_duplicating_credential(tmp_path, monkeypatch):
    """A pre-0.18 pool (loopback_pkce row) + live singleton tokens must load to
    exactly ONE xai-oauth credential, on the canonical device_code source, with
    tokens intact — not two rows sharing one single-use refresh token."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "live-access",
                        "refresh_token": "live-refresh",
                    }
                }
            },
            "credential_pool": {
                "xai-oauth": [
                    {
                        "id": "legacy1",
                        "label": "grok",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "loopback_pkce",
                        "access_token": "live-access",
                        "refresh_token": "live-refresh",
                    }
                ]
            },
        },
    )

    pool = load_pool("xai-oauth")
    entries = pool._entries

    xai_entries = [e for e in entries if e.provider == "xai-oauth"]
    assert len(xai_entries) == 1, (
        f"expected a single migrated credential, got sources "
        f"{[e.source for e in xai_entries]}"
    )
    assert xai_entries[0].source == "device_code"
    # The refresh token survived the migration through the disk-boundary sanitizer.
    assert xai_entries[0].refresh_token == "live-refresh"


def test_load_pool_persists_migrated_source_to_disk(tmp_path, monkeypatch):
    """After the migrating load, auth.json no longer carries a loopback_pkce row,
    so subsequent loads are plain no-ops (idempotent migration)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {"access_token": "a", "refresh_token": "r"}
                }
            },
            "credential_pool": {
                "xai-oauth": [
                    {
                        "id": "legacy1",
                        "label": "grok",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "loopback_pkce",
                        "access_token": "a",
                        "refresh_token": "r",
                    }
                ]
            },
        },
    )

    load_pool("xai-oauth")

    store = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    pool_rows = store.get("credential_pool", {}).get("xai-oauth", [])
    sources = [row.get("source") for row in pool_rows]
    assert "loopback_pkce" not in sources
    assert sources.count("device_code") == 1
