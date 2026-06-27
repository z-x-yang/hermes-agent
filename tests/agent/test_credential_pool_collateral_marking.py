"""Regression test for cross-key collateral marking in credential-pool recovery.

Real-world incident (gptcodex, June 2026): a session delegated work to several
concurrent subagents sharing one credential pool with two keys — a healthy
``cheap-primary`` and an ``expensive-backup`` whose upstream billing group had
been disabled (HTTP 403 ``GROUP_DISABLED``). Only subagents that leased the
backup key should have failed.

Instead BOTH pool entries ended up marked ``exhausted`` with
``last_error_reason=GROUP_DISABLED`` — including the healthy primary, which the
live API still accepts. Root cause: ``recover_with_credential_pool`` called
``mark_exhausted_and_rotate`` WITHOUT ``api_key_hint``, so the pool fell back to
the shared ``current()`` pointer to decide which entry to mark. Under concurrent
``acquire_lease`` calls that pointer is overwritten by other subagents, so the
failing backup key's 403 got recorded against whichever key ``current()``
happened to reference — collaterally poisoning the healthy key.

``auxiliary_client._recover_provider_pool`` already passes ``api_key_hint`` (the
exact key the failing request used). The main-agent recovery path must do the
same so the *actually failing* key is marked, never an innocent bystander.
"""
from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.error_classifier import FailoverReason

GPTCODEX_URL = "https://gptcodex.top/v1"
HEALTHY_KEY = "sk-AAAAAAAA-healthy-primary-key"
DISABLED_KEY = "sk-BBBBBBBB-disabled-group-key"


def _write_two_key_pool(tmp_path):
    import json

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "custom:gptcodex": [
                        {
                            "id": "key-a",
                            "label": "cheap-primary",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual",
                            "access_token": HEALTHY_KEY,
                            "base_url": GPTCODEX_URL,
                        },
                        {
                            "id": "key-b",
                            "label": "expensive-backup",
                            "auth_type": "api_key",
                            "priority": 1,
                            "source": "manual",
                            "access_token": DISABLED_KEY,
                            "base_url": GPTCODEX_URL,
                        },
                    ]
                },
            },
            indent=2,
        )
    )


def test_auth_403_marks_failing_key_not_current_pointer(tmp_path, monkeypatch):
    """A 403 on the backup key must mark the backup key — never the healthy
    primary that ``current()`` happens to point at after a concurrent lease."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_two_key_pool(tmp_path)

    from agent.credential_pool import load_pool

    pool = load_pool("custom:gptcodex")

    # Concurrency simulation: another subagent's lease moved current() to the
    # HEALTHY key-A, but the request that just 403'd was made with key-B.
    pool.acquire_lease("key-a")
    current = pool.current()
    assert current is not None
    assert current.id == "key-a"

    agent = MagicMock()
    agent.provider = "custom"
    agent.base_url = GPTCODEX_URL
    agent._credential_pool = pool
    agent.client.api_key = DISABLED_KEY  # the key the FAILING request actually used
    agent._is_entitlement_failure.return_value = False

    with patch(
        "agent.credential_pool.get_custom_provider_pool_key",
        return_value="custom:gptcodex",
    ):
        recover_with_credential_pool(
            agent,
            status_code=403,
            has_retried_429=False,
            classified_reason=FailoverReason.auth,
            error_context={
                "code": "GROUP_DISABLED",
                "message": "API Key 所属分组已停用",
            },
        )

    by_id = {e.id: e for e in pool.entries()}
    assert by_id["key-b"].last_status == "exhausted", (
        "the key that actually returned 403 was not marked exhausted"
    )
    assert by_id["key-a"].last_status is None, (
        f"healthy primary key-A was collaterally marked "
        f"{by_id['key-a'].last_status!r}/{by_id['key-a'].last_error_reason!r} — "
        "cross-key collateral marking bug"
    )


def test_hint_miss_does_not_fall_back_to_current_for_rotation(tmp_path, monkeypatch):
    """A non-empty api_key_hint that matches no pool entry is safer to reject
    than to fall back to current(), which may point at an innocent lease."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_two_key_pool(tmp_path)

    from agent.credential_pool import load_pool

    pool = load_pool("custom:gptcodex")
    pool.acquire_lease("key-a")
    current = pool.current()
    assert current is not None
    assert current.id == "key-a"

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=403,
        error_context={"code": "GROUP_DISABLED"},
        api_key_hint="sk-not-in-this-pool",
    )

    assert next_entry is None
    by_id = {e.id: e for e in pool.entries()}
    assert by_id["key-a"].last_status is None
    assert by_id["key-b"].last_status is None


def test_hint_miss_does_not_fall_back_to_current_for_refresh(tmp_path, monkeypatch):
    """Auth recovery must not force-refresh/mark current() when the failed
    request key is known and it is not this pool's key."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_two_key_pool(tmp_path)

    from agent.credential_pool import load_pool

    pool = load_pool("custom:gptcodex")
    pool.acquire_lease("key-a")
    current = pool.current()
    assert current is not None
    assert current.id == "key-a"

    refreshed = pool.try_refresh_current(api_key_hint="sk-not...pool")

    assert refreshed is None
    by_id = {e.id: e for e in pool.entries()}
    assert by_id["key-a"].last_status is None
    assert by_id["key-b"].last_status is None


def test_oauth_singleton_hint_miss_can_adopt_rotated_entry(tmp_path, monkeypatch):
    """A singleton OAuth token may rotate between client creation and 401
    recovery; an old-token hint should refresh/adopt that singleton lineage,
    not fail as an unknown unrelated key."""
    import json

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "id": "xai-loopback",
                            "label": "xai-loopback",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "loopback_pkce",
                            "access_token": "new-token",
                            "refresh_token": "new-refresh",
                            "base_url": "https://api.x.ai/v1",
                        }
                    ]
                },
            },
            indent=2,
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    refresh_calls = []

    def fake_refresh(access_token, refresh_token, **kwargs):
        refresh_calls.append((access_token, refresh_token))
        return {
            "access_token": "newer-token",
            "refresh_token": "newer-refresh",
            "last_refresh": "2026-06-24T00:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", fake_refresh)

    from agent.credential_pool import load_pool

    pool = load_pool("xai-oauth")
    refreshed = pool.try_refresh_current(api_key_hint="old-token")

    assert refreshed is not None
    assert refreshed.id == "xai-loopback"
    assert refreshed.access_token == "newer-token"
    assert refresh_calls == [("new-token", "new-refresh")]


def test_oauth_singleton_hint_miss_does_not_steal_mixed_manual_pool(tmp_path, monkeypatch):
    """Rotated-token adoption is only safe for singleton-only pools; in a
    mixed singleton+manual pool, an unknown manual hint must fail closed."""
    import json

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "id": "xai-loopback",
                            "label": "xai-loopback",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "loopback_pkce",
                            "access_token": "singleton-token",
                            "refresh_token": "singleton-refresh",
                            "base_url": "https://api.x.ai/v1",
                        },
                        {
                            "id": "xai-manual",
                            "label": "xai-manual",
                            "auth_type": "oauth",
                            "priority": 1,
                            "source": "manual:xai_pkce",
                            "access_token": "manual-new-token",
                            "refresh_token": "manual-new-refresh",
                            "base_url": "https://api.x.ai/v1",
                        },
                    ]
                },
            },
            indent=2,
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    refresh_calls = []

    def fake_refresh(access_token, refresh_token, **kwargs):
        refresh_calls.append((access_token, refresh_token))
        return {
            "access_token": "should-not-be-used",
            "refresh_token": "should-not-be-used",
            "last_refresh": "2026-06-24T00:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", fake_refresh)

    from agent.credential_pool import load_pool

    pool = load_pool("xai-oauth")
    refreshed = pool.try_refresh_current(api_key_hint="manual-old-token")

    assert refreshed is None
    assert refresh_calls == []
    by_id = {entry.id: entry for entry in pool.entries()}
    assert by_id["xai-loopback"].access_token == "singleton-token"
    assert by_id["xai-manual"].access_token == "manual-new-token"


def test_oauth_singleton_hint_miss_can_mark_rotated_quota_entry(tmp_path, monkeypatch):
    """For singleton-only OAuth pools, a 429/402 on an old access token still
    belongs to the rotated singleton lineage and should mark that entry."""
    import json

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "id": "xai-loopback",
                            "label": "xai-loopback",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "loopback_pkce",
                            "access_token": "new-token",
                            "refresh_token": "new-refresh",
                            "base_url": "https://api.x.ai/v1",
                        }
                    ]
                },
            },
            indent=2,
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.credential_pool import load_pool

    pool = load_pool("xai-oauth")
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"code": "rate_limit_exceeded"},
        api_key_hint="old-token",
    )

    assert next_entry is None
    entry = pool.entries()[0]
    assert entry.id == "xai-loopback"
    assert entry.last_status == "exhausted"
    assert entry.last_error_code == 429


def test_rate_limit_preexhausted_check_uses_failed_key_not_current_pointer(tmp_path, monkeypatch):
    """A different exhausted current() lease must not make the fresh failed
    key skip its first 429 retry and get marked exhausted immediately."""
    import json
    import time

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "custom:gptcodex": [
                        {
                            "id": "key-a",
                            "label": "already-exhausted",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual",
                            "access_token": HEALTHY_KEY,
                            "base_url": GPTCODEX_URL,
                            "last_status": "exhausted",
                            "last_status_at": time.time(),
                            "last_error_code": 429,
                        },
                        {
                            "id": "key-b",
                            "label": "fresh-failing-key",
                            "auth_type": "api_key",
                            "priority": 1,
                            "source": "manual",
                            "access_token": DISABLED_KEY,
                            "base_url": GPTCODEX_URL,
                        },
                    ]
                },
            },
            indent=2,
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.credential_pool import load_pool

    pool = load_pool("custom:gptcodex")
    pool.acquire_lease("key-a")
    agent = MagicMock()
    agent.provider = "custom"
    agent.base_url = GPTCODEX_URL
    agent._credential_pool = pool
    agent.client.api_key = DISABLED_KEY

    with patch(
        "agent.credential_pool.get_custom_provider_pool_key",
        return_value="custom:gptcodex",
    ):
        recovered, retried = recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.rate_limit,
            error_context={"code": "rate_limit_exceeded"},
        )

    assert (recovered, retried) == (False, True)
    by_id = {entry.id: entry for entry in pool.entries()}
    assert by_id["key-b"].last_status is None


def test_raw_nous_agent_key_hint_matches_even_when_runtime_key_is_unusable(tmp_path, monkeypatch):
    """Nous runtime_api_key hides expired/near-expired agent_key values; hint
    matching must still compare the raw agent_key used by the failed request."""
    import json

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "nous": [
                        {
                            "id": "nous-device",
                            "label": "nous-device",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "device_code",
                            "access_token": "portal-access-token",
                            "refresh_token": "portal-refresh-token",
                            "agent_key": "expired-agent-key",
                            "agent_key_expires_at": "2000-01-01T00:00:00+00:00",
                            "inference_base_url": "https://inference-api.nousresearch.com/v1",
                        }
                    ]
                },
            },
            indent=2,
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.credential_pool import load_pool

    pool = load_pool("nous")
    pool._refresh_entry = lambda entry, force=True: entry  # type: ignore[method-assign]

    refreshed = pool.try_refresh_current(api_key_hint="expired-agent-key")

    assert refreshed is not None
    assert refreshed.id == "nous-device"


def test_hinted_recovery_does_not_downgrade_dead_entry_to_exhausted(tmp_path, monkeypatch):
    """A stale request can race after another process marked the credential
    terminally dead; later 429/402 recovery must not give it a TTL cooldown."""
    import json

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "id": "xai-loopback",
                            "label": "xai-loopback",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "loopback_pkce",
                            "access_token": "new-token",
                            "refresh_token": "dead-refresh",
                            "base_url": "https://api.x.ai/v1",
                            "last_status": "dead",
                            "last_error_code": 401,
                            "last_error_reason": "invalid_grant",
                        }
                    ]
                },
            },
            indent=2,
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.credential_pool import load_pool

    pool = load_pool("xai-oauth")

    assert pool.mark_exhausted_and_rotate(status_code=429, api_key_hint="old-token") is None
    assert pool.try_refresh_current(api_key_hint="new-token") is None
    entry = pool.entries()[0]
    assert entry.last_status == "dead"
    assert entry.last_error_code == 401
    assert entry.last_error_reason == "invalid_grant"
