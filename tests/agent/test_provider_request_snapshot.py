from __future__ import annotations

from agent.chat_completion_helpers import build_provider_request_snapshot
from agent.compression_summary_runtime import cache_visible_request_payload


def test_cache_visible_request_payload_keeps_prompt_shape_only():
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "demo"}}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "timeout": 600,
        "max_tokens": 4_096,
        "temperature": 0.3,
    }

    visible = cache_visible_request_payload(payload)

    assert visible == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "demo"}}],
    }


def test_cache_visible_request_payload_strips_credentials_but_keeps_cache_scope():
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "api_key": "top-level-secret",
        "access_token": "access-secret",
        "reasoning": {"effort": "high"},
        "reasoning_effort": "high",
        "store": True,
        "verbosity": "high",
        "extra_headers": {
            "Authorization": "Bearer auth-secret",
            "X-API-Key": "header-secret",
            "Cookie": "session=secret",
            "session_id": "cache-scope-a",
            "x-cache-tag": "stable-prefix",
        },
    }

    visible = cache_visible_request_payload(payload)

    assert visible == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "extra_headers": {
            "session_id": "cache-scope-a",
            "x-cache-tag": "stable-prefix",
        },
    }
    assert payload["extra_headers"]["Authorization"] == "Bearer auth-secret"


def test_provider_request_snapshot_is_deep_copied_and_credential_sensitive():
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "max_tokens": 4_096,
    }
    lineage = ("custom:demo", "gpt-test", "chat_completions", "https://api.test/v1", "key-one")

    first = build_provider_request_snapshot(
        payload,
        source_message_count=1,
        lineage=lineage,
    )
    second = build_provider_request_snapshot(
        payload,
        source_message_count=1,
        lineage=(*lineage[:-1], "key-two"),
    )
    payload["messages"][0]["content"] = "mutated"

    assert first.provider_request["messages"][0]["content"] == "hello"
    assert "key-one" not in repr(first)
    assert first.semantic_tokens > 0
    assert first.cache_fingerprint != second.cache_fingerprint
    assert first.calibration_fingerprint != second.calibration_fingerprint
