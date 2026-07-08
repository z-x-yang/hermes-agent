from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent import runtime_context_status as rcs


class _Agent(SimpleNamespace):
    session_id: str = "sess-1"
    _current_turn_id: str = "turn-1"


def _agent(mode: str = "inject", *, audit: bool = True):
    return _Agent(
        session_id="sess-1",
        _current_turn_id="turn-1",
        _runtime_context_status_mode=mode,
        _runtime_context_status_audit_enabled=audit,
        _pending_runtime_context_statuses=[],
        _queued_runtime_context_status_keys=set(),
    )


@pytest.fixture()
def audit_home(tmp_path, monkeypatch):
    monkeypatch.setattr(rcs, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def _audit_records(home):
    path = home / "logs" / "runtime_context_status_audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_notices_use_natural_language_not_machine_fields():
    pre = rcs.build_pre_compression_notice()
    post = rcs.build_post_compression_notice()
    for text in (pre, post):
        assert text.startswith("<hermes-runtime-context>")
        assert text.endswith("</hermes-runtime-context>")
        assert "temporary Hermes runtime metadata" in text
        assert "NOT new user input" in text
        assert "status_type:" not in text
        assert "scope:" not in text
        assert "saved_to_conversation:" not in text
    assert "No compression has happened yet" in pre
    assert "immediately before this model call" in post


def test_mode_off_does_not_queue_and_audits_disabled(audit_home):
    agent = _agent("off")

    rcs.queue_runtime_context_status(
        agent,
        rcs.build_pre_compression_notice(),
        kind="pre_near_compression",
        dedupe_key="near:0",
        metadata={"rough_tokens": 90, "threshold_tokens": 100},
    )

    assert agent._pending_runtime_context_statuses == []
    records = _audit_records(audit_home)
    assert records[-1]["event"] == "queue"
    assert records[-1]["result"] == "disabled"
    assert records[-1]["mode"] == "off"
    assert "content" not in records[-1]


def test_queue_dedupes_by_key_and_consume_clears(audit_home):
    agent = _agent("inject")

    for _ in range(2):
        rcs.queue_runtime_context_status(
            agent,
            rcs.build_post_compression_notice(),
            kind="post_compression_completed",
            dedupe_key="post:1",
            metadata={"compression_count": 1},
        )

    assert len(agent._pending_runtime_context_statuses) == 1
    pending = rcs.consume_runtime_context_statuses(agent)
    assert len(pending) == 1
    assert pending[0]["kind"] == "post_compression_completed"
    assert rcs.consume_runtime_context_statuses(agent) == []
    assert agent._queued_runtime_context_status_keys == set()


def test_mode_shadow_does_not_mutate_api_messages_but_audits_shadow_logged(audit_home):
    agent = _agent("shadow")
    pending = [{
        "kind": "pre_near_compression",
        "content": rcs.build_pre_compression_notice(),
        "dedupe_key": "near:0",
        "metadata": {},
    }]
    api_messages = [{"role": "user", "content": "hello"}]

    assert rcs.inject_runtime_context_statuses(api_messages, pending, agent=agent, turn_id="turn-1") is False

    assert api_messages == [{"role": "user", "content": "hello"}]
    records = _audit_records(audit_home)
    assert records[-1]["event"] == "inject"
    assert records[-1]["result"] == "shadow_logged"
    assert records[-1]["target_role"] == "user"
    assert records[-1]["target_message_index_from_end"] == 0


def test_mode_inject_appends_to_last_user_message_only(audit_home):
    agent = _agent("inject")
    status = rcs.build_post_compression_notice()
    pending = [{
        "kind": "post_compression_completed",
        "content": status,
        "dedupe_key": "post:2",
        "metadata": {"compression_count": 2},
    }]
    api_messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "older user"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "latest user"},
    ]

    assert rcs.inject_runtime_context_statuses(api_messages, pending, agent=agent, turn_id="turn-1") is True

    assert api_messages[1]["content"] == "older user"
    assert api_messages[-1]["content"].startswith("latest user\n\n<hermes-runtime-context>")
    assert "immediately before this model call" in api_messages[-1]["content"]
    records = _audit_records(audit_home)
    assert records[-1]["result"] == "injected"
    assert records[-1]["status_kind"] == "post_compression_completed"
    assert records[-1]["status_chars"] == len(status)
    assert records[-1]["compression_count"] == 2


def test_inject_fails_closed_without_string_user_content_and_audits_reason(audit_home):
    agent = _agent("inject")
    api_messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": [{"type": "text", "text": "not string"}]},
    ]
    original = [m.copy() for m in api_messages]
    pending = [{
        "kind": "pre_near_compression",
        "content": rcs.build_pre_compression_notice(),
        "dedupe_key": "near:1",
        "metadata": {},
    }]

    assert rcs.inject_runtime_context_statuses(api_messages, pending, agent=agent, turn_id="turn-1") is False

    assert api_messages == original
    records = _audit_records(audit_home)
    assert records[-1]["event"] == "drop"
    assert records[-1]["result"] == "dropped_no_string_user_message"


def test_audit_records_are_content_free(audit_home):
    agent = _agent("inject")
    secret_user_text = "sensitive user text should never be logged"
    status = rcs.build_pre_compression_notice()
    pending = [{
        "kind": "pre_near_compression",
        "content": status,
        "dedupe_key": "near:secret-session-id",
        "metadata": {"rough_tokens": 91, "threshold_tokens": 100},
    }]
    api_messages = [{"role": "user", "content": secret_user_text}]

    rcs.inject_runtime_context_statuses(api_messages, pending, agent=agent, turn_id="turn-1")

    raw = (audit_home / "logs" / "runtime_context_status_audit.jsonl").read_text(encoding="utf-8")
    assert secret_user_text not in raw
    assert "The visible conversation is close" not in raw
    assert "near:secret-session-id" not in raw
    assert "rough_tokens" in raw
