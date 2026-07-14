from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.provider_attempt import execute_provider_attempt, prepare_provider_attempt


def _agent():
    return SimpleNamespace(
        session_id="child-session",
        platform="subagent",
        model="test-model",
        provider="test-provider",
        base_url="https://example.invalid/v1",
        api_mode="chat_completions",
        max_tokens=1024,
        context_compressor=SimpleNamespace(context_length=272_000),
        _governance_diagnostics={
            "fingerprint": "legacy-fingerprint",
            "profile_id": "default",
            "total_bytes": 10,
        },
    )


def test_large_child_payload_uses_shared_request_path_without_byte_governance_gate(monkeypatch):
    agent = _agent()
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "x" * 838_000},
        ],
        "max_tokens": 1024,
    }
    prepared = prepare_provider_attempt(
        agent,
        payload,
        task_id="task",
        turn_id="turn",
        api_request_id="request",
        api_call_count=31,
    )
    backend = MagicMock(return_value="accepted")
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        lambda request, next_call, **_kwargs: next_call(request),
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_verified_model_context_length",
        lambda *args, **kwargs: SimpleNamespace(tokens=272_000, source="test"),
    )

    result = execute_provider_attempt(
        agent,
        prepared,
        backend,
        task_id="task",
        turn_id="turn",
        api_request_id="request",
        api_call_count=31,
    )

    assert result == "accepted"
    backend.assert_called_once()


def test_request_middleware_failure_does_not_create_governance_transport_error(monkeypatch):
    agent = _agent()
    payload = {"model": "test-model", "messages": [{"role": "user", "content": "safe"}]}

    def fail_middleware(*_args, **_kwargs):
        raise RuntimeError("middleware unavailable")

    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware",
        fail_middleware,
    )
    prepared = prepare_provider_attempt(
        agent,
        payload,
        task_id="task",
        turn_id="turn",
        api_request_id="request",
        api_call_count=1,
    )
    assert prepared.payload == payload
    assert prepared.original_payload == payload
    assert prepared.middleware_trace == ()
