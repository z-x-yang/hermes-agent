"""Governed-child provider request-fit preflight tests (H11)."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

from agent.model_metadata import (
    DEFAULT_FALLBACK_CONTEXT,
    VerifiedContextLimit,
    get_model_context_length,
    get_verified_model_context_length,
)
from agent.subagent_governance import (
    GovernancePreflightError,
    GovernanceRequestFit,
    assert_governance_request_fits,
)


def _agent(*, governed: bool = True, context_length: int = 20_000):
    diagnostics = {"fingerprint": "governance-fingerprint"} if governed else None
    return SimpleNamespace(
        _governance_diagnostics=diagnostics,
        context_compressor=SimpleNamespace(context_length=context_length),
        model="test-model",
        provider="test-provider",
        api_mode="chat_completions",
        base_url="https://example.test/v1",
        api_key="secret-not-request-body",
        max_tokens=321,
        _config_context_length=None,
        _custom_providers=None,
    )


def _verified(tokens: int = 20_000) -> VerifiedContextLimit:
    return VerifiedContextLimit(tokens=tokens, source="test-authority")


def test_final_payload_serialization_includes_messages_system_and_exact_tool_schema():
    agent = _agent()
    tool_schema = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "schema-canary-工具",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "const": "exact-canary"}},
            },
        },
    }
    api_kwargs = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "governance-canary"},
            {"role": "user", "content": "task-and-continuation-canary"},
        ],
        "tools": [tool_schema],
        "temperature": 0,
        "max_tokens": 321,
    }
    expected = json.dumps(
        api_kwargs, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    with patch(
        "agent.model_metadata.get_verified_model_context_length",
        return_value=_verified(),
    ):
        fit = assert_governance_request_fits(agent, api_kwargs)

    assert fit == GovernanceRequestFit(
        serialized_utf8_bytes=len(expected),
        input_token_upper_bound=len(expected) + 2_048,
        output_reserve_tokens=321,
        context_limit_tokens=20_000,
    )
    diagnostics = agent._governance_request_fit_diagnostics
    assert diagnostics["request_fingerprint"] == hashlib.sha256(expected).hexdigest()
    assert diagnostics["governance_fingerprint"] == "governance-fingerprint"
    assert "governance-canary" not in repr(diagnostics)
    assert "exact-canary" not in repr(diagnostics)


def test_codex_responses_without_output_cap_uses_verified_dynamic_remainder():
    agent = _agent(context_length=20_000)
    agent.api_mode = "codex_responses"
    agent.max_tokens = None
    api_kwargs = {
        "model": "test-model",
        "input": [{"role": "user", "content": "live-default-path"}],
    }
    expected = json.dumps(
        api_kwargs, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    input_upper_bound = len(expected) + 2_048

    with patch(
        "agent.model_metadata.get_verified_model_context_length",
        return_value=_verified(),
    ):
        fit = assert_governance_request_fits(agent, api_kwargs)

    assert fit == GovernanceRequestFit(
        serialized_utf8_bytes=len(expected),
        input_token_upper_bound=input_upper_bound,
        output_reserve_tokens=20_000 - input_upper_bound,
        context_limit_tokens=20_000,
    )
    assert (
        agent._governance_request_fit_diagnostics["output_reserve_source"]
        == "provider_dynamic_remainder"
    )


def test_unknown_context_limit_default_is_not_proof_for_governed_child():
    agent = _agent(context_length=256_000)
    governance_before = dict(agent._governance_diagnostics)
    api_kwargs = {
        "model": "unknown-model",
        "messages": [{"role": "system", "content": "complete-governance"}],
        "max_tokens": 100,
    }

    with patch(
        "agent.model_metadata.get_verified_model_context_length",
        return_value=None,
    ):
        with pytest.raises(GovernancePreflightError) as exc_info:
            assert_governance_request_fits(agent, api_kwargs)

    assert exc_info.value.code == "governance_transport_unverifiable"
    assert agent._governance_diagnostics == governance_before


def test_output_reserve_participates_in_context_fit_without_mutating_governance():
    agent = _agent(context_length=2_200)
    governance_before = dict(agent._governance_diagnostics)
    api_kwargs = {
        "model": "test-model",
        "messages": [{"role": "system", "content": "complete-governance"}],
        "max_output_tokens": 500,
    }

    with patch(
        "agent.model_metadata.get_verified_model_context_length",
        return_value=_verified(2_200),
    ):
        with pytest.raises(GovernancePreflightError) as exc_info:
            assert_governance_request_fits(agent, api_kwargs)

    assert exc_info.value.code == "governance_context_too_large"
    assert agent._governance_diagnostics == governance_before


def test_unserializable_provider_visible_value_fails_closed_without_default_str():
    agent = _agent()
    api_kwargs = {
        "model": "test-model",
        "messages": [{"role": "system", "content": "complete-governance"}],
        "max_tokens": 100,
        "provider_extension": object(),
    }

    with patch(
        "agent.model_metadata.get_verified_model_context_length",
        return_value=_verified(),
    ):
        with pytest.raises(GovernancePreflightError) as exc_info:
            assert_governance_request_fits(agent, api_kwargs)

    assert exc_info.value.code == "governance_transport_unverifiable"


def test_verified_context_resolver_accepts_explicit_attempt_limit():
    assert get_verified_model_context_length(
        "private-model",
        provider="custom",
        config_context_length=77_777,
    ) == VerifiedContextLimit(tokens=77_777, source="config")


def test_verified_context_resolver_rejects_generic_default_as_proof(monkeypatch):
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *args, **kwargs: DEFAULT_FALLBACK_CONTEXT,
    )
    assert get_verified_model_context_length(
        "totally-unknown-private-model",
        provider="custom",
    ) is None


def test_custom_relay_known_model_slug_does_not_gain_catalog_proof(monkeypatch):
    monkeypatch.setattr(
        "agent.model_metadata._resolve_model_context_length",
        lambda *args, **kwargs: 1_050_000,
    )
    route = {
        "model": "gpt-5.5-private-relay",
        "provider": "custom",
        "base_url": "https://relay.example.test/v1",
    }

    assert get_model_context_length(**route) == 1_050_000
    assert get_verified_model_context_length(**route) is None


def test_custom_unknown_nondefault_resolution_is_not_route_proof(monkeypatch):
    monkeypatch.setattr(
        "agent.model_metadata._resolve_model_context_length",
        lambda *args, **kwargs: 73_123,
    )
    route = {
        "model": "private-unknown-model",
        "provider": "custom",
        "base_url": "https://relay-two.example.test/v1",
    }

    assert get_model_context_length(**route) == 73_123
    assert get_verified_model_context_length(**route) is None


def test_custom_provider_per_model_config_remains_verified(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.get_custom_provider_context_length",
        lambda **_kwargs: 88_000,
    )
    route = {
        "model": "private-model",
        "provider": "custom",
        "base_url": "https://configured-relay.example.test/v1",
        "custom_providers": [{"name": "configured-relay"}],
    }

    assert get_model_context_length(**route) == 88_000
    assert get_verified_model_context_length(**route) == VerifiedContextLimit(
        88_000, "custom_provider_config"
    )


def test_verified_proof_cache_does_not_cross_base_url_or_provider():
    route = {
        "model": "exact-private-model",
        "provider": "custom-a",
        "base_url": "https://route-a.example.test/v1",
    }
    get_model_context_length(**route, config_context_length=91_000)

    assert get_verified_model_context_length(**route) == VerifiedContextLimit(
        91_000, "config"
    )
    assert get_verified_model_context_length(
        route["model"],
        provider="custom-a",
        base_url="https://route-b.example.test/v1",
    ) is None
    assert get_verified_model_context_length(
        route["model"],
        provider="custom-b",
        base_url=route["base_url"],
    ) is None


def test_cached_agent_route_proof_must_match_base_url():
    agent = _agent()
    agent._governance_context_limit_proof = {
        "model": agent.model,
        "provider": agent.provider,
        "api_mode": agent.api_mode,
        "base_url": "https://different-route.example.test/v1",
        "limit": _verified(),
    }
    api_kwargs = {
        "model": agent.model,
        "messages": [{"role": "user", "content": "route-bound"}],
        "max_tokens": 100,
    }

    with patch(
        "agent.model_metadata.get_verified_model_context_length",
        return_value=None,
    ):
        with pytest.raises(GovernancePreflightError) as exc_info:
            assert_governance_request_fits(agent, api_kwargs)

    assert exc_info.value.code == "governance_transport_unverifiable"


def _runtime_agent(*, fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "complete-governance-canary"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._governance_diagnostics = {"fingerprint": "governance-fingerprint"}
    return agent


def _stop_response(text: str):
    message = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(
        id="fake-response",
        choices=[choice],
        model="fake-model",
        usage=None,
    )


def test_primary_denial_occurs_before_backend_hook_and_request_dump(monkeypatch):
    agent = _runtime_agent()
    backend = agent.client.chat.completions.create
    backend.return_value = _stop_response("must-not-run")
    hook_calls = []
    hook = MagicMock(side_effect=lambda name, **kwargs: hook_calls.append(name) or [])
    request_dump = MagicMock()
    governance_before = dict(agent._governance_diagnostics)

    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "pre_api_request")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)
    monkeypatch.setattr(agent, "_dump_api_request_debug", request_dump)
    monkeypatch.setattr(
        "agent.model_metadata.get_verified_model_context_length",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(agent, "_persist_session", MagicMock())
    monkeypatch.setattr(agent, "_save_trajectory", MagicMock())
    monkeypatch.setattr(agent, "_cleanup_task_resources", MagicMock())

    result = agent.run_conversation("task-payload-canary")

    assert result["failed"] is True
    assert result["error"] == "governance_transport_unverifiable"
    assert result["api_calls"] == 0
    backend.assert_not_called()
    assert "pre_api_request" not in hook_calls
    request_dump.assert_not_called()
    assert agent._governance_diagnostics == governance_before


def test_governed_request_middleware_failure_denies_before_all_sinks(monkeypatch):
    agent = _runtime_agent()
    backend = MagicMock(return_value=_stop_response("must-not-run"))
    execution = MagicMock()
    hook_calls = []
    hook = MagicMock(side_effect=lambda name, **_kwargs: hook_calls.append(name) or [])
    request_dump = MagicMock()
    secret_canary = "request-governance-body-canary"

    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")
    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware",
        MagicMock(side_effect=RuntimeError(secret_canary)),
    )
    monkeypatch.setattr("hermes_cli.middleware.run_llm_execution_middleware", execution)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)
    monkeypatch.setattr(agent, "_dump_api_request_debug", request_dump)
    monkeypatch.setattr(agent, "_interruptible_api_call", backend)

    result = agent.run_conversation(secret_canary)

    assert result["error"] == "governance_transport_unverifiable"
    assert result["api_calls"] == 0
    backend.assert_not_called()
    execution.assert_not_called()
    assert "pre_api_request" not in hook_calls
    request_dump.assert_not_called()
    assert secret_canary not in result["error"]
    assert secret_canary not in repr(
        getattr(agent, "_governance_request_fit_diagnostics", None)
    )


def test_non_governed_request_middleware_failure_keeps_legacy_backend_path(monkeypatch):
    agent = _runtime_agent()
    agent._governance_diagnostics = None
    backend = MagicMock(return_value=_stop_response("legacy-ok"))
    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware",
        MagicMock(side_effect=RuntimeError("legacy middleware failure")),
    )
    monkeypatch.setattr(agent, "_interruptible_api_call", backend)
    monkeypatch.setattr(
        agent,
        "_interruptible_streaming_api_call",
        lambda kwargs, **_extra: backend(kwargs),
    )

    result = agent.run_conversation("legacy-parent")

    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert backend.call_count == 1


def test_governed_middleware_denial_then_fallback_success_keeps_call_charge(
    monkeypatch,
):
    from hermes_cli.middleware import apply_llm_request_middleware as real_apply

    agent = _runtime_agent()
    refunds = _spy_iteration_refunds(monkeypatch, agent)
    fallback_activations = 0

    def _activate_once():
        nonlocal fallback_activations
        if fallback_activations:
            return False
        fallback_activations += 1
        return _activate_test_route(agent, "fallback")

    monkeypatch.setattr(
        agent,
        "_try_activate_fallback",
        MagicMock(side_effect=_activate_once),
    )

    middleware_providers = []

    def _middleware(api_kwargs, **kwargs):
        middleware_providers.append(agent.provider)
        if agent.provider != "fallback":
            raise RuntimeError("private-body-canary")
        return real_apply(api_kwargs, **kwargs)

    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware", _middleware
    )
    monkeypatch.setattr(
        "agent.subagent_governance.assert_governance_request_fits",
        lambda _agent, _kwargs: None,
    )
    backend = MagicMock(return_value=_stop_response("fallback-ok"))
    monkeypatch.setattr(agent, "_interruptible_api_call", backend)
    monkeypatch.setattr(
        agent,
        "_interruptible_streaming_api_call",
        lambda kwargs, **_extra: backend(kwargs),
    )

    result = agent.run_conversation("task-payload-canary")

    assert result["completed"] is True, (result, middleware_providers)
    assert result["api_calls"] == 1
    assert backend.call_count == 1
    assert refunds == []


def test_execution_middleware_rewrite_is_rechecked_before_observer_and_backend(
    monkeypatch,
):
    from agent.provider_attempt import (
        execute_provider_attempt,
        prepare_provider_attempt,
    )

    agent = _runtime_agent()
    original = {
        "model": getattr(agent, "model", ""),
        "messages": [{"role": "user", "content": "safe"}],
        "max_tokens": 20,
    }
    prepared = prepare_provider_attempt(
        agent,
        original,
        task_id="task",
        turn_id="turn",
        api_request_id="request",
        api_call_count=1,
    )
    rewritten = dict(prepared.payload)
    rewritten["messages"] = [
        {"role": "user", "content": "execution-middleware-canary"}
    ]
    backend = MagicMock(return_value=_stop_response("must-not-run"))
    observer = MagicMock()
    fit_payloads = []

    def _fit(_agent, payload):
        fit_payloads.append(payload)
        if payload is rewritten:
            raise GovernancePreflightError("governance_transport_unverifiable")

    monkeypatch.setattr(
        "agent.subagent_governance.assert_governance_request_fits",
        _fit,
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        lambda _request, next_call, **_kwargs: next_call(rewritten),
    )

    with pytest.raises(GovernancePreflightError):
        execute_provider_attempt(
            agent,
            prepared,
            backend,
            task_id="task",
            turn_id="turn",
            api_request_id="request",
            api_call_count=1,
            pre_api_observer=observer,
        )

    assert fit_payloads == [rewritten]
    observer.assert_not_called()
    backend.assert_not_called()


def test_max_iteration_summary_strips_tools_added_by_request_middleware(monkeypatch):
    agent = _runtime_agent()
    backend = MagicMock(return_value=_stop_response("tool-less-summary"))

    def _inject_tools(api_kwargs, **_kwargs):
        payload = dict(api_kwargs)
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "must be stripped",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = True
        return SimpleNamespace(
            payload=payload,
            original_payload=dict(api_kwargs),
            trace=[],
        )

    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware",
        _inject_tools,
    )
    monkeypatch.setattr(
        "agent.subagent_governance.assert_governance_request_fits",
        lambda _agent, _payload: None,
    )
    monkeypatch.setattr(agent, "_interruptible_api_call", backend)

    result = agent._handle_max_iterations(
        [{"role": "user", "content": "summarize without tools"}],
        60,
    )

    assert result == "tool-less-summary"
    sent = backend.call_args.args[0]
    assert "tools" not in sent
    assert "tool_choice" not in sent
    assert "parallel_tool_calls" not in sent


def test_max_iteration_summary_rechecks_governance_on_fallback_before_any_sink(
    monkeypatch,
):
    agent = _runtime_agent()
    primary_direct_backend = agent.client.chat.completions.create
    primary_direct_backend.return_value = _stop_response("primary-leak")
    fallback_backend = MagicMock(return_value=_stop_response("fallback-summary"))
    hook_providers = []
    request_dump = MagicMock()

    monkeypatch.setattr(
        agent,
        "_try_activate_fallback",
        MagicMock(side_effect=lambda: _activate_test_route(agent, "fallback")),
    )

    def _preflight(_agent, _kwargs):
        if agent.provider != "fallback":
            raise GovernancePreflightError("governance_transport_unverifiable")

    monkeypatch.setattr(
        "agent.subagent_governance.assert_governance_request_fits",
        _preflight,
    )
    monkeypatch.setattr(agent, "_interruptible_api_call", fallback_backend)
    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda name: name == "pre_api_request",
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **_kwargs: hook_providers.append(agent.provider) or [],
    )
    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")
    monkeypatch.setattr(agent, "_dump_api_request_debug", request_dump)

    result = agent._handle_max_iterations(
        [{"role": "user", "content": "summarize governed work"}],
        60,
    )

    assert result == "fallback-summary"
    primary_direct_backend.assert_not_called()
    assert fallback_backend.call_count == 1
    assert agent.session_api_calls == 1
    assert getattr(agent, "_last_max_iteration_summary_api_calls", 0) == 1
    assert hook_providers == ["fallback"]
    assert request_dump.call_count == 1


def _activate_test_route(agent, provider: str):
    agent.provider = provider
    agent.model = f"{provider}-model"
    agent.base_url = f"https://{provider}.example.test/v1"
    return True


def _spy_iteration_refunds(monkeypatch, agent):
    calls = []
    budget_type = type(agent.iteration_budget)
    original_refund = budget_type.refund

    def _refund(budget):
        calls.append(budget)
        return original_refund(budget)

    monkeypatch.setattr(budget_type, "refund", _refund)
    return calls


def test_primary_denied_then_fallback_succeeds_keeps_one_logical_call(monkeypatch):
    agent = _runtime_agent()
    refunds = _spy_iteration_refunds(monkeypatch, agent)
    monkeypatch.setattr(
        agent,
        "_try_activate_fallback",
        MagicMock(side_effect=lambda: _activate_test_route(agent, "fallback")),
    )

    def _preflight(_agent, _kwargs):
        if agent.provider != "fallback":
            raise GovernancePreflightError("governance_transport_unverifiable")

    monkeypatch.setattr(
        "agent.subagent_governance.assert_governance_request_fits", _preflight
    )
    backend = MagicMock(return_value=_stop_response("fallback-ok"))
    monkeypatch.setattr(agent, "_interruptible_api_call", backend)
    monkeypatch.setattr(
        agent,
        "_interruptible_streaming_api_call",
        lambda kwargs, **_extra: backend(kwargs),
    )

    result = agent.run_conversation("task-payload-canary")

    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert backend.call_count == 1
    assert refunds == []


def test_all_denied_provider_chain_refunds_logical_call_exactly_once(monkeypatch):
    agent = _runtime_agent()
    refunds = _spy_iteration_refunds(monkeypatch, agent)
    fallback_routes = iter(("fallback-one", "fallback-two"))

    def _next_fallback():
        try:
            return _activate_test_route(agent, next(fallback_routes))
        except StopIteration:
            return False

    monkeypatch.setattr(agent, "_try_activate_fallback", MagicMock(side_effect=_next_fallback))
    monkeypatch.setattr(
        "agent.subagent_governance.assert_governance_request_fits",
        MagicMock(
            side_effect=GovernancePreflightError(
                "governance_transport_unverifiable"
            )
        ),
    )
    backend = MagicMock(return_value=_stop_response("must-not-run"))
    monkeypatch.setattr(agent, "_interruptible_api_call", backend)

    result = agent.run_conversation("task-payload-canary")

    assert result["failed"] is True
    assert result["api_calls"] == 0
    backend.assert_not_called()
    assert len(refunds) == 1


def test_primary_backend_ran_then_fallback_denied_preserves_logical_call(monkeypatch):
    agent = _runtime_agent()
    agent._api_max_retries = 1
    refunds = _spy_iteration_refunds(monkeypatch, agent)
    fallback_activated = False

    def _next_fallback():
        nonlocal fallback_activated
        if fallback_activated:
            return False
        fallback_activated = True
        return _activate_test_route(agent, "fallback")

    monkeypatch.setattr(agent, "_try_activate_fallback", MagicMock(side_effect=_next_fallback))

    def _preflight(_agent, _kwargs):
        if agent.provider == "fallback":
            raise GovernancePreflightError("governance_transport_unverifiable")

    monkeypatch.setattr(
        "agent.subagent_governance.assert_governance_request_fits", _preflight
    )
    backend = MagicMock(
        side_effect=GovernancePreflightError("governance_transport_unverifiable")
    )
    monkeypatch.setattr(agent, "_interruptible_api_call", backend)
    monkeypatch.setattr(
        agent,
        "_interruptible_streaming_api_call",
        lambda kwargs, **_extra: backend(kwargs),
    )

    result = agent.run_conversation("task-payload-canary")

    assert result["api_calls"] == 1
    assert backend.call_count == 1
    assert refunds == []


def test_non_governed_agent_preserves_legacy_unknown_model_behavior():
    agent = _agent(governed=False)
    api_kwargs = {
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "legacy-parent"}],
    }

    with patch(
        "agent.model_metadata.get_verified_model_context_length"
    ) as resolver:
        assert assert_governance_request_fits(agent, api_kwargs) is None

    resolver.assert_not_called()
