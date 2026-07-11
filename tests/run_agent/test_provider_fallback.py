"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.model_metadata import VerifiedContextLimit
from agent.subagent_governance import assert_governance_request_fits
from run_agent import AIAgent, _pool_may_recover_from_rate_limit


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
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
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None

    def test_single_dict_backwards_compat(self):
        fb = {"provider": "openai", "model": "gpt-4o"}
        agent = _make_agent(fallback_model=fb)
        assert agent._fallback_chain == [fb]
        assert agent._fallback_model == fb

    def test_list_of_providers(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 2
        assert agent._fallback_model == fbs[0]

    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"

    def test_empty_list(self):
        agent = _make_agent(fallback_model=[])
        assert agent._fallback_chain == []
        assert agent._fallback_model is None

    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


class TestFallbackChainAdvancement:
    def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"
            assert agent._fallback_activated is True

    def test_second_fallback_works(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "resolved")):
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._try_activate_fallback() is True
            assert agent.model == "glm-4.7"
            assert agent._fallback_index == 2

    def test_all_exhausted_returns_false(self):
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is False

    def test_skips_unconfigured_provider_to_next(self):
        """If resolve_provider_client returns None, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                (None, None),                    # broken provider
                (_mock_client(), "gpt-4o"),       # fallback succeeds
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2

    def test_skips_provider_that_raises_to_next(self):
        """If resolve_provider_client raises, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                RuntimeError("auth failed"),
                (_mock_client(), "gpt-4o"),
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"

    def test_resolves_key_env_for_fallback_provider(self):
        fbs = [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.example/v1",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(
                        base_url="https://fallback.example/v1",
                        api_key="env-secret",
                    ),
                    "fallback-model",
                ),
            ) as mock_rpc,
        ):
            assert agent._try_activate_fallback() is True
            assert mock_rpc.call_args.kwargs["explicit_api_key"] == "env-secret"

    def test_anthropic_host_custom_provider_uses_anthropic_messages(self):
        """A custom provider on the native api.anthropic.com host (no
        "/anthropic" path suffix, name != "anthropic") must resolve to the
        anthropic_messages wire protocol — not default to chat_completions,
        which POSTs /v1/chat/completions and 404s. Mirrors the primary-path
        determine_api_mode() host check."""
        fbs = [
            {
                "provider": "cron-anthropic",
                "model": "claude-sonnet-4-6",
                "base_url": "https://api.anthropic.com",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url="https://api.anthropic.com"),
                    "claude-sonnet-4-6",
                ),
            ),
            patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
        ):
            assert agent._try_activate_fallback() is True
            assert agent.api_mode == "anthropic_messages"

    def test_same_model_provider_fallback_does_not_inherit_primary_context_cap(self):
        """Fallback activation must resolve the fallback route's own context.

        ``model.context_length`` config belongs to the primary route that was
        initialized at startup.  When openai-codex/gpt-5.5 falls back to a custom
        gptcodex/gpt-5.5 proxy, carrying the primary 272K legacy cap makes
        append-cached summary preflight think a 1M-capable request overflows.
        """
        fbs = [{"provider": "gptcodex", "model": "gpt-5.5"}]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openai-codex"
        agent.model = "gpt-5.5"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent._config_context_length = 272_000

        seen_caps = []

        def _ctx(_model, **kwargs):
            seen_caps.append(kwargs.get("config_context_length"))
            cap = kwargs.get("config_context_length")
            return int(cap) if cap else 1_000_000

        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url="https://gptcodex.top/v1/"),
                    "gpt-5.5",
                ),
            ),
            patch("agent.model_metadata.get_model_context_length", side_effect=_ctx),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert seen_caps == [None]
        assert agent.context_compressor.context_length == 1_000_000

    def test_same_provider_fallback_keeps_primary_context_cap(self):
        """If fallback stays on the same backend, the primary cap still applies.

        Operators often use ``model.context_length`` to cap a local/custom
        endpoint whose catalog overreports the true window. Dropping that cap is
        only safe when the fallback route changes provider/base URL.
        """
        fbs = [{"provider": "custom-local", "model": "sibling-model"}]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "custom-local"
        agent.model = "primary-model"
        agent.base_url = "http://127.0.0.1:9999/v1"
        agent._config_context_length = 80_000

        seen_caps = []

        def _ctx(_model, **kwargs):
            seen_caps.append(kwargs.get("config_context_length"))
            cap = kwargs.get("config_context_length")
            return int(cap) if cap else 1_000_000

        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url="http://127.0.0.1:9999/v1"),
                    "sibling-model",
                ),
            ),
            patch("agent.model_metadata.get_model_context_length", side_effect=_ctx),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert seen_caps == [80_000]
        assert agent.context_compressor.context_length == 80_000


class _RateLimitError(Exception):
    status_code = 429
    body = {"error": {"message": "rate limited"}}


def test_governed_child_rechecks_smaller_fallback_before_backend():
    """Primary may call once; an oversized fallback must remain backend-zero."""
    fallback = [{
        "provider": "zai",
        "model": "glm-4.7",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "context_length": 2_200,
    }]
    agent = _make_agent(fallback_model=fallback)
    agent._cached_system_prompt = "complete-governance-canary"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_tokens = 100
    agent._api_max_retries = 1
    agent._governance_diagnostics = {"fingerprint": "governance-fingerprint"}
    agent.context_compressor.context_length = 200_000
    agent._governance_context_limit_proof = {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "limit": VerifiedContextLimit(200_000, "primary-test-proof"),
    }

    backend_attempts = []

    def _backend(_api_kwargs):
        backend_attempts.append((agent.provider, agent.model))
        if len(backend_attempts) == 1:
            raise _RateLimitError("rate limited")
        raise AssertionError("fallback backend must not run")

    fallback_client = _mock_client(
        base_url="https://open.bigmodel.cn/api/coding/paas/v4"
    )
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=_backend),
        patch.object(
            agent,
            "_interruptible_streaming_api_call",
            side_effect=lambda kwargs, **_extra: _backend(kwargs),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "glm-4.7"),
        ),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda model, provider: model,
        ),
        patch("agent.model_metadata.get_model_context_length", return_value=2_200),
        patch(
            "agent.subagent_governance.assert_governance_request_fits",
            wraps=assert_governance_request_fits,
        ) as preflight,
    ):
        result = agent.run_conversation("task-payload-canary")

    assert backend_attempts == [("", "")]
    assert preflight.call_count == 2
    assert result["error"] == "governance_context_too_large"
    assert result["api_calls"] == 1
    proof = agent._governance_context_limit_proof
    assert proof["model"] == "glm-4.7"
    assert proof["provider"] == "zai"
    assert proof["base_url"] == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert proof["api_mode"] == agent.api_mode
    assert proof["limit"] == VerifiedContextLimit(2_200, "config")


def test_governed_child_relay_fallback_with_explicit_context_reaches_backend():
    fallback_url = "https://private-relay.example.test/v1"
    fallback = [{
        "provider": "custom-relay",
        "model": "private-relay-model",
        "base_url": fallback_url,
        "context_length": 80_000,
    }]
    agent = _make_agent(fallback_model=fallback)
    agent._cached_system_prompt = "complete-governance-canary"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_tokens = 100
    agent._governance_diagnostics = {"fingerprint": "governance-fingerprint"}
    backend_routes = []

    def _response():
        message = SimpleNamespace(
            content="relay-ok", tool_calls=None, reasoning_content=None
        )
        return SimpleNamespace(
            id="relay-response",
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="private-relay-model",
            usage=None,
        )

    def _backend(_kwargs):
        backend_routes.append((agent.provider, agent.model, agent.base_url))
        return _response()

    fallback_client = _mock_client(base_url=fallback_url)
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=_backend),
        patch.object(
            agent,
            "_interruptible_streaming_api_call",
            side_effect=lambda kwargs, **_extra: _backend(kwargs),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "private-relay-model"),
        ),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda model, provider: model,
        ),
        patch("agent.model_metadata.get_model_context_length", return_value=80_000),
    ):
        result = agent.run_conversation("task-payload-canary")

    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert backend_routes == [
        ("custom-relay", "private-relay-model", fallback_url)
    ]
    assert agent._governance_context_limit_proof == {
        "model": "private-relay-model",
        "provider": "custom-relay",
        "base_url": fallback_url,
        "api_mode": agent.api_mode,
        "limit": VerifiedContextLimit(80_000, "config"),
    }


# ── Pool-rotation vs fallback gating (#11314) ────────────────────────────


def _pool(n_entries: int, has_available: bool = True):
    """Make a minimal credential-pool stand-in for rotation-room checks."""
    pool = MagicMock()
    pool.entries.return_value = [MagicMock() for _ in range(n_entries)]
    pool.has_available.return_value = has_available
    return pool


class TestPoolRotationRoom:
    def test_none_pool_returns_false(self):
        assert _pool_may_recover_from_rate_limit(None) is False

    def test_single_credential_returns_false(self):
        """With one credential that just 429'd, rotation has nowhere to go.

        The pool may still report has_available() True once cooldown expires,
        but retrying against the same entry will hit the same daily-quota
        429 and burn the retry budget.  Must fall back.
        """
        assert _pool_may_recover_from_rate_limit(_pool(1)) is False

    def test_single_credential_in_cooldown_returns_false(self):
        assert _pool_may_recover_from_rate_limit(_pool(1, has_available=False)) is False

    def test_two_credentials_available_returns_true(self):
        """With >1 credentials and at least one available, rotate instead of fallback."""
        assert _pool_may_recover_from_rate_limit(_pool(2)) is True

    def test_multiple_credentials_all_in_cooldown_returns_false(self):
        """All credentials cooling down — fall back rather than wait."""
        assert _pool_may_recover_from_rate_limit(_pool(3, has_available=False)) is False

    def test_many_credentials_available_returns_true(self):
        assert _pool_may_recover_from_rate_limit(_pool(10)) is True


# ── Skip-self dedup (#22548) ───────────────────────────────────────────────


class TestFallbackChainDedup:
    """A fallback chain entry that resolves to the current provider/model
    (or the same custom-provider base_url) must be skipped, not retried.
    Otherwise a misconfigured chain or two custom_providers entries pointing
    at the same shim loop the same failure. See issue #22548."""

    def test_skips_entry_matching_current_provider_and_model(self):
        """Chain has [same-as-current, real-fallback]; activate must skip
        the first and use the second."""
        fbs = [
            # First entry == current state. Should be skipped.
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
            # Second entry: real fallback.
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        # Stub out resolve_provider_client so we can assert which entry was
        # actually used — return a MagicMock client tagged with the provider.
        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # The first entry was skipped — only the second reached resolve.
        assert called == [("zai", "glm-4.7")], (
            f"expected fallback to skip same-state entry, got call order: {called}"
        )

    def test_skips_entry_matching_current_base_url_and_model(self):
        """Two custom_providers entries pointing at the same shim URL
        with the same model should dedup even if their provider names differ."""
        fbs = [
            # Different provider name but same shim URL + model — same backend.
            {"provider": "claude-cli-alt", "model": "claude-opus-4.7",
             "base_url": "http://127.0.0.1:7891/v1"},
            # Real different fallback.
            {"provider": "openrouter", "model": "anthropic/claude-opus-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "claude-cli"
        agent.model = "claude-opus-4.7"
        agent.base_url = "http://127.0.0.1:7891/v1"

        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # Same shim/base_url+model entry skipped, second one used.
        assert called == [("openrouter", "anthropic/claude-opus-4.7")], (
            f"expected base_url-aware dedup, got call order: {called}"
        )

    def test_returns_false_when_only_self_matching_entries(self):
        """A chain with only self-matching entries exhausts to False."""
        fbs = [
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        mock_resolve.assert_not_called()
