"""Unit tests for the extracted turn prologue (``agent/turn_context.py``).

These exercise ``build_turn_context`` against a lightweight fake agent to
confirm the prologue produces the right ``TurnContext`` and applies the
``agent`` side effects the loop relies on — without spinning up a real
``AIAgent`` or hitting any provider.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.turn_context import TurnContext, build_turn_context
from hermes_state import SessionDB


class _FakeTodoStore:
    def has_items(self):
        return True

    def _hydrate(self, *_a, **_k):
        pass


class _FakeGuardrails:
    def __init__(self):
        self.reset_called = False

    def reset_for_turn(self):
        self.reset_called = True


class _FakeAgent:
    """Minimal stand-in covering only what the prologue touches."""

    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cli"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._skip_mcp_refresh = False
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2, protect_last_n=2
        )
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        # Attributes the prologue assigns; recorded for assertions.
        self._invalid_tool_retries = -1
        self._vision_supported = None
        self._persist_calls = 0
        # Records _cached_system_prompt at the moment _ensure_db_session()
        # is called (regression guard for #45499 turn-setup ordering).
        self._ensure_db_prompt_at_call = "<unset>"

    # --- methods the prologue calls ---
    def _ensure_db_session(self):
        self._ensure_db_prompt_at_call = self._cached_system_prompt

    def _restore_primary_runtime(self) -> bool:
        return False

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, *_a, **_k):
        self._persist_calls += 1


def _make_agent_with_cooldown(db_path, session_id, *, cooldown_until=None):
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent._emit_status = MagicMock()
    agent._compress_context = MagicMock(
        side_effect=lambda messages, *_a, **_k: (messages, "SYSTEM")
    )

    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source="cli")
    if cooldown_until is not None:
        db.record_compression_failure_cooldown(session_id, cooldown_until, "timeout")

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.bind_session_state(db, session_id)
    agent.context_compressor = compressor
    agent._session_db = db
    return agent


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    """``build_turn_context`` calls ``auxiliary_client.set_runtime_main`` as a
    production side effect (telling aux tools the live main provider/model).
    That writes a module-level global these unit tests don't care about and
    which would otherwise leak into sibling tests (e.g. provider-parity
    resolution) when the per-test process isolation plugin is disabled. Stub
    it out so the prologue tests stay hermetic.
    """
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def _build(agent, **overrides):
    kwargs = dict(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


def test_returns_turn_context_with_user_message_appended():
    agent = _FakeAgent()
    ctx = _build(agent)
    assert isinstance(ctx, TurnContext)
    assert ctx.user_message == "hello"
    # The user turn was appended and indexed.
    assert ctx.messages[-1] == {"role": "user", "content": "hello"}
    assert ctx.current_turn_user_idx == len(ctx.messages) - 1
    assert ctx.active_system_prompt == "SYSTEM"


def test_applies_agent_side_effects():
    agent = _FakeAgent()
    _build(agent)
    # Retry counters reset, guardrails reset, vision re-armed, turn counted.
    assert agent._invalid_tool_retries == 0
    assert agent._tool_guardrails.reset_called is True
    assert agent._vision_supported is True
    assert agent._user_turn_count == 1
    # Crash-resilience persistence fired once.
    assert agent._persist_calls == 1
    # task/turn ids assigned on the agent.
    assert agent._current_task_id
    assert agent._current_turn_id


def test_task_id_passthrough():
    agent = _FakeAgent()
    ctx = _build(agent, task_id="fixed-task")
    assert ctx.effective_task_id == "fixed-task"
    assert agent._current_task_id == "fixed-task"


def test_persist_user_message_becomes_original():
    agent = _FakeAgent()
    ctx = _build(agent, user_message="api-prefixed", persist_user_message="clean")
    # original_user_message tracks the clean persist override.
    assert ctx.original_user_message == "clean"
    # but the appended user turn carries the full (sanitized) message.
    assert ctx.messages[-1]["content"] == "api-prefixed"


def test_memory_nudge_fires_at_interval():
    agent = _FakeAgent()
    agent._memory_nudge_interval = 1
    agent.valid_tool_names = {"memory"}
    agent._memory_store = object()
    ctx = _build(agent)
    assert ctx.should_review_memory is True
    assert agent._turns_since_memory == 0  # reset after firing


def test_no_review_when_memory_disabled():
    agent = _FakeAgent()
    ctx = _build(agent)
    assert ctx.should_review_memory is False


def test_ensure_db_session_runs_after_system_prompt_restore():
    """Regression for #45499.

    On a fresh API/gateway agent (``_cached_system_prompt is None``) the DB
    session row must be created AFTER the system prompt is restored/built, so
    the persisted snapshot is written non-NULL. If ``_ensure_db_session()``
    ran first it would insert ``system_prompt=NULL`` and trip the misleading
    "stored system prompt is null; rebuilding" warning plus a first-turn
    prefix cache miss.
    """
    agent = _FakeAgent()
    agent._cached_system_prompt = None  # fresh agent, no cached prompt yet

    def _restore(_agent, _system_message, _history):
        _agent._cached_system_prompt = "REBUILT-SYSTEM"

    _build(agent, restore_or_build_system_prompt=_restore)

    # The prompt was populated before the DB row was created.
    assert agent._ensure_db_prompt_at_call == "REBUILT-SYSTEM"
    assert agent._cached_system_prompt == "REBUILT-SYSTEM"


def test_runtime_main_sync_happens_after_primary_restore():
    """Auxiliary routing must not capture a stale fallback runtime.

    Gateway reuses an ``AIAgent`` across messages. If a prior turn activated
    fallback, the next turn's prologue restores the primary runtime before any
    compression/title-generation auxiliary calls should resolve "main".  The
    process-local ``set_runtime_main`` hook feeds those aux calls, so it must
    observe the post-restore primary values, not the stale fallback values.
    """
    agent = _FakeAgent()
    agent.provider = "openrouter"
    agent.model = "anthropic/claude-sonnet-4"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "fallback-key"
    agent.api_mode = "chat_completions"

    def _restore_primary_runtime():
        agent.provider = "openai-codex"
        agent.model = "gpt-5.5"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent.api_key = "primary-token"
        agent.api_mode = "codex_responses"
        return True

    agent._restore_primary_runtime = _restore_primary_runtime
    observed = []

    def _record_runtime(provider, model, **kwargs):
        observed.append(
            {
                "provider": provider,
                "model": model,
                "base_url": kwargs.get("base_url"),
                "api_key": kwargs.get("api_key"),
                "api_mode": kwargs.get("api_mode"),
            }
        )

    with patch("agent.auxiliary_client.set_runtime_main", side_effect=_record_runtime):
        _build(agent)

    assert observed == [
        {
            "provider": "openai-codex",
            "model": "gpt-5.5",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "primary-token",
            "api_mode": "codex_responses",
        }
    ]


# ── Between-turns MCP refresh (cache-safe late-binding) ──────────────────────
#
# A slow MCP server that connects after the agent's build-time tool snapshot
# must become callable by the user's NEXT turn — without mutating an in-flight
# turn's cached request prefix. The prologue is exactly that boundary, so the
# refresh hook lives here. These assert the contract (R1/R2/R6 in the spec),
# not timing permutations.


def test_between_turns_refresh_adds_late_tool_when_servers_registered():
    """R1: a tool that registered since build lands in this turn's snapshot."""
    agent = _FakeAgent()

    new_def = {"type": "function", "function": {"name": "mcp_x_tool", "description": "", "parameters": {}}}

    import model_tools
    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(model_tools, "get_tool_definitions", return_value=[new_def]):
        _build(agent)

    assert "mcp_x_tool" in agent.valid_tool_names
    assert any(t["function"]["name"] == "mcp_x_tool" for t in agent.tools)


def test_between_turns_refresh_skipped_when_no_servers():
    """R6: the common case (no MCP servers) never walks the registry."""
    agent = _FakeAgent()
    import model_tools

    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=False), \
         patch.object(model_tools, "get_tool_definitions") as gtd:
        _build(agent)

    gtd.assert_not_called()


def test_between_turns_refresh_skipped_when_skip_flag_set():
    """Internal forks (background_review) set _skip_mcp_refresh to keep tools[]
    byte-identical to the parent for cache parity — the hook must honor it even
    when MCP servers are registered."""
    agent = _FakeAgent()
    agent._skip_mcp_refresh = True
    import model_tools

    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(model_tools, "get_tool_definitions") as gtd:
        _build(agent)

    gtd.assert_not_called()


def test_between_turns_refresh_no_churn_when_unchanged():
    """R2: an unchanged tool set leaves the snapshot object identity intact
    (no needless swap → nothing for the next request prefix to diff against)."""
    agent = _FakeAgent()
    same = [{"type": "function", "function": {"name": "a", "description": "", "parameters": {}}}]
    agent.tools = same
    agent.valid_tool_names = {"a"}

    import model_tools
    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(
             model_tools, "get_tool_definitions",
             return_value=[{"type": "function", "function": {"name": "a", "description": "", "parameters": {}}}],
         ):
        _build(agent)

    assert agent.tools is same  # not replaced → no churn


def test_near_compression_preflight_gate_uses_near_threshold_when_enabled():
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent._runtime_context_status_mode = "inject"
    agent._runtime_context_status_audit_enabled = False
    agent._runtime_context_status_near_threshold_ratio = 0.90
    agent._pending_runtime_context_statuses = []
    agent._queued_runtime_context_status_keys = set()
    agent._last_context_pressure_notice_compression_count = -1
    compressor = types.SimpleNamespace(
        protect_first_n=10,
        protect_last_n=10,
        threshold_tokens=1000,
        context_length=2000,
        compression_count=0,
        last_prompt_tokens=0,
        last_real_prompt_tokens=0,
        get_active_compression_failure_cooldown=lambda: None,
        should_compress=lambda tokens: tokens >= 1000,
    )
    agent.context_compressor = compressor
    agent._compress_context = MagicMock()

    with patch("agent.turn_context.estimate_messages_tokens_rough", return_value=950), \
         patch("agent.turn_context.estimate_request_tokens_rough", return_value=950):
        _build(agent)

    agent._compress_context.assert_not_called()
    assert len(agent._pending_runtime_context_statuses) == 1
    assert agent._pending_runtime_context_statuses[0]["kind"] == "pre_near_compression"


def test_near_compression_queues_runtime_context_status_once():
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent._runtime_context_status_mode = "inject"
    agent._runtime_context_status_audit_enabled = False
    agent._pending_runtime_context_statuses = []
    agent._queued_runtime_context_status_keys = set()
    agent._last_context_pressure_notice_compression_count = -1
    compressor = types.SimpleNamespace(
        protect_first_n=0,
        protect_last_n=0,
        threshold_tokens=1000,
        context_length=2000,
        compression_count=0,
        last_prompt_tokens=0,
        last_real_prompt_tokens=0,
        get_active_compression_failure_cooldown=lambda: None,
        should_compress=lambda tokens: tokens >= 1000,
    )
    agent.context_compressor = compressor
    agent._compress_context = MagicMock()

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch("agent.turn_context.estimate_request_tokens_rough", return_value=950):
        _build(agent)
        _build(agent)

    agent._compress_context.assert_not_called()
    assert len(agent._pending_runtime_context_statuses) == 1
    pending = agent._pending_runtime_context_statuses[0]
    assert pending["kind"] == "pre_near_compression"
    assert "The visible conversation is close" in pending["content"]
    assert agent._last_context_pressure_notice_compression_count == 0


def test_near_compression_notice_not_queued_when_compression_triggers():
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent._runtime_context_status_mode = "inject"
    agent._runtime_context_status_audit_enabled = False
    agent._pending_runtime_context_statuses = []
    agent._queued_runtime_context_status_keys = set()
    compressor = types.SimpleNamespace(
        protect_first_n=0,
        protect_last_n=0,
        threshold_tokens=1000,
        context_length=2000,
        compression_count=0,
        last_prompt_tokens=0,
        last_real_prompt_tokens=0,
        get_active_compression_failure_cooldown=lambda: None,
        should_compress=lambda tokens: tokens >= 1000,
    )
    agent.context_compressor = compressor
    agent._compress_context = MagicMock(side_effect=lambda messages, *_a, **_k: (messages, "SYSTEM"))

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch("agent.turn_context.estimate_request_tokens_rough", return_value=1000):
        _build(agent)

    assert agent._pending_runtime_context_statuses == []
    agent._compress_context.assert_called()


def test_preflight_skips_when_persisted_cooldown_survives_restart(tmp_path):
    agent = _make_agent_with_cooldown(
        tmp_path / "state.db",
        "sess-1",
        cooldown_until=4_000_000_000.0,
    )

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch.object(agent.context_compressor, "estimate_provider_request_tokens", return_value=999_999):
        ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    agent._emit_status.assert_not_called()
    agent._compress_context.assert_not_called()


def test_preflight_still_runs_for_other_session_with_same_db(tmp_path):
    db_path = tmp_path / "state.db"
    _make_agent_with_cooldown(
        db_path,
        "sess-1",
        cooldown_until=4_000_000_000.0,
    )
    agent = _make_agent_with_cooldown(db_path, "sess-2")

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch.object(agent.context_compressor, "estimate_provider_request_tokens", return_value=999_999):
        ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    agent._emit_status.assert_called_once()
    agent._compress_context.assert_called()


def test_expired_cooldown_allows_preflight(tmp_path):
    agent = _make_agent_with_cooldown(
        tmp_path / "state.db",
        "sess-1",
        cooldown_until=1.0,
    )

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch.object(agent.context_compressor, "estimate_provider_request_tokens", return_value=999_999):
        ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    agent._emit_status.assert_called_once()
    agent._compress_context.assert_called()

