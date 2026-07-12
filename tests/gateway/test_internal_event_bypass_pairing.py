"""Tests that internal synthetic events (e.g. background process completion)
bypass user authorization and do not trigger DM pairing.

Regression test for the bug where ``_run_process_watcher`` with
``notify_on_complete=True`` injected a ``MessageEvent`` without ``user_id``,
causing ``_is_user_authorized`` to reject it and the gateway to send a
pairing code to the chat.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools.process_registry import ProcessRegistry, ProcessSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRegistry:
    """Return pre-canned sessions, then None once exhausted."""

    def __init__(self, sessions):
        self._sessions = list(sessions)
        self._completion_consumed: set = set()

    def get(self, session_id):
        if self._sessions:
            return self._sessions.pop(0)
        return None

    def is_completion_consumed(self, session_id):
        return session_id in self._completion_consumed

    def completion_notification_decision(self, session_id):
        return "suppress" if self.is_completion_consumed(session_id) else "notify"


def _build_runner(monkeypatch, tmp_path) -> GatewayRunner:
    """Create a GatewayRunner with notifications set to 'all'."""
    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: all\n",
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter
    return runner


def _watcher_dict_with_notify():
    return {
        "session_id": "proc_test_internal",
        "check_interval": 0,
        "session_key": "agent:main:discord:dm:123",
        "platform": "discord",
        "chat_id": "123",
        "thread_id": "",
        "notify_on_complete": True,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notify_on_complete_sets_internal_flag(monkeypatch, tmp_path):
    """Synthetic completion event must have internal=True."""
    import tools.process_registry as pr_module

    sessions = [
        SimpleNamespace(
            output_buffer="done\n", exited=True, exit_code=0, command="echo test"
        ),
    ]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.DISCORD]

    await runner._run_process_watcher(_watcher_dict_with_notify())

    assert adapter.handle_message.await_count == 1
    assert getattr(adapter.send, "await_count") == 0
    event = adapter.handle_message.await_args.args[0]
    assert isinstance(event, MessageEvent)
    assert event.internal is True, "Synthetic completion event must be marked internal"
    assert event.metadata["hermes_process_completion_id"] == "proc_test_internal"


@pytest.mark.asyncio
async def test_poll_suppresses_duplicate_notify_on_complete_watcher(monkeypatch, tmp_path):
    """Polling an exited process consumes the completion for late watcher delivery."""
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_polled_completion",
        command="echo done",
        output_buffer="done\n",
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._finished[session.id] = session

    poll_result = registry.poll(session.id)
    assert poll_result["status"] == "exited"
    assert registry.is_completion_consumed(session.id)

    monkeypatch.setattr(pr_module, "process_registry", registry)

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.DISCORD]

    watcher = _watcher_dict_with_notify()
    watcher["session_id"] = session.id

    await runner._run_process_watcher(watcher)

    assert getattr(adapter.handle_message, "await_count") == 0
    assert getattr(adapter.send, "await_count") == 0


@pytest.mark.parametrize("termination_source", ["process.kill", "kill_all"])
@pytest.mark.asyncio
async def test_expected_process_kill_does_not_inject_gateway_completion(
    monkeypatch, tmp_path, termination_source
):
    """An agent-requested kill remains inspectable but must not wake a new turn."""
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_expected_kill",
        command="sleep 60",
        output_buffer="",
        exited=True,
        exit_code=-15,
        completion_reason="killed",
        termination_source=termination_source,
        notify_on_complete=True,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.DISCORD]
    watcher = _watcher_dict_with_notify()
    watcher["session_id"] = session.id

    await runner._run_process_watcher(watcher)

    assert registry.get(session.id) is session
    assert session.completion_reason == "killed"
    assert session.termination_source == termination_source
    assert getattr(adapter.handle_message, "await_count") == 0
    assert getattr(adapter.send, "await_count") == 0


@pytest.mark.asyncio
async def test_gateway_watcher_defers_until_kill_outcome_is_known(monkeypatch, tmp_path):
    """Reader-observed exit during kill intent must wait for the signal result."""
    import tools.process_registry as pr_module

    session = SimpleNamespace(
        output_buffer="",
        exited=True,
        exit_code=-15,
        command="sleep 60",
        completion_reason="exited",
        termination_source="",
    )

    class _RaceRegistry:
        def __init__(self):
            self.decisions = ["defer", "suppress"]
            self.seen = []

        def get(self, _session_id):
            return session

        def completion_notification_decision(self, session_id):
            decision = self.decisions.pop(0)
            self.seen.append((session_id, decision))
            return decision

    registry = _RaceRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.DISCORD]

    await runner._run_process_watcher(_watcher_dict_with_notify())

    assert registry.seen == [
        ("proc_test_internal", "defer"),
        ("proc_test_internal", "suppress"),
    ]
    assert getattr(adapter.handle_message, "await_count") == 0
    assert getattr(adapter.send, "await_count") == 0


@pytest.mark.asyncio
async def test_text_watcher_rechecks_policy_immediately_before_send(monkeypatch, tmp_path):
    """A terminal state consumed during rendering must not leak via text fallback."""
    import tools.process_registry as pr_module

    session = SimpleNamespace(
        output_buffer="done\n",
        exited=True,
        exit_code=0,
        command="echo done",
        completion_reason="exited",
        termination_source="",
    )

    class _ConsumeRaceRegistry:
        def __init__(self):
            self.decisions = ["notify", "suppress"]

        def get(self, _session_id):
            return session

        def completion_notification_decision(self, _session_id):
            return self.decisions.pop(0)

    registry = _ConsumeRaceRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.DISCORD]
    watcher = _watcher_dict_with_notify()
    watcher["notify_on_complete"] = False

    await runner._run_process_watcher(watcher)

    assert registry.decisions == []
    assert getattr(adapter.handle_message, "await_count") == 0
    assert getattr(adapter.send, "await_count") == 0


@pytest.mark.asyncio
async def test_queued_completion_is_dropped_when_consumed_before_dispatch(monkeypatch, tmp_path):
    """A watcher-queued completion must not become a stale follow-up turn.

    Reproduces the live race: the watcher creates the synthetic event while the
    origin session is busy, then process.poll observes the exited handle before
    the queued event is dispatched after that turn.
    """
    import tools.process_registry as pr_module

    session = SimpleNamespace(
        output_buffer="done\n",
        exited=True,
        exit_code=0,
        command="echo test",
    )
    registry = _FakeRegistry([session])
    monkeypatch.setattr(pr_module, "process_registry", registry)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.DISCORD]

    watcher = _watcher_dict_with_notify()
    await runner._run_process_watcher(watcher)
    queued_event = getattr(adapter.handle_message, "await_args").args[0]

    registry._completion_consumed.add(watcher["session_id"])
    runner._handle_message_with_agent = AsyncMock(return_value="stale")

    result = await runner._handle_message(queued_event)

    assert result is None
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_event_bypasses_authorization(monkeypatch, tmp_path):
    """An internal event should skip _is_user_authorized entirely."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = GatewayRunner(GatewayConfig())

    # Create an internal event with no user_id (simulates the bug scenario)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="dm",
    )
    event = MessageEvent(
        text="[SYSTEM: Background process completed]",
        source=source,
        internal=True,
    )

    # Track if _is_user_authorized is called
    auth_called = False
    original_auth = GatewayRunner._is_user_authorized

    def tracking_auth(self, src):
        nonlocal auth_called
        auth_called = True
        return original_auth(self, src)

    monkeypatch.setattr(GatewayRunner, "_is_user_authorized", tracking_auth)

    # Stop execution before the agent runner so the test doesn't block in
    # run_in_executor.  Auth check happens before _handle_message_with_agent.
    async def _raise(*_a, **_kw):
        raise RuntimeError("sentinel — stop here")
    monkeypatch.setattr(GatewayRunner, "_handle_message_with_agent", _raise)

    try:
        await runner._handle_message(event)
    except RuntimeError:
        pass  # Expected sentinel

    assert not auth_called, (
        "_is_user_authorized should NOT be called for internal events"
    )


@pytest.mark.asyncio
async def test_internal_event_does_not_trigger_pairing(monkeypatch, tmp_path):
    """An internal event with no user_id must not generate a pairing code."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = GatewayRunner(GatewayConfig())
    # Add adapter so pairing would have somewhere to send
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="dm",  # DM would normally trigger pairing
    )
    event = MessageEvent(
        text="[SYSTEM: Background process completed]",
        source=source,
        internal=True,
    )

    # Track pairing code generation
    generate_called = False
    original_generate = runner.pairing_store.generate_code

    def tracking_generate(*args, **kwargs):
        nonlocal generate_called
        generate_called = True
        return original_generate(*args, **kwargs)

    runner.pairing_store.generate_code = tracking_generate

    # Stop execution before the agent runner so the test doesn't block in
    # run_in_executor.  Pairing check happens before _handle_message_with_agent.
    async def _raise(*_a, **_kw):
        raise RuntimeError("sentinel — stop here")
    monkeypatch.setattr(GatewayRunner, "_handle_message_with_agent", _raise)

    try:
        await runner._handle_message(event)
    except RuntimeError:
        pass  # Expected sentinel

    assert not generate_called, (
        "Pairing code should NOT be generated for internal events"
    )


@pytest.mark.asyncio
async def test_notify_on_complete_preserves_user_identity(monkeypatch, tmp_path):
    """Synthetic completion event should carry user_id and user_name from the watcher."""
    import tools.process_registry as pr_module

    sessions = [
        SimpleNamespace(
            output_buffer="done\n", exited=True, exit_code=0, command="echo test"
        ),
    ]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.DISCORD]

    watcher = _watcher_dict_with_notify()
    watcher["user_id"] = "user-42"
    watcher["user_name"] = "alice"

    await runner._run_process_watcher(watcher)

    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.source.user_id == "user-42"
    assert event.source.user_name == "alice"


@pytest.mark.asyncio
async def test_notify_on_complete_uses_session_store_origin_for_group_topic(monkeypatch, tmp_path):
    import tools.process_registry as pr_module
    from gateway.session import SessionSource

    sessions = [
        SimpleNamespace(
            output_buffer="done\n", exited=True, exit_code=0, command="echo test"
        ),
    ]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="user-42",
            user_name="alice",
        )
    )

    watcher = {
        "session_id": "proc_test_internal",
        "check_interval": 0,
        "session_key": "agent:main:telegram:group:-100:42",
        "platform": "telegram",
        "chat_id": "-100",
        "thread_id": "42",
        "notify_on_complete": True,
    }

    await runner._run_process_watcher(watcher)

    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.source.platform == Platform.TELEGRAM
    assert event.source.chat_id == "-100"
    assert event.source.chat_type == "group"
    assert event.source.thread_id == "42"
    assert event.source.user_id == "user-42"
    assert event.source.user_name == "alice"


@pytest.mark.asyncio
async def test_none_user_id_skips_pairing(monkeypatch, tmp_path):
    """A non-internal event with user_id=None should be silently dropped."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id=None,
    )
    event = MessageEvent(
        text="service message",
        source=source,
        internal=False,
    )

    result = await runner._handle_message(event)

    # Should return None (dropped) and NOT send any pairing message
    assert result is None
    assert adapter.send.await_count == 0


@pytest.mark.asyncio
async def test_none_user_id_does_not_generate_pairing_code(monkeypatch, tmp_path):
    """A message with user_id=None must never call generate_code."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter

    generate_called = False
    original_generate = runner.pairing_store.generate_code

    def tracking_generate(*args, **kwargs):
        nonlocal generate_called
        generate_called = True
        return original_generate(*args, **kwargs)

    runner.pairing_store.generate_code = tracking_generate

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="456",
        chat_type="dm",
        user_id=None,
    )
    event = MessageEvent(text="anonymous", source=source, internal=False)

    await runner._handle_message(event)

    assert not generate_called, (
        "Pairing code should NOT be generated for messages with user_id=None"
    )


@pytest.mark.asyncio
async def test_non_internal_event_without_user_triggers_pairing(monkeypatch, tmp_path):
    """Verify the normal (non-internal) path still triggers pairing for unknown users."""
    import gateway.run as gateway_run
    import gateway.pairing as pairing_mod

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    # gateway.pairing.PAIRING_DIR is a module-level constant captured at
    # import time from whichever HERMES_HOME was set then. Per-test
    # HERMES_HOME redirection in conftest doesn't retroactively move it.
    # Override directly so pairing rate-limit state lives in this test's
    # tmp_path (and so stale state from prior xdist workers can't leak in).
    pairing_dir = tmp_path / "pairing"
    pairing_dir.mkdir()
    monkeypatch.setattr(pairing_mod, "PAIRING_DIR", pairing_dir)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    # Clear env vars that could let all users through (loaded by
    # module-level dotenv in gateway/run.py from the real ~/.hermes/.env).
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("DISCORD_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="dm",
        user_id="unknown_user_999",
    )
    # Normal event (not internal)
    event = MessageEvent(
        text="hello",
        source=source,
        internal=False,
    )

    result = await runner._handle_message(event)

    # Should return None (unauthorized) and send pairing message
    assert result is None
    assert adapter.send.await_count == 1
    sent_text = adapter.send.await_args.args[1]
    assert "don't recognize you" in sent_text
