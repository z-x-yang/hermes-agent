"""Regression tests for service-managed gateway /restart exit behavior."""

import pytest


@pytest.mark.asyncio
async def test_service_restart_uses_hard_exit_before_asyncio_runner_cleanup(monkeypatch):
    """A planned service restart must not be hostage to lingering asyncio tasks.

    Discord /restart runs inside the gateway process. After the gateway has
    already drained, disconnected adapters, released locks, and chosen the
    service-restart exit code, returning through ``asyncio.run`` can still hang
    while Python cancels stray background tasks. The service manager cannot
    relaunch until this process actually exits, so the restart path must use the
    gateway's hard-exit hook at that point.
    """
    import cron.scheduler_provider as scheduler_provider
    import gateway.run as gateway_run
    import gateway.status as gateway_status
    import hermes_cli.nous_auth_keepalive as keepalive
    import tools.mcp_tool as mcp_tool
    import tools.skills_sync as skills_sync
    from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE

    class HardExitCalled(RuntimeError):
        pass

    hard_exits: list[tuple[int, str]] = []

    def fake_hard_exit(code: int, reason: str) -> None:
        hard_exits.append((code, reason))
        raise HardExitCalled

    class FakeRunner:
        def __init__(self, config):
            self.adapters = {}
            self.should_exit_cleanly = False
            self.should_exit_with_failure = False
            self.exit_reason = "Gateway restart requested"
            self.exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE
            self._restart_requested = True
            self._restart_via_service = True
            # 0.18's start_gateway checks runner._running after start() to
            # tell a normal run from a startup aborted by restart/shutdown.
            self._running = True

        async def start(self):
            return True

        async def wait_for_shutdown(self):
            return None

    class FakeCronProvider:
        def start(self, stop_event, **kwargs):
            stop_event.wait(0.01)

        def stop(self):
            return None

    # Keep this test focused on the post-shutdown exit decision, not gateway
    # startup side effects.
    monkeypatch.setattr(gateway_run, "GatewayRunner", FakeRunner)
    monkeypatch.setattr(gateway_run, "_hard_exit_process", fake_hard_exit, raising=False)
    monkeypatch.setattr(gateway_run, "_run_planned_stop_watcher", lambda *a, **k: None)
    monkeypatch.setattr(gateway_run, "_start_gateway_housekeeping", lambda stop_event, **kwargs: stop_event.wait(0.01))
    monkeypatch.setattr(gateway_run, "_ensure_windows_gateway_venv_imports", lambda: None)
    monkeypatch.setattr(gateway_run.threading, "main_thread", lambda: object())
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda: None)
    monkeypatch.setattr(gateway_status, "acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr(gateway_status, "write_pid_file", lambda: None)
    monkeypatch.setattr(gateway_status, "remove_pid_file", lambda: None)
    monkeypatch.setattr(gateway_status, "release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr(skills_sync, "sync_skills", lambda quiet=True: None)
    monkeypatch.setattr(keepalive, "start_nous_auth_keepalive", lambda: None)
    monkeypatch.setattr(keepalive, "stop_nous_auth_keepalive", lambda: None)
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", lambda: None)
    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr(scheduler_provider, "resolve_cron_scheduler", lambda: FakeCronProvider())

    with pytest.raises(HardExitCalled):
        await gateway_run.start_gateway(verbosity=None)

    assert hard_exits == [
        (GATEWAY_SERVICE_RESTART_EXIT_CODE, "service-managed gateway restart")
    ]
