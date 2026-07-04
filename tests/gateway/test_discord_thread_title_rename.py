"""Tests for propagating Hermes session titles to Discord thread names."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    adapter = SimpleNamespace(rename_thread=AsyncMock())
    runner.adapters = {Platform.DISCORD: adapter}
    runner._session_db = None
    return runner, adapter


class _TitledTrackingAdapter:
    """Adapter stub whose per-thread auto-title memory behaves like the real one."""

    def __init__(self):
        self._titled: set[str] = set()
        self.rename_thread = AsyncMock(return_value=SimpleNamespace(success=True, error=None))

    def has_auto_titled_thread(self, thread_id: str) -> bool:
        return str(thread_id) in self._titled

    def mark_auto_titled_thread(self, thread_id: str) -> None:
        self._titled.add(str(thread_id))


def _make_runner_with_tracking_adapter():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    adapter = _TitledTrackingAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._session_db = None
    return runner, adapter


def _make_source(*, thread_id: str | None = "1517731349685993587") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="1480933001889321195",
        parent_chat_id="1480933001889321195" if thread_id else None,
        chat_type="thread" if thread_id else "channel",
        user_id="222",
        user_name="tester",
        thread_id=thread_id,
        guild_id="1473189377902379092",
    )


@pytest.mark.asyncio
async def test_rename_discord_thread_for_session_title_uses_adapter():
    runner, adapter = _make_runner()

    await runner._rename_discord_thread_for_session_title(
        _make_source(),
        "sess-discord",
        "  Auto   Session   Title  ",
    )

    adapter.rename_thread.assert_awaited_once_with(
        thread_id="1517731349685993587",
        name="Auto Session Title",
    )


@pytest.mark.asyncio
async def test_rename_discord_thread_for_session_title_ignores_non_threads():
    runner, adapter = _make_runner()

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id=None),
        "sess-discord",
        "Channel Title",
    )

    adapter.rename_thread.assert_not_awaited()


def test_sanitize_discord_thread_title_truncates_to_discord_limit():
    runner, _adapter = _make_runner()

    title = runner._sanitize_discord_thread_title("  " + "A" * 130 + "  ")

    assert len(title) == 100
    assert title.endswith("...")


@pytest.mark.asyncio
async def test_auto_rename_fires_at_most_once_per_thread():
    """A long-lived thread must not be re-titled when a fresh, title-less
    session is minted for it (e.g. after a restart's ``agent_close``).

    Auto-titling is keyed to the session, so each new session's title guard is
    blind to the fact the thread was named already. The gateway must skip the
    second automatic rename for the same thread.
    """
    runner, adapter = _make_runner_with_tracking_adapter()
    source = _make_source(thread_id="1517731349685993587")

    # First session for the thread → renames once and remembers the thread.
    runner._schedule_discord_thread_title_rename(source, "sess-1", "First Title")
    await asyncio.sleep(0.05)
    assert adapter.rename_thread.await_count == 1
    assert adapter.has_auto_titled_thread("1517731349685993587")

    # A later, title-less session for the SAME thread must not rename again.
    runner._schedule_discord_thread_title_rename(source, "sess-2", "Second Title")
    await asyncio.sleep(0.05)
    assert adapter.rename_thread.await_count == 1


@pytest.mark.asyncio
async def test_auto_rename_still_fires_for_distinct_threads():
    """The per-thread one-shot must not bleed across different threads."""
    runner, adapter = _make_runner_with_tracking_adapter()

    runner._schedule_discord_thread_title_rename(
        _make_source(thread_id="1111"), "sess-a", "Title A"
    )
    runner._schedule_discord_thread_title_rename(
        _make_source(thread_id="2222"), "sess-b", "Title B"
    )
    await asyncio.sleep(0.05)

    assert adapter.rename_thread.await_count == 2
    assert adapter.has_auto_titled_thread("1111")
    assert adapter.has_auto_titled_thread("2222")


@pytest.mark.asyncio
async def test_manual_rename_bypasses_per_thread_guard():
    """A manual ``/title`` rename must always fire, even when the thread was
    already auto-titled — the at-most-once guard only governs AUTOMATIC titling.

    Regression: the 0.18 port routed ``/title`` through this guarded scheduler,
    so re-titling an already-named thread on demand was silently dropped.
    """
    runner, adapter = _make_runner_with_tracking_adapter()
    source = _make_source(thread_id="1517731349685993587")

    # Thread gets its one automatic title.
    runner._schedule_discord_thread_title_rename(source, "sess-1", "Auto Title")
    await asyncio.sleep(0.05)
    assert adapter.rename_thread.await_count == 1

    # A manual /title rename must still go through despite the guard.
    runner._schedule_discord_thread_title_rename(
        source, "sess-1", "User Chosen Title", manual=True
    )
    await asyncio.sleep(0.05)
    assert adapter.rename_thread.await_count == 2
    adapter.rename_thread.assert_awaited_with(
        thread_id="1517731349685993587", name="User Chosen Title"
    )


@pytest.mark.asyncio
async def test_manual_rename_marks_thread_so_later_auto_is_suppressed():
    """A manual rename on a fresh thread still records it as titled, so a later
    title-less session's AUTOMATIC rename does not override the user's name."""
    runner, adapter = _make_runner_with_tracking_adapter()
    source = _make_source(thread_id="9999")

    runner._schedule_discord_thread_title_rename(
        source, "sess-1", "User Chosen", manual=True
    )
    await asyncio.sleep(0.05)
    assert adapter.rename_thread.await_count == 1
    assert adapter.has_auto_titled_thread("9999")

    # Later automatic rename for the same thread must be suppressed.
    runner._schedule_discord_thread_title_rename(source, "sess-2", "Auto Would-Override")
    await asyncio.sleep(0.05)
    assert adapter.rename_thread.await_count == 1
