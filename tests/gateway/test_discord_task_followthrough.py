from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter = object.__new__(DiscordAdapter)
    adapter._threads = MagicMock()
    adapter._dispatch_thread_session = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_dispatch_task_followthrough_marks_thread_and_reuses_thread_session_dispatch():
    adapter = _make_adapter()
    interaction = SimpleNamespace(user=SimpleNamespace(id=42, display_name="Need222Say"))

    await adapter.dispatch_task_followthrough(
        interaction,
        thread_id="777",
        thread_name="Reply thread",
        text="Selected option: 2. 先起草回复/材料\n\n等价用户消息：按这个方向继续。",
    )

    adapter._threads.mark.assert_called_once_with("777")
    adapter._dispatch_thread_session.assert_awaited_once_with(
        interaction,
        "777",
        "Reply thread",
        "Selected option: 2. 先起草回复/材料\n\n等价用户消息：按这个方向继续。",
    )
