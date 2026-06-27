import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.discord.notion_tasks import buttons as b

PID = "1f17a58d229e816f839bef72f6f2ec72"


def test_make_and_match_custom_id():
    cid = b.make_custom_id("done", PID)
    assert cid == f"ntask:done:{PID}"
    m = re.fullmatch(b.CUSTOM_ID_RE, cid)
    assert m and m.group("action") == "done" and m.group("page_id") == PID


def test_build_button_labels_and_style():
    done = b.build_button("done", PID, title="Reply to Alice")
    assert "完成" in done.item.label
    undo = b.build_button("undo", PID)
    assert "撤销" in undo.item.label
    assert done.item.custom_id == f"ntask:done:{PID}"


@pytest.mark.asyncio
async def test_callback_routes_to_active_controller(monkeypatch):
    ctrl = SimpleNamespace(handle_action=AsyncMock())
    monkeypatch.setattr(b, "get_active_controller", lambda: ctrl)
    btn = b.build_button("done", PID)
    interaction = SimpleNamespace()
    await btn.callback(interaction)
    ctrl.handle_action.assert_awaited_once_with("done", PID, interaction)


@pytest.mark.asyncio
async def test_callback_noop_when_no_controller(monkeypatch):
    monkeypatch.setattr(b, "get_active_controller", lambda: None)
    btn = b.build_button("undo", PID)
    # must not raise
    await btn.callback(SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock())))
