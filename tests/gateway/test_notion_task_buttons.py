import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.discord.notion_tasks import buttons as b

PID = "1f17a58d229e816f839bef72f6f2ec72"


def test_make_and_match_custom_id():
    cid = b.make_custom_id("done", PID)
    assert cid == f"ntask:v1:done:{PID}"
    m = re.fullmatch(b.CUSTOM_ID_RE, cid)
    assert m and m.group("action") == "done" and m.group("page_id") == PID

    snooze = b.make_custom_id("snooze", PID)
    m2 = re.fullmatch(b.CUSTOM_ID_RE, snooze)
    assert m2 and m2.group("action") == "snooze" and m2.group("page_id") == PID

    legacy = re.fullmatch(b.CUSTOM_ID_RE, f"ntask:snooze:{PID}")
    assert legacy and legacy.group("action") == "snooze"


def test_build_button_styles_and_labels():
    # no number -> legacy full-text label (restart-rebuilt buttons, never shown)
    done = b.build_button("done", PID)
    undo = b.build_button("undo", PID)
    snooze = b.build_button("snooze", PID)
    assert "完成" in done.item.label
    assert "撤销" in undo.item.label
    assert "稍后" in snooze.item.label
    assert done.item.custom_id == f"ntask:v1:done:{PID}"
    assert snooze.item.custom_id == f"ntask:v1:snooze:{PID}"
    assert b.build_button("open_thread", PID).item.custom_id == f"ntask:v1:open_thread:{PID}"
    assert b.build_button("drop", PID).item.custom_id == f"ntask:v1:drop:{PID}"
    assert b.build_button("choice1", PID).item.label == "1."
    assert b.build_button("choice2", PID).item.custom_id == f"ntask:v1:choice2:{PID}"
    assert b.build_button("other", PID).item.label == "Other"


def test_build_button_numbered_label():
    # buttons carry ONLY the row number — full task text lives in the card embed
    assert b.build_button("done", PID, num=3).item.label == "✓ 3"
    assert b.build_button("snooze", PID, num=1).item.label == "⏰ 1"
    assert b.build_button("undo", PID, num=2).item.label == "↩ 2"
    assert b.build_button("open_thread", PID, num=2).item.label == "🧵 2"
    assert b.build_button("hold", PID, num=2).item.label == "暂挂 2"
    assert b.build_button("drop", PID, num=2).item.label == "弃置 2"


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
