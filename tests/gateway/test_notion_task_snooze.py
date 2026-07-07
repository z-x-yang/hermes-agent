from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.discord.notion_tasks.controller import NotionTaskController
from plugins.platforms.discord.notion_tasks.tracker import NotionTaskTracker

PID = "1f17a58d229e816f839bef72f6f2ec72"
TASK_PAGE = {
    "parent": {"type": "database_id", "database_id": "1f17a58d-229e-816f-839b-ef72f6f2ec72"},
    "properties": {"Name": {"type": "title", "title": [{"plain_text": "Reply to Alice"}]},
                   "Status": {"type": "status", "status": {"name": "To Do"}}},
}
DONE_PAGE = {
    **TASK_PAGE,
    "properties": {**TASK_PAGE["properties"],
                   "Status": {"type": "status", "status": {"name": "Done"}}},
}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", tmp_path)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _now():
    return datetime(2026, 6, 24, 15, 0, 0)


def _ctrl(notion, *, fetch_channel=None, now_fn=_now):
    from plugins.platforms.discord.notion_tasks.snooze import SnoozeStore
    return NotionTaskController(
        notion=notion,
        tracker=NotionTaskTracker(),
        snoozes=SnoozeStore(),
        allowed_ids_getter=lambda: ({"42"}, set()),
        tasks_ids={PID},
        fetch_channel=fetch_channel,
        now_fn=now_fn,
    )


def _interaction(user_id="42", msg_id="1001", channel_id="9001", content="Reply to Alice"):
    user = SimpleNamespace(id=user_id, roles=[])
    channel = SimpleNamespace(id=channel_id, parent_id="8000")
    msg = SimpleNamespace(id=msg_id, content=content, channel=channel)
    return SimpleNamespace(
        user=user,
        message=msg,
        channel=channel,
        guild=SimpleNamespace(id="147"),
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock(), defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def test_snooze_store_persists_due_and_cancel_pending(home):
    from plugins.platforms.discord.notion_tasks.snooze import SnoozeStore

    store = SnoozeStore()
    rec_id = store.schedule(
        page_id=PID,
        title="Reply to Alice",
        due_at=100.0,
        channel_id="9001",
        message_id="1001",
        user_id="42",
        original_content="body",
        preset="1h",
    )

    due = SnoozeStore().due(now=101.0)
    assert [r["id"] for r in due] == [rec_id]
    assert due[0]["status"] == "pending"

    assert store.cancel_pending(PID, reason="completed") == 1
    assert SnoozeStore().due(now=101.0) == []
    assert SnoozeStore().get(rec_id)["status"] == "cancelled"


def test_snooze_time_presets_are_deterministic():
    from plugins.platforms.discord.notion_tasks.snooze import resolve_due

    now = datetime(2026, 6, 24, 15, 0, 0)  # Wednesday
    assert resolve_due("1h", now=now).isoformat() == "2026-06-24T16:00:00"
    assert resolve_due("tonight", now=now).isoformat() == "2026-06-24T20:30:00"
    assert resolve_due("tomorrow_morning", now=now).isoformat() == "2026-06-25T09:30:00"
    assert resolve_due("three_days_morning", now=now).isoformat() == "2026-06-27T09:30:00"
    assert resolve_due("next_monday", now=now).isoformat() == "2026-06-29T09:30:00"


def test_format_notion_datetime_ceil_to_minute():
    from plugins.platforms.discord.notion_tasks.snooze import format_notion_datetime

    now = datetime(2026, 6, 24, 16, 0, 17, 123456)

    assert format_notion_datetime(now) == "2026-06-24T16:01:00"


@pytest.mark.asyncio
async def test_snooze_action_edits_original_message_with_choice_menu():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status=AsyncMock())
    ctrl = _ctrl(notion)
    inter = _interaction(content=f"Task https://notion.so/{PID}")

    await ctrl.handle_action("snooze", PID, inter)

    notion.set_status.assert_not_awaited()
    inter.response.send_message.assert_not_awaited()
    inter.response.edit_message.assert_awaited_once()
    kwargs = inter.response.edit_message.call_args.kwargs
    assert "content" not in kwargs
    view = kwargs["view"]
    assert any(getattr(child, "custom_id", "").startswith("ntask:snooze-select:") for child in view.children)
    assert [child.item.label for child in view.children if hasattr(child, "item")] == [
        "🧵1", "✓1", "⏸1", "🗑1", "⏰1"]


@pytest.mark.asyncio
async def test_slash_snooze_opens_ephemeral_choice_menu_without_notion_write():
    page = {**TASK_PAGE, "id": PID}
    notion = SimpleNamespace(
        find_task_by_discord_thread_id=AsyncMock(return_value=[page]),
        set_hold_verified=AsyncMock(),
    )
    ctrl = _ctrl(notion)
    inter = _interaction(content="")

    await ctrl.handle_slash_snooze(inter)

    notion.set_hold_verified.assert_not_awaited()
    inter.response.send_message.assert_awaited_once()
    assert "稍后多久提醒" in inter.response.send_message.await_args.args[0]
    view = inter.response.send_message.await_args.kwargs["view"]
    assert any(getattr(child, "custom_id", "").startswith("ntask:snooze-select:") for child in view.children)


@pytest.mark.asyncio
async def test_snooze_choice_writes_notion_hold_and_confirms():
    from cron.jobs import list_jobs

    held_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                 "Status": {"type": "status", "status": {"name": "Hold"}},
                 "Next Check": {"type": "date", "date": {"start": "2026-06-24T16:00:00"}},
                 "Hold Reason": {"type": "rich_text", "rich_text": [{"plain_text": "snoozed"}]}}}
    notion = SimpleNamespace(
        get_page=AsyncMock(return_value=TASK_PAGE),
        set_hold_verified=AsyncMock(return_value=held_page),
    )
    ctrl = _ctrl(notion)
    inter = _interaction(content=f"Task https://notion.so/{PID}")

    await ctrl.handle_snooze_choice(
        PID,
        "1h",
        inter,
        source_channel_id="9001",
        source_message_id="1001",
        source_content=f"Task https://notion.so/{PID}",
    )

    notion.set_hold_verified.assert_awaited_once()
    assert notion.set_hold_verified.await_args.kwargs["reason"] == "snoozed"
    assert notion.set_hold_verified.await_args.kwargs["next_check"] == "2026-06-24T16:00:00"
    jobs = list_jobs(include_disabled=True)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["no_agent"] is True
    assert job["script"].startswith("task_snooze/")
    assert job["deliver"] == "discord:8000:9001"
    assert job["repeat"]["times"] == 1
    pending = ctrl.snoozes.pending_for(PID)
    assert pending["status"] == "cron_scheduled"
    assert pending["cron_job_id"] == job["id"]
    assert pending["deliver"] == "discord:8000:9001"
    assert ctrl.snoozes.due(now=_now().timestamp() + 3601) == []
    from hermes_constants import get_hermes_home
    assert (get_hermes_home() / "scripts" / job["script"]).exists()
    inter.response.send_message.assert_not_awaited()
    inter.response.edit_message.assert_awaited_once()
    kwargs = inter.response.edit_message.call_args.kwargs
    assert "⏰ 已延后·06/24 16:00" in kwargs["embed"].description


@pytest.mark.asyncio
async def test_snooze_choice_reports_partial_failure_when_cron_create_fails(monkeypatch):
    from plugins.platforms.discord.notion_tasks import controller as controller_mod

    held_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                 "Status": {"type": "status", "status": {"name": "Hold"}},
                 "Next Check": {"type": "date", "date": {"start": "2026-06-24T16:00:00"}},
                 "Hold Reason": {"type": "rich_text", "rich_text": [{"plain_text": "snoozed"}]}}}
    notion = SimpleNamespace(
        get_page=AsyncMock(return_value=TASK_PAGE),
        set_hold_verified=AsyncMock(return_value=held_page),
    )
    ctrl = _ctrl(notion)
    inter = _interaction(content=f"Task https://notion.so/{PID}")

    def boom(*args, **kwargs):
        raise RuntimeError("cron down")

    monkeypatch.setattr(controller_mod, "create_snooze_cron", boom, raising=False)

    await ctrl.handle_snooze_choice(
        PID,
        "1h",
        inter,
        source_channel_id="9001",
        source_message_id="1001",
        source_content=f"Task https://notion.so/{PID}",
    )

    notion.set_hold_verified.assert_awaited_once()
    inter.response.send_message.assert_awaited_once()
    assert "Cron" in inter.response.send_message.await_args.args[0]
    inter.response.edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_snooze_choice_sends_minute_aligned_next_check():
    held_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                 "Status": {"type": "status", "status": {"name": "Hold"}},
                 "Next Check": {"type": "date", "date": {"start": "2026-06-24T16:01:00"}},
                 "Hold Reason": {"type": "rich_text", "rich_text": [{"plain_text": "snoozed"}]}}}
    notion = SimpleNamespace(
        get_page=AsyncMock(return_value=TASK_PAGE),
        set_hold_verified=AsyncMock(return_value=held_page),
    )
    ctrl = _ctrl(notion, now_fn=lambda: datetime(2026, 6, 24, 15, 0, 17, 123456))
    inter = _interaction(content=f"Task https://notion.so/{PID}")

    await ctrl.handle_snooze_choice(
        PID,
        "1h",
        inter,
        source_channel_id="9001",
        source_message_id="1001",
        source_content=f"Task https://notion.so/{PID}",
    )

    assert notion.set_hold_verified.await_args.kwargs["next_check"] == "2026-06-24T16:01:00"


@pytest.mark.asyncio
async def test_dispatch_due_snooze_sends_reminder_when_task_still_open():
    channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=222, content="")))
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE))
    ctrl = _ctrl(notion, fetch_channel=AsyncMock(return_value=channel))
    ctrl.snoozes.schedule(
        page_id=PID,
        title="Reply to Alice",
        due_at=_now().timestamp() - 1,
        channel_id="9001",
        message_id="1001",
        user_id="42",
        original_content=f"Task https://notion.so/{PID}",
        preset="1h",
    )

    sent = await ctrl.dispatch_due_snoozes()

    assert sent == 1
    channel.send.assert_awaited_once()
    kwargs = channel.send.call_args.kwargs
    assert kwargs["content"].startswith("⏰ 稍后提醒：Reply to Alice")
    assert "https://app.notion.com/p/" in kwargs["content"]
    assert [child.item.label for child in kwargs["view"].children] == [
        "🧵1", "✓1", "⏸1", "🗑1", "⏰1"]
    # reminder carries a fresh open-row task card (the pending record itself
    # must not paint the row as 已延后 — this IS the reminder firing)
    assert f"1️⃣ [Reply to Alice](https://www.notion.so/{PID})" in kwargs["embed"].description
    assert "已延后" not in kwargs["embed"].description
    assert ctrl.tracker.locations(PID)[0]["message_id"] == "222"
    assert ctrl.snoozes.due(now=_now().timestamp()) == []


@pytest.mark.asyncio
async def test_dispatch_due_snooze_cancels_when_task_already_done():
    channel = SimpleNamespace(send=AsyncMock())
    notion = SimpleNamespace(get_page=AsyncMock(return_value=DONE_PAGE))
    ctrl = _ctrl(notion, fetch_channel=AsyncMock(return_value=channel))
    rec_id = ctrl.snoozes.schedule(
        page_id=PID,
        title="Reply to Alice",
        due_at=_now().timestamp() - 1,
        channel_id="9001",
        message_id="1001",
        user_id="42",
        original_content=f"Task https://notion.so/{PID}",
        preset="1h",
    )

    sent = await ctrl.dispatch_due_snoozes()

    assert sent == 0
    channel.send.assert_not_awaited()
    rec = ctrl.snoozes.get(rec_id)
    assert rec["status"] == "cancelled"
    assert rec["cancel_reason"] == "already_done"


@pytest.mark.asyncio
async def test_dispatch_due_snooze_retries_without_sending_when_notion_state_unknown():
    channel = SimpleNamespace(send=AsyncMock())
    notion = SimpleNamespace(get_page=AsyncMock(side_effect=RuntimeError("notion offline")))
    ctrl = _ctrl(notion, fetch_channel=AsyncMock(return_value=channel))
    rec_id = ctrl.snoozes.schedule(
        page_id=PID,
        title="Reply to Alice",
        due_at=_now().timestamp() - 1,
        channel_id="9001",
        message_id="1001",
        user_id="42",
        original_content=f"Task https://notion.so/{PID}",
        preset="1h",
    )

    sent = await ctrl.dispatch_due_snoozes()

    assert sent == 0
    channel.send.assert_not_awaited()
    rec = ctrl.snoozes.get(rec_id)
    assert rec is not None
    assert rec["status"] == "pending"
    assert rec["attempts"] == 1
    assert "notion offline" in rec["last_error"]


@pytest.mark.asyncio
async def test_dispatch_due_snooze_retries_without_sending_when_status_unreadable():
    channel = SimpleNamespace(send=AsyncMock())
    page_without_status = {"parent": TASK_PAGE["parent"], "properties": {"Name": TASK_PAGE["properties"]["Name"]}}
    notion = SimpleNamespace(get_page=AsyncMock(return_value=page_without_status))
    ctrl = _ctrl(notion, fetch_channel=AsyncMock(return_value=channel))
    rec_id = ctrl.snoozes.schedule(
        page_id=PID,
        title="Reply to Alice",
        due_at=_now().timestamp() - 1,
        channel_id="9001",
        message_id="1001",
        user_id="42",
        original_content=f"Task https://notion.so/{PID}",
        preset="1h",
    )

    sent = await ctrl.dispatch_due_snoozes()

    assert sent == 0
    channel.send.assert_not_awaited()
    rec = ctrl.snoozes.get(rec_id)
    assert rec is not None
    assert rec["status"] == "pending"
    assert rec["attempts"] == 1
    assert rec["last_error"] == "Notion Status property unreadable"


def test_pending_for_returns_pending_record_and_none_after_cancel(home):
    from plugins.platforms.discord.notion_tasks.snooze import SnoozeStore
    s = SnoozeStore()
    s.schedule(page_id=PID, title="t", due_at=123.0, channel_id="c",
               message_id="m", user_id="u", original_content="", preset="1h")
    rec = s.pending_for(PID)
    assert rec and rec["due_at"] == 123.0 and rec["status"] == "pending"
    s.cancel_pending(PID, reason="x")
    assert s.pending_for(PID) is None


def test_pending_for_unknown_page_none(home):
    from plugins.platforms.discord.notion_tasks.snooze import SnoozeStore
    assert SnoozeStore().pending_for("a" * 32) is None
