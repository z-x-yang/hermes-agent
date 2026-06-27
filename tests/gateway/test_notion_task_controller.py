from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.discord.notion_tasks.controller import NotionTaskController
from plugins.platforms.discord.notion_tasks.tracker import NotionTaskTracker
from plugins.platforms.discord.notion_tasks.notion_client import NotionError

PID = "1f17a58d229e816f839bef72f6f2ec72"
TASK_PAGE = {
    "parent": {"type": "database_id", "database_id": "1f17a58d-229e-816f-839b-ef72f6f2ec72"},
    "properties": {"Name": {"type": "title", "title": [{"plain_text": "Reply to Alice"}]},
                   "Status": {"type": "status", "status": {"name": "To Do"}}},
}
NON_TASK = {"parent": {"type": "page_id"}, "properties": {}}


@pytest.fixture(autouse=True)
def _clear_auth_env(monkeypatch):
    for n in ("DISCORD_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(n, raising=False)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def _ctrl(notion, fetch_channel=None):
    return NotionTaskController(notion=notion, tracker=NotionTaskTracker(),
                               allowed_ids_getter=lambda: ({"42"}, set()),
                               tasks_ids={PID}, fetch_channel=fetch_channel)


def _interaction(user_id="42", msg_id="m1", channel_id="c1", content="Reply to Alice"):
    user = SimpleNamespace(id=user_id, roles=[])
    msg = SimpleNamespace(id=msg_id, content=content, channel=SimpleNamespace(id=channel_id))
    return SimpleNamespace(
        user=user, message=msg,
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock(), defer=AsyncMock()),
        client=MagicMock(),
    )


@pytest.mark.asyncio
async def test_render_view_attaches_button_only_for_task_links():
    notion = SimpleNamespace(get_page=AsyncMock(side_effect=lambda pid: TASK_PAGE if pid == PID else NON_TASK))
    ctrl = _ctrl(notion)
    text = f"Built [Reply to Alice](https://notion.so/{PID}) and a [doc](https://notion.so/{'a' * 32})"
    view = await ctrl.render_view_for_text(text)
    assert view is not None and len(view.children) == 1   # only the task gets a button


@pytest.mark.asyncio
async def test_render_view_none_when_no_task_links():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=NON_TASK))
    assert await _ctrl(notion).render_view_for_text("plain message") is None


@pytest.mark.asyncio
async def test_complete_fetches_live_sets_done_and_strikes_content():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    inter = _interaction(content="Reply to Alice today")
    await ctrl.handle_action("done", PID, inter)
    # kind read live from the page (status, not select)
    notion.set_status.assert_awaited_once_with(PID, "Done", "status")
    rec = ctrl.tracker.get(PID)
    assert rec["done"] is True
    assert rec["original_status"] == "To Do"   # captured pre-done status
    # original content preserved (struck), not replaced by the title
    kwargs = inter.response.edit_message.call_args.kwargs
    assert kwargs["content"] == "✅ ~~Reply to Alice today~~"


@pytest.mark.asyncio
async def test_undo_restores_original_status_and_content():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    # seed as if it was completed earlier: original captured + a known message+content
    ctrl.tracker.upsert_meta(PID, title="Reply to Alice", status_kind="status",
                             original_status="To Do", done=True)
    ctrl.tracker.add_location(PID, message_id="m1", channel_id="c1", orig_content="Reply to Alice today")
    inter = _interaction(content="✅ ~~Reply to Alice today~~")  # currently struck
    await ctrl.handle_action("undo", PID, inter)
    notion.set_status.assert_awaited_once_with(PID, "To Do", "status")
    assert ctrl.tracker.get(PID)["done"] is False
    kwargs = inter.response.edit_message.call_args.kwargs
    assert kwargs["content"] == "Reply to Alice today"   # restored verbatim


@pytest.mark.asyncio
async def test_undo_without_recorded_original_fails_fast():
    # persistent undo button + lost tracker: must NOT guess a status.
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status=AsyncMock())
    ctrl = _ctrl(notion)
    inter = _interaction(content="✅ ~~something~~")
    await ctrl.handle_action("undo", PID, inter)
    notion.set_status.assert_not_awaited()           # no guessed write
    inter.response.send_message.assert_awaited_once()  # explicit error to user
    inter.response.edit_message.assert_not_awaited()


def _thread(starter_id=111, starter_chan=999, content=None):
    # bare URL (no markdown anchor) so the title falls back to the page title
    return SimpleNamespace(
        id=555,
        starter_message=SimpleNamespace(
            id=starter_id, content=content if content is not None else f"https://notion.so/{PID}",
            channel=SimpleNamespace(id=starter_chan)),
        send=AsyncMock(return_value=SimpleNamespace(id=222)),
        fetch_message=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_thread_open_registers_starter_and_thread_locations():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE))
    ctrl = _ctrl(notion)
    thread = _thread()
    await ctrl.on_thread_opened(thread)
    locs = {l["message_id"] for l in ctrl.tracker.locations(PID)}
    assert locs == {"111", "222"}     # starter (main) AND thread mirror both tracked
    # not done (live status To Do) -> plain base content, not struck
    assert thread.send.call_args.kwargs["content"] == "📌 Reply to Alice"


@pytest.mark.asyncio
async def test_thread_open_done_state_from_live_status():
    done_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                 "Status": {"type": "status", "status": {"name": "Done"}}}}
    notion = SimpleNamespace(get_page=AsyncMock(return_value=done_page))
    ctrl = _ctrl(notion)
    thread = _thread()
    await ctrl.on_thread_opened(thread)
    assert thread.send.call_args.kwargs["content"] == "✅ ~~📌 Reply to Alice~~"


@pytest.mark.asyncio
async def test_thread_first_click_syncs_main_message():
    # HIGH fix: clicking in the thread first must still update the main message.
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status=AsyncMock(return_value={}))
    main_msg = SimpleNamespace(content="Reply to Alice today", edit=AsyncMock())
    main_chan = SimpleNamespace(fetch_message=AsyncMock(return_value=main_msg))

    async def _fetch_channel(cid):
        assert str(cid) == "999"
        return main_chan

    main_content = f"Reply to Alice today https://notion.so/{PID}"
    main_msg.content = main_content
    ctrl = _ctrl(notion, fetch_channel=_fetch_channel)
    thread = _thread(starter_id=111, starter_chan=999, content=main_content)
    await ctrl.on_thread_opened(thread)   # registers main(111@999) + thread(222@555)
    # click done on the THREAD mirror message
    inter = _interaction(msg_id="222", channel_id="555", content="📌 Reply to Alice")
    await ctrl.handle_action("done", PID, inter)
    main_msg.edit.assert_awaited_once()
    assert main_msg.edit.call_args.kwargs["content"] == f"✅ ~~{main_content}~~"


@pytest.mark.asyncio
async def test_unauthorized_user_rejected():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status=AsyncMock())
    ctrl = _ctrl(notion)
    inter = _interaction(user_id="99999")
    await ctrl.handle_action("done", PID, inter)
    notion.set_status.assert_not_awaited()
    notion.get_page.assert_not_awaited()       # auth gate is before any Notion read
    inter.response.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_complete_failure_reports_and_does_not_mark_done():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE),
                             set_status=AsyncMock(side_effect=NotionError("502")))
    ctrl = _ctrl(notion)
    inter = _interaction()
    await ctrl.handle_action("done", PID, inter)
    assert (ctrl.tracker.get(PID) or {}).get("done") in (None, False)  # NOT marked done
    inter.response.send_message.assert_awaited()       # user sees an error
    inter.response.edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_page_failure_at_click_reports_and_aborts():
    notion = SimpleNamespace(get_page=AsyncMock(side_effect=NotionError("502")), set_status=AsyncMock())
    ctrl = _ctrl(notion)
    inter = _interaction()
    await ctrl.handle_action("done", PID, inter)
    notion.set_status.assert_not_awaited()
    inter.response.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_sync_edits_the_other_location():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status=AsyncMock(return_value={}))
    other_msg = SimpleNamespace(content="thread copy of task", edit=AsyncMock())
    other_chan = SimpleNamespace(fetch_message=AsyncMock(return_value=other_msg))

    async def _fetch_channel(cid):
        assert cid == "c2"
        return other_chan

    ctrl = _ctrl(notion, fetch_channel=_fetch_channel)
    # a second known location (e.g. the thread copy) exists. Discord message ids
    # are numeric snowflakes — _sync_other int()s them to fetch.
    ctrl.tracker.add_location(PID, message_id="2002", channel_id="c2")
    inter = _interaction(msg_id="1001", channel_id="c1", content="Reply to Alice")
    await ctrl.handle_action("done", PID, inter)
    other_msg.edit.assert_awaited_once()
    assert other_msg.edit.call_args.kwargs["content"] == "✅ ~~thread copy of task~~"
