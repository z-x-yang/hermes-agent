from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.discord.notion_tasks.controller import NotionTaskController
from plugins.platforms.discord.notion_tasks.tracker import NotionTaskTracker
from plugins.platforms.discord.notion_tasks.notion_client import NotionError
from plugins.platforms.discord.notion_tasks.buttons import TaskActionButton

PID = "1f17a58d229e816f839bef72f6f2ec72"
TASK_PAGE = {
    "parent": {"type": "database_id", "database_id": "1f17a58d-229e-816f-839b-ef72f6f2ec72"},
    "properties": {"Name": {"type": "title", "title": [{"plain_text": "Reply to Alice"}]},
                   "Status": {"type": "status", "status": {"name": "To Do"}}},
}
THREAD_URL = "https://discord.com/channels/147/777"
BOUND_TASK_PAGE = {
    **TASK_PAGE,
    "properties": {
        **TASK_PAGE["properties"],
        "Discord Thread ID": {"type": "rich_text", "rich_text": [{"plain_text": "777"}]},
        "Discord Thread URL": {"type": "url", "url": THREAD_URL},
    },
}
NON_TASK = {"parent": {"type": "page_id"}, "properties": {}}


def _link(title, pid=PID):
    """卡片行标题现在渲染为 masked link；断言用完整链接形式。"""
    return f"[{title}](https://www.notion.so/{pid})"


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
        user=user, message=msg, channel=msg.channel,
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock(), defer=AsyncMock()),
        client=MagicMock(),
    )


def _item(child):
    return getattr(child, "item", child)


def _select_from_view(view):
    for child in getattr(view, "children", []) or []:
        item = _item(child)
        if hasattr(item, "options"):
            return item
    raise AssertionError("view has no select")


def _button_from_view(view, label):
    for child in getattr(view, "children", []) or []:
        item = _item(child)
        if getattr(item, "label", None) == label:
            return item
    raise AssertionError(f"view has no button {label!r}")


def _task_clarify_interaction(**kwargs):
    inter = _interaction(**kwargs)
    inter.message.embeds = [SimpleNamespace(
        title="🧭 Task Clarify · Paper reply task",
        description=(
            "**这是什么**：合作者回了论文修改意见\n\n"
            "**可选下一步**\n"
            "1. **推荐：先开子区整理上下文** — 整理背景。\n"
            "2. **先起草回复/材料** — 先在子区里起草，不直接发送。\n"
            "3. **先梳理执行图** — 整理依赖。\n\n"
            "Other：你也可以直接写自己的方向。"
        ),
    )]
    inter.message.guild = SimpleNamespace(id="147")
    return inter


class _DeferredResponse:
    def __init__(self, events):
        self._done = False
        self._events = events
        self.send_message = AsyncMock()
        self.edit_message = AsyncMock()

    def is_done(self):
        return self._done

    async def defer(self, *args, **kwargs):
        self._events.append("defer")
        self._done = True


def _deferred_interaction(user_id="42", msg_id="m1", channel_id="c1", content="Reply to Alice"):
    events = []
    inter = _interaction(user_id=user_id, msg_id=msg_id, channel_id=channel_id, content=content)
    inter.response = _DeferredResponse(events)
    inter.edit_original_response = AsyncMock(side_effect=lambda **_kwargs: events.append("edit_original"))
    inter._events = events
    return inter


# ===========================================================================
# send path — render_send_attachments returns (numbered view, task-card embed)
# ===========================================================================


@pytest.mark.asyncio
async def test_resolve_task_for_slash_uses_current_thread_binding():
    page = {**TASK_PAGE, "id": PID}
    notion = SimpleNamespace(find_task_by_discord_thread_id=AsyncMock(return_value=[page]))
    ctrl = _ctrl(notion)
    inter = _interaction(channel_id="1523")

    resolved = await ctrl.resolve_task_for_discord_interaction(inter)

    assert resolved["page_id"] == PID
    assert resolved["page"] == page
    assert resolved["title"] == "Reply to Alice"
    assert resolved["source"] == "current_thread_binding"
    notion.find_task_by_discord_thread_id.assert_awaited_once_with("1523", ctrl.tasks_ids)


@pytest.mark.asyncio
async def test_resolve_task_for_slash_fails_closed_on_unbound_thread():
    notion = SimpleNamespace(find_task_by_discord_thread_id=AsyncMock(return_value=[]))
    ctrl = _ctrl(notion)
    inter = _interaction(channel_id="1523")

    with pytest.raises(RuntimeError, match="未绑定"):
        await ctrl.resolve_task_for_discord_interaction(inter)


@pytest.mark.asyncio
async def test_resolve_task_for_slash_fails_closed_on_multiple_thread_matches():
    notion = SimpleNamespace(find_task_by_discord_thread_id=AsyncMock(return_value=[{"id": "a"}, {"id": "b"}]))
    ctrl = _ctrl(notion)
    inter = _interaction(channel_id="1523")

    with pytest.raises(RuntimeError, match="多个 Notion Task"):
        await ctrl.resolve_task_for_discord_interaction(inter)


@pytest.mark.asyncio
async def test_slash_done_resolves_current_task_and_confirms():
    page = {**TASK_PAGE, "id": PID}
    done_page = {**page, "properties": {**page["properties"],
                 "Status": {"type": "status", "status": {"name": "Done"}}}}
    notion = SimpleNamespace(
        find_task_by_discord_thread_id=AsyncMock(return_value=[page]),
        set_status_verified=AsyncMock(return_value=done_page),
    )
    ctrl = _ctrl(notion)
    inter = _interaction(channel_id="1523")

    await ctrl.handle_slash_done(inter)

    inter.response.defer.assert_awaited_once()
    notion.set_status_verified.assert_awaited_once_with(PID, "Done", "status")
    inter.response.send_message.assert_awaited_once()
    assert "已完成" in inter.response.send_message.await_args.args[0]


@pytest.mark.asyncio
async def test_slash_hold_opens_reason_modal():
    ctrl = _ctrl(SimpleNamespace())
    inter = _interaction(channel_id="1523")
    inter.response.send_modal = AsyncMock()

    await ctrl.handle_slash_hold(inter)

    inter.response.send_modal.assert_awaited_once()
    modal = inter.response.send_modal.await_args.args[0]
    assert getattr(modal, "title", "") == "暂挂原因"


@pytest.mark.asyncio
async def test_slash_hold_submit_allows_blank_reason():
    page = {**TASK_PAGE, "id": PID}
    held_page = {**page, "properties": {**page["properties"],
                 "Status": {"type": "status", "status": {"name": "Hold"}},
                 "Next Check": {"type": "date", "date": None},
                 "Hold Reason": {"type": "rich_text", "rich_text": []}}}
    notion = SimpleNamespace(
        find_task_by_discord_thread_id=AsyncMock(return_value=[page]),
        set_hold_verified=AsyncMock(return_value=held_page),
    )
    ctrl = _ctrl(notion)
    inter = _interaction(channel_id="1523")

    await ctrl.handle_slash_hold_submit("", inter)

    notion.set_hold_verified.assert_awaited_once_with(PID, next_check=None, reason="", waiting_for=None)
    inter.response.send_message.assert_awaited_once()
    assert "已暂挂" in inter.response.send_message.await_args.args[0]


@pytest.mark.asyncio
async def test_slash_reopen_resolves_current_task_and_clears_snooze():
    page = {**TASK_PAGE, "id": PID}
    notion = SimpleNamespace(
        find_task_by_discord_thread_id=AsyncMock(return_value=[page]),
        reopen_verified=AsyncMock(return_value=page),
    )
    ctrl = _ctrl(notion)
    ctrl.snoozes.schedule(page_id=PID, title="Reply to Alice", due_at=123.0,
                          channel_id="1523", message_id="m1", user_id="42",
                          original_content="", preset="1h")
    inter = _interaction(channel_id="1523")

    await ctrl.handle_slash_reopen(inter)

    inter.response.defer.assert_awaited_once()
    notion.reopen_verified.assert_awaited_once_with(PID)
    assert ctrl.snoozes.pending_for(PID) is None
    inter.response.send_message.assert_awaited_once()
    assert "已重新打开" in inter.response.send_message.await_args.args[0]


@pytest.mark.asyncio
async def test_slash_bind_writes_current_thread_binding_manual_locked():
    page = {**TASK_PAGE, "id": PID, "properties": {**TASK_PAGE["properties"],
            "Discord Thread ID": {"type": "rich_text", "rich_text": []},
            "Discord Thread URL": {"type": "url", "url": None},
            "Thread Title Mode": {"type": "select", "select": None},
            "Thread Title Version": {"type": "number", "number": None}}}
    bound_page = {**page, "properties": {**page["properties"],
                  "Discord Thread ID": {"type": "rich_text", "rich_text": [{"plain_text": "1523"}]},
                  "Discord Thread URL": {"type": "url", "url": "https://discord.com/channels/147/1523"},
                  "Thread Title Mode": {"type": "select", "select": {"name": "manual_locked"}},
                  "Thread Title Version": {"type": "number", "number": 1}}}
    notion = SimpleNamespace(
        get_page=AsyncMock(return_value=page),
        find_task_by_discord_thread_id=AsyncMock(return_value=[]),
        set_thread_binding_verified=AsyncMock(return_value=bound_page),
    )
    ctrl = _ctrl(notion)
    ctrl.adapter = SimpleNamespace(mark_auto_titled_thread=MagicMock())
    inter = _interaction(channel_id="1523")
    inter.guild = SimpleNamespace(id="147")

    await ctrl.handle_slash_bind_submit(PID, inter)

    notion.set_thread_binding_verified.assert_awaited_once_with(
        PID,
        thread_id="1523",
        thread_url="https://discord.com/channels/147/1523",
        title_mode="manual_locked",
        title_version=1,
    )
    ctrl.adapter.mark_auto_titled_thread.assert_called_once_with("1523")
    inter.response.send_message.assert_awaited_once()
    assert "已绑定 Notion Task" in inter.response.send_message.await_args.args[0]


@pytest.mark.asyncio
async def test_slash_bind_without_task_ref_shows_recent_picker_and_search_button():
    page = {**TASK_PAGE, "id": PID}
    notion = SimpleNamespace(search_tasks_for_bind=AsyncMock(return_value=[page]))
    ctrl = _ctrl(notion)
    inter = _interaction(channel_id="1523")

    await ctrl.handle_slash_bind(inter, "")

    inter.response.defer.assert_awaited_once()
    notion.search_tasks_for_bind.assert_awaited_once_with("", ctrl.tasks_ids, limit=25)
    inter.response.send_message.assert_awaited_once()
    assert "选择要绑定" in inter.response.send_message.await_args.args[0]
    view = inter.response.send_message.await_args.kwargs["view"]
    select = _select_from_view(view)
    assert "Reply to Alice" in [opt.label for opt in select.options]
    _button_from_view(view, "搜索任务")


@pytest.mark.asyncio
async def test_slash_bind_search_submit_queries_keyword_and_shows_picker():
    page = {**TASK_PAGE, "id": PID}
    notion = SimpleNamespace(search_tasks_for_bind=AsyncMock(return_value=[page]))
    ctrl = _ctrl(notion)
    inter = _interaction(channel_id="1523")

    await ctrl.handle_slash_bind_search_submit("Alice", inter)

    inter.response.defer.assert_awaited_once()
    notion.search_tasks_for_bind.assert_awaited_once_with("Alice", ctrl.tasks_ids, limit=25)
    inter.response.send_message.assert_awaited_once()
    assert "Alice" in inter.response.send_message.await_args.args[0]
    view = inter.response.send_message.await_args.kwargs["view"]
    select = _select_from_view(view)
    assert select.options[0].value == PID
    _button_from_view(view, "重新搜索")


@pytest.mark.asyncio
async def test_slash_bind_rejects_target_already_bound_to_another_thread():
    page = {**TASK_PAGE, "id": PID, "properties": {**TASK_PAGE["properties"],
            "Discord Thread ID": {"type": "rich_text", "rich_text": [{"plain_text": "9999"}]},
            "Discord Thread URL": {"type": "url", "url": "https://discord.com/channels/147/9999"},
            "Thread Title Mode": {"type": "select", "select": {"name": "auto"}}}}
    notion = SimpleNamespace(
        get_page=AsyncMock(return_value=page),
        find_task_by_discord_thread_id=AsyncMock(return_value=[]),
        set_thread_binding_verified=AsyncMock(),
    )
    ctrl = _ctrl(notion)
    inter = _interaction(channel_id="1523")
    inter.guild = SimpleNamespace(id="147")

    await ctrl.handle_slash_bind_submit(PID, inter)

    notion.set_thread_binding_verified.assert_not_awaited()
    inter.response.send_message.assert_awaited_once()
    assert "已经绑定另一个子区" in inter.response.send_message.await_args.args[0]

@pytest.mark.asyncio
async def test_render_send_attachments_numbered_view_and_card():
    notion = SimpleNamespace(get_page=AsyncMock(side_effect=lambda pid: TASK_PAGE if pid == PID else NON_TASK))
    ctrl = _ctrl(notion)
    text = f"Built [Reply to Alice](https://notion.so/{PID}) and a [doc](https://notion.so/{'a' * 32})"
    view, embed = await ctrl.render_send_attachments(text)
    assert view is not None and len(view.children) == 5
    assert [child.item.label for child in view.children] == ["🧵1", "✓1", "⏸1", "🗑1", "⏰1"]
    assert f"1️⃣ {_link('Reply to Alice')}" in embed.description
    assert embed.title == "📋 任务"


@pytest.mark.asyncio
async def test_render_send_attachments_uses_link_button_for_existing_thread():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=BOUND_TASK_PAGE))
    ctrl = _ctrl(notion)

    view, _embed = await ctrl.render_send_attachments(f"Built [Reply](https://notion.so/{PID})")

    assert view is not None
    open_thread = _item(view.children[0])
    assert getattr(open_thread, "url", None) == THREAD_URL
    assert getattr(open_thread, "custom_id", None) is None
    assert _item(view.children[1]).custom_id == f"ntask:v1:done:{PID}"


@pytest.mark.asyncio
async def test_render_send_attachments_none_when_no_task_links():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=NON_TASK))
    assert await _ctrl(notion).render_send_attachments("plain message") == (None, None)


# ===========================================================================
# click path — the card embed is the ONLY thing edited; content never is
# ===========================================================================

@pytest.mark.asyncio
async def test_done_edits_embed_only_never_content():
    # the interaction message is an OLD bare-text message (pre-card): the first
    # click upgrades it in place by attaching the card embed — content untouched.
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status_verified=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    inter = _interaction(content="Reply to Alice today")
    await ctrl.handle_action("done", PID, inter)
    # kind read live from the page (status, not select)
    notion.set_status_verified.assert_awaited_once_with(PID, "Done", "status")
    rec = ctrl.tracker.get(PID)
    assert rec["done"] is True
    assert rec["original_status"] == "To Do"   # captured pre-done status
    kwargs = inter.response.edit_message.call_args.kwargs
    # THE invariant: message body is never edited
    assert "content" not in kwargs
    assert f"✅ ~~{_link('Reply to Alice')}~~" in kwargs["embed"].description
    assert "1/1 已完成" in kwargs["embed"].title
    assert [c.item.label for c in kwargs["view"].children] == ["↩ 1"]


@pytest.mark.asyncio
async def test_done_defers_before_notion_work_and_edits_original_response():
    events = []

    async def _get_page(_pid):
        events.append("get_page")
        return TASK_PAGE

    async def _set_status_verified(*_args):
        events.append("set_status")
        return {}

    notion = SimpleNamespace(
        get_page=AsyncMock(side_effect=_get_page),
        set_status_verified=AsyncMock(side_effect=_set_status_verified),
    )
    ctrl = _ctrl(notion)
    inter = _deferred_interaction(content="Reply to Alice today")
    inter.response._events = events
    inter._events = events
    inter.edit_original_response = AsyncMock(side_effect=lambda **_kwargs: events.append("edit_original"))

    await ctrl.handle_action("done", PID, inter)

    assert events[0] == "defer"
    assert events.index("defer") < events.index("get_page")
    assert events[-1] == "edit_original"
    inter.response.edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_undo_restores_open_row_and_status():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status_verified=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    ctrl.tracker.upsert_meta(PID, title="Reply to Alice", status_kind="status",
                             original_status="To Do", done=True)
    ctrl.tracker.add_location(PID, message_id="m1", channel_id="c1", orig_content="Reply to Alice today")
    inter = _interaction(content="Reply to Alice today")
    await ctrl.handle_action("undo", PID, inter)
    notion.set_status_verified.assert_awaited_once_with(PID, "To Do", "status")
    assert ctrl.tracker.get(PID)["done"] is False
    kwargs = inter.response.edit_message.call_args.kwargs
    assert "content" not in kwargs
    assert f"1️⃣ {_link('Reply to Alice')}" in kwargs["embed"].description   # open row again
    assert "已完成" not in kwargs["embed"].title
    labels = [c.item.label for c in kwargs["view"].children]
    assert labels == ["🧵1", "✓1", "⏸1", "🗑1", "⏰1"]


@pytest.mark.asyncio
async def test_task_clarify_undo_restores_authored_body_not_bare_link():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status_verified=AsyncMock(return_value=TASK_PAGE))
    ctrl = _ctrl(notion)
    ctrl.tracker.upsert_meta(PID, title="Reply to Alice", status_kind="status",
                             original_status="To Do", done=True)
    inter = _task_clarify_interaction(content="")

    await ctrl.handle_action("undo", PID, inter)

    kwargs = inter.response.edit_message.call_args.kwargs
    assert "content" not in kwargs
    assert kwargs["embed"].title.startswith("🧭 Task Clarify")
    assert "合作者回了论文修改意见" in kwargs["embed"].description
    assert "**可选下一步**" in kwargs["embed"].description
    labels = [_item(child).label for child in kwargs["view"].children]
    assert labels == ["1.", "2.", "3.", "Other", "🧵", "⏰", "⏸", "🗑", "✓"]


@pytest.mark.asyncio
async def test_task_clarify_done_then_undo_restores_original_choices():
    done_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                 "Status": {"type": "status", "status": {"name": "Done"}}}}

    async def _set_status(_pid, target, _kind):
        return done_page if target == "Done" else TASK_PAGE

    notion = SimpleNamespace(
        get_page=AsyncMock(side_effect=[TASK_PAGE, done_page]),
        set_status_verified=AsyncMock(side_effect=_set_status),
    )
    ctrl = _ctrl(notion)
    first = _task_clarify_interaction(content="")

    await ctrl.handle_action("done", PID, first)
    done_embed = first.response.edit_message.call_args.kwargs["embed"]
    assert "已选择：完成" in done_embed.description
    assert "**可选下一步**" not in done_embed.description

    second = _task_clarify_interaction(content="")
    second.message.embeds = [done_embed]
    await ctrl.handle_action("undo", PID, second)

    kwargs = second.response.edit_message.call_args.kwargs
    desc = kwargs["embed"].description
    assert "合作者回了论文修改意见" in desc
    assert "**可选下一步**" in desc
    assert "1. **推荐：先开子区整理上下文**" in desc
    assert "已选择：完成" not in desc
    assert "状态：已完成" not in desc
    labels = [_item(child).label for child in kwargs["view"].children]
    assert labels == ["1.", "2.", "3.", "Other", "🧵", "⏰", "⏸", "🗑", "✓"]


@pytest.mark.asyncio
async def test_drop_action_is_immediate_struck_and_undoable():
    dropped_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                    "Status": {"type": "status", "status": {"name": "Dropped"}}}}
    notion = SimpleNamespace(
        get_page=AsyncMock(return_value=TASK_PAGE),
        set_dropped_verified=AsyncMock(return_value=dropped_page),
    )
    ctrl = _ctrl(notion)
    inter = _interaction(content="Reply to Alice today")

    await ctrl.handle_action("drop", PID, inter)

    notion.set_dropped_verified.assert_awaited_once_with(
        PID,
        reason="user_dropped",
        source_fingerprint=None,
    )
    inter.response.send_message.assert_not_awaited()
    kwargs = inter.response.edit_message.call_args.kwargs
    assert "content" not in kwargs
    assert f"🛑 已弃置 · ~~{_link('Reply to Alice')}~~" in kwargs["embed"].description
    assert [c.item.label for c in kwargs["view"].children] == ["↩ 1"]


@pytest.mark.asyncio
async def test_undo_without_recorded_original_fails_fast():
    # persistent undo button + lost tracker: must NOT guess a status.
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status_verified=AsyncMock())
    ctrl = _ctrl(notion)
    inter = _interaction(content="something")
    await ctrl.handle_action("undo", PID, inter)
    notion.set_status_verified.assert_not_awaited()           # no guessed write
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
    kwargs = thread.send.call_args.kwargs
    assert kwargs["content"] == "📌 Reply to Alice"
    assert f"1️⃣ {_link('Reply to Alice')}" in kwargs["embed"].description   # open row in the card


@pytest.mark.asyncio
async def test_thread_open_done_state_renders_in_card_not_content():
    done_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                 "Status": {"type": "status", "status": {"name": "Done"}}}}
    notion = SimpleNamespace(get_page=AsyncMock(return_value=done_page))
    ctrl = _ctrl(notion)
    thread = _thread()
    await ctrl.on_thread_opened(thread)
    kwargs = thread.send.call_args.kwargs
    # content is NEVER struck — done state lives in the card embed
    assert kwargs["content"] == "📌 Reply to Alice"
    assert f"✅ ~~{_link('Reply to Alice')}~~" in kwargs["embed"].description


@pytest.mark.asyncio
async def test_thread_first_click_syncs_main_message():
    # clicking in the thread first must still update the main message (embed only).
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status_verified=AsyncMock(return_value={}))
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
    kwargs = main_msg.edit.call_args.kwargs
    assert "content" not in kwargs
    assert f"✅ ~~{_link('Reply to Alice')}~~" in kwargs["embed"].description


@pytest.mark.asyncio
async def test_unauthorized_user_rejected():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status_verified=AsyncMock())
    ctrl = _ctrl(notion)
    inter = _interaction(user_id="99999")
    await ctrl.handle_action("done", PID, inter)
    notion.set_status_verified.assert_not_awaited()
    notion.get_page.assert_not_awaited()       # auth gate is before any Notion read
    inter.response.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_complete_failure_reports_and_does_not_mark_done():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE),
                             set_status_verified=AsyncMock(side_effect=NotionError("502")))
    ctrl = _ctrl(notion)
    inter = _interaction()
    await ctrl.handle_action("done", PID, inter)
    assert (ctrl.tracker.get(PID) or {}).get("done") in (None, False)  # NOT marked done
    inter.response.send_message.assert_awaited()       # user sees an error
    inter.response.edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_write_requires_verified_client():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    inter = _interaction()
    await ctrl.handle_action("done", PID, inter)
    notion.set_status.assert_not_awaited()
    inter.response.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_get_page_failure_at_click_reports_and_aborts():
    notion = SimpleNamespace(get_page=AsyncMock(side_effect=NotionError("502")), set_status_verified=AsyncMock())
    ctrl = _ctrl(notion)
    inter = _interaction()
    await ctrl.handle_action("done", PID, inter)
    notion.set_status_verified.assert_not_awaited()
    inter.response.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_multi_task_complete_marks_only_its_row_and_keeps_buttons():
    # Two Notion task links in one reminder. Completing the first must mark ONLY
    # the first — in Notion AND in the card — and keep the second task's buttons.
    PID2 = "2f17a58d229e816f839bef72f6f2ec72"
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE),
                             set_status_verified=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    body = f"两个任务：[A](https://notion.so/{PID}) 和 [B](https://notion.so/{PID2})"
    inter = _interaction(content=body)
    await ctrl.handle_action("done", PID, inter)

    # only the clicked task is written to Notion
    notion.set_status_verified.assert_awaited_once_with(PID, "Done", "status")

    kwargs = inter.response.edit_message.call_args.kwargs
    # the body is not touched at all (no content kwarg — Discord keeps it as-is)
    assert "content" not in kwargs
    # card: clicked row struck, sibling row still open + numbered
    assert f"✅ ~~{_link('A')}~~" in kwargs["embed"].description
    assert f"2️⃣ {_link('B', PID2)}" in kwargs["embed"].description
    assert "1/2 已完成" in kwargs["embed"].title
    cids = {child.item.custom_id for child in kwargs["view"].children}
    assert f"ntask:v1:done:{PID2}" in cids
    assert f"ntask:v1:snooze:{PID2}" in cids
    assert f"ntask:v1:undo:{PID}" in cids
    # numbered labels match card rows
    labels = {child.item.custom_id: child.item.label for child in kwargs["view"].children}
    assert labels[f"ntask:v1:undo:{PID}"] == "↩ 1"
    assert labels[f"ntask:v1:done:{PID2}"] == "✓2"


@pytest.mark.asyncio
async def test_multi_task_refreshed_view_caps_at_discord_component_limit():
    # 25 distinct task links: one PRIMARY action per task (undo for the clicked
    # task, done for the rest), no snooze — never exceeds Discord's 25 cap.
    pids = [PID] + [f"{i:032x}" for i in range(1, 25)]
    assert len(set(pids)) == 25
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE),
                             set_status_verified=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    body = " ".join(f"[T{i}](https://notion.so/{pid})" for i, pid in enumerate(pids))
    inter = _interaction(content=body)
    await ctrl.handle_action("done", PID, inter)

    view = inter.response.edit_message.call_args.kwargs["view"]
    assert len(view.children) == 25                       # never exceeds Discord's cap
    cids = [child.item.custom_id for child in view.children]
    assert f"ntask:v1:undo:{PID}" in cids                    # clicked task flips to undo
    for pid in pids[1:]:
        assert f"ntask:v1:done:{pid}" in cids                # every other task keeps completion
    snoozes = [c for c in cids if c.startswith("ntask:v1:snooze:")]
    assert len(snoozes) == 0                              # primaries use all 25 slots


@pytest.mark.asyncio
async def test_multi_task_refreshed_view_spends_spare_slots_on_snooze():
    # A handful of tasks fits full Workbench controls for each open task; task
    # groups pack without straddling row boundaries.
    pids = [PID] + [f"{i:032x}" for i in range(1, 4)]      # 4 distinct tasks
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE),
                             set_status_verified=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    body = " ".join(f"[T{i}](https://notion.so/{pid})" for i, pid in enumerate(pids))
    inter = _interaction(content=body)
    await ctrl.handle_action("done", PID, inter)

    view = inter.response.edit_message.call_args.kwargs["view"]
    cids = [child.item.custom_id for child in view.children]
    assert f"ntask:v1:undo:{PID}" in cids                    # clicked -> undo (done task: no snooze)
    for pid in pids[1:]:
        assert f"ntask:v1:done:{pid}" in cids
        assert f"ntask:v1:snooze:{pid}" in cids
        assert f"ntask:v1:open_thread:{pid}" in cids
        assert f"ntask:v1:hold:{pid}" in cids
        assert f"ntask:v1:drop:{pid}" in cids
    assert f"ntask:v1:snooze:{PID}" not in cids              # the done task never offers snooze
    # 16 buttons (1 undo + 3×5 full open-task controls), groups intact.
    row_of = {}
    for child in view.children:
        row_of.setdefault(child.item.custom_id.split(":")[-1], child.row)
    assert None not in row_of.values()
    assert set(row_of.values()) == {0, 1, 2, 3}
    # each task's own buttons never straddle a row boundary
    per_pid_rows = {}
    for child in view.children:
        per_pid_rows.setdefault(child.item.custom_id.split(":")[-1], set()).add(child.row)
    assert all(len(rows) == 1 for rows in per_pid_rows.values())


# ===========================================================================
# click path — sibling task state must come from Notion, not the tracker cache
# ===========================================================================

@pytest.mark.asyncio
async def test_rebuild_sibling_state_from_notion_not_tracker_cache():
    # Task 2 has a STALE tracker record (done=True from an old click) but its
    # Notion status was since corrected back to "To Do" (e.g. by the triage
    # agent). Clicking ✓ on task 1 rebuilds the card — task 2 must render from
    # Notion's authoritative state (open), never the local cache.
    pid2 = "2" * 32
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE),
                             set_status_verified=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    ctrl.tracker.upsert_meta(pid2, title="T2", status_kind="status",
                             original_status="To Do", done=True)   # stale cache
    body = f"[T1](https://notion.so/{PID}) [T2](https://notion.so/{pid2})"
    inter = _interaction(content=body)
    await ctrl.handle_action("done", PID, inter)

    kwargs = inter.response.edit_message.call_args.kwargs
    lines = kwargs["embed"].description.splitlines()
    assert "~~" in lines[0]                       # clicked task struck
    assert "~~" not in lines[1]                   # sibling NOT struck (Notion: To Do)
    assert "1/2 已完成" in kwargs["embed"].title
    cids = [c.item.custom_id for c in kwargs["view"].children]
    assert f"ntask:v1:done:{pid2}" in cids           # sibling keeps its ✓ button
    assert f"ntask:v1:undo:{pid2}" not in cids


@pytest.mark.asyncio
async def test_rebuild_sibling_done_in_notion_renders_done():
    # Converse: sibling really IS Done in Notion (no tracker record at all) —
    # the rebuilt card must strike it and offer undo.
    pid2 = "2" * 32
    done_page = {
        "parent": TASK_PAGE["parent"],
        "properties": {"Name": {"type": "title", "title": [{"plain_text": "T2"}]},
                       "Status": {"type": "status", "status": {"name": "Done"}}},
    }
    notion = SimpleNamespace(
        get_page=AsyncMock(side_effect=lambda pid: done_page if pid == pid2 else TASK_PAGE),
        set_status_verified=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    body = f"[T1](https://notion.so/{PID}) [T2](https://notion.so/{pid2})"
    inter = _interaction(content=body)
    await ctrl.handle_action("done", PID, inter)

    kwargs = inter.response.edit_message.call_args.kwargs
    lines = kwargs["embed"].description.splitlines()
    assert "~~" in lines[0] and "~~" in lines[1]  # both struck
    assert "2/2 已完成" in kwargs["embed"].title
    cids = [c.item.custom_id for c in kwargs["view"].children]
    assert f"ntask:v1:undo:{pid2}" in cids


@pytest.mark.asyncio
async def test_sync_edits_the_other_location_embed_only():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status_verified=AsyncMock(return_value={}))
    other_msg = SimpleNamespace(content="thread copy of task", edit=AsyncMock())
    other_chan = SimpleNamespace(fetch_message=AsyncMock(return_value=other_msg))

    async def _fetch_channel(cid):
        assert cid == "c2"
        return other_chan

    ctrl = _ctrl(notion, fetch_channel=_fetch_channel)
    ctrl.tracker.add_location(PID, message_id="2002", channel_id="c2")
    inter = _interaction(msg_id="1001", channel_id="c1", content="Reply to Alice")
    await ctrl.handle_action("done", PID, inter)
    other_msg.edit.assert_awaited_once()
    kwargs = other_msg.edit.call_args.kwargs
    assert "content" not in kwargs
    assert f"✅ ~~{_link('Reply to Alice')}~~" in kwargs["embed"].description


@pytest.mark.asyncio
async def test_sync_other_rebuilds_multi_task_sibling():
    # a tracked sibling location is ITSELF a multi-task reminder. Completing PID
    # must keep PID2's buttons there and strike only PID's card row.
    PID2 = "2f17a58d229e816f839bef72f6f2ec72"
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE),
                             set_status_verified=AsyncMock(return_value={}))
    sibling_body = f"两个任务：[A](https://notion.so/{PID}) 和 [B](https://notion.so/{PID2})"
    other_msg = SimpleNamespace(content=sibling_body, edit=AsyncMock())
    other_chan = SimpleNamespace(fetch_message=AsyncMock(return_value=other_msg))

    async def _fetch_channel(cid):
        assert cid == "c2"
        return other_chan

    ctrl = _ctrl(notion, fetch_channel=_fetch_channel)
    ctrl.tracker.add_location(PID, message_id="2002", channel_id="c2")
    # click done on the (single-task) main message
    inter = _interaction(msg_id="1001", channel_id="c1", content="Reply to Alice")
    await ctrl.handle_action("done", PID, inter)

    other_msg.edit.assert_awaited_once()
    kwargs = other_msg.edit.call_args.kwargs
    assert "content" not in kwargs
    assert f"✅ ~~{_link('A')}~~" in kwargs["embed"].description
    assert f"2️⃣ {_link('B', PID2)}" in kwargs["embed"].description
    cids = {child.item.custom_id for child in kwargs["view"].children}
    assert f"ntask:v1:done:{PID2}" in cids
    assert f"ntask:v1:snooze:{PID2}" in cids
    assert f"ntask:v1:undo:{PID}" in cids


@pytest.mark.asyncio
async def test_refresh_failure_after_set_status_preserves_message_and_buttons():
    # clicked message has PID + PID2; after PID is set_status'd, reading PID2
    # for the rebuild fails. Rebuilding without PID2 would silently drop its
    # row/buttons — refuse, keep the message untouched, tell the user.
    PID2 = "2f17a58d229e816f839bef72f6f2ec72"

    def _get_page(pid):
        if pid == PID2:
            raise NotionError("502 reading sibling task")
        return TASK_PAGE

    notion = SimpleNamespace(get_page=AsyncMock(side_effect=_get_page),
                             set_status_verified=AsyncMock(return_value={}))
    ctrl = _ctrl(notion)
    body = f"两个任务：[A](https://notion.so/{PID}) 和 [B](https://notion.so/{PID2})"
    inter = _interaction(content=body)
    await ctrl.handle_action("done", PID, inter)

    # the clicked task WAS written to Notion
    notion.set_status_verified.assert_awaited_once_with(PID, "Done", "status")
    # did NOT rebuild a partial card (which would drop PID2's row/controls)
    inter.response.edit_message.assert_not_awaited()
    # surfaced explicitly to the user instead of silently dropping controls
    inter.response.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_snooze_choice_marks_row_delayed_on_source_message():
    held_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                 "Status": {"type": "status", "status": {"name": "Hold"}},
                 "Next Check": {"type": "date", "date": {"start": "2026-07-07T09:30:00-04:00"}},
                 "Hold Reason": {"type": "rich_text", "rich_text": [{"plain_text": "snoozed"}]}}}
    notion = SimpleNamespace(
        get_page=AsyncMock(side_effect=[TASK_PAGE, held_page, held_page]),
        set_hold_verified=AsyncMock(return_value=held_page),
    )
    src_msg = SimpleNamespace(content=f"Task https://notion.so/{PID}", edit=AsyncMock())
    src_chan = SimpleNamespace(fetch_message=AsyncMock(return_value=src_msg))
    ctrl = _ctrl(notion, fetch_channel=AsyncMock(return_value=src_chan))
    inter = _interaction(content=f"Task https://notion.so/{PID}")

    await ctrl.handle_snooze_choice(
        PID, "1h", inter,
        source_channel_id="c1", source_message_id="3003",
        source_content=f"Task https://notion.so/{PID}",
    )

    notion.set_hold_verified.assert_awaited_once()
    assert notion.set_hold_verified.await_args.kwargs["reason"] == "snoozed"
    src_msg.edit.assert_awaited_once()
    kwargs = src_msg.edit.call_args.kwargs
    assert "content" not in kwargs
    assert "⏰ 已延后·" in kwargs["embed"].description
    assert "Reply to Alice" in kwargs["embed"].description


@pytest.mark.asyncio
async def test_hold_confirm_sets_notion_hold_and_cancels_snooze():
    held_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                 "Status": {"type": "status", "status": {"name": "Hold"}}}}
    notion = SimpleNamespace(set_hold_verified=AsyncMock(return_value=held_page))
    ctrl = _ctrl(notion)
    ctrl.snoozes.cancel_pending = MagicMock()
    inter = _interaction()

    await ctrl.handle_hold_confirm(PID, "manual_hold", None, inter)

    notion.set_hold_verified.assert_awaited_once_with(
        PID,
        next_check=None,
        reason="manual_hold",
        waiting_for=None,
    )
    ctrl.snoozes.cancel_pending.assert_called_once_with(PID, reason="moved_to_notion_hold")
    inter.response.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_hold_button_opens_optional_reason_modal(monkeypatch):
    monkeypatch.setattr(
        "plugins.platforms.discord.notion_tasks.buttons.get_active_controller",
        lambda: SimpleNamespace(handle_action=AsyncMock()),
    )
    response = SimpleNamespace(send_modal=AsyncMock(), send_message=AsyncMock())
    inter = SimpleNamespace(response=response)
    button = TaskActionButton("hold", PID)

    await button.callback(inter)

    response.send_modal.assert_awaited_once()
    modal = response.send_modal.await_args.args[0]
    assert modal.title == "暂挂原因"
    assert modal.reason_input.required is False


@pytest.mark.asyncio
async def test_snooze_action_edits_original_message_with_select_instead_of_new_message():
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
    select = next(child for child in view.children if getattr(child, "custom_id", "").startswith("ntask:snooze-select:"))
    assert [child.item.label for child in view.children if hasattr(child, "item")] == [
        "🧵1", "✓1", "⏸1", "🗑1", "⏰1"]
    labels = [option.label for option in select.options]
    assert "3天后 9:30" in labels


@pytest.mark.asyncio
async def test_task_clarify_snooze_picker_preserves_authored_card_body():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status=AsyncMock())
    ctrl = _ctrl(notion)
    inter = _task_clarify_interaction(content="")

    await ctrl.handle_action("snooze", PID, inter)

    inter.response.send_message.assert_not_awaited()
    kwargs = inter.response.edit_message.call_args.kwargs
    assert kwargs["embed"].title.startswith("🧭 Task Clarify")
    assert "合作者回了论文修改意见" in kwargs["embed"].description
    assert "**可选下一步**" in kwargs["embed"].description
    labels = [_item(child).label for child in kwargs["view"].children if hasattr(child, "item")]
    assert labels == ["1.", "2.", "3.", "Other", "🧵", "⏰", "⏸", "🗑", "✓"]
    assert any(getattr(child, "custom_id", "").startswith("ntask:snooze-select:")
               for child in kwargs["view"].children)


@pytest.mark.asyncio
async def test_snooze_picker_timeout_restores_original_buttons():
    src_msg = SimpleNamespace(content=f"Task https://notion.so/{PID}", edit=AsyncMock())
    src_chan = SimpleNamespace(fetch_message=AsyncMock(return_value=src_msg))
    ctrl = _ctrl(SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE)),
                 fetch_channel=AsyncMock(return_value=src_chan))
    inter = _interaction(msg_id="3003", channel_id="c1", content=f"Task https://notion.so/{PID}")

    await ctrl.handle_action("snooze", PID, inter)
    view = inter.response.edit_message.call_args.kwargs["view"]

    await view.on_timeout()

    src_msg.edit.assert_awaited_once()
    restored = src_msg.edit.call_args.kwargs["view"]
    assert [child.item.label for child in restored.children] == [
        "🧵1", "✓1", "⏸1", "🗑1", "⏰1"]
    assert not any(getattr(child, "custom_id", "").startswith("ntask:snooze-select:") for child in restored.children)


@pytest.mark.asyncio
async def test_snooze_picker_no_spare_row_falls_back_without_editing_source():
    pids = [PID] + [f"{i:032x}" for i in range(1, 5)]
    body = " ".join(f"[T{i}](https://notion.so/{pid})" for i, pid in enumerate(pids))
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE), set_status=AsyncMock())
    ctrl = _ctrl(notion)
    inter = _interaction(content=body)

    await ctrl.handle_action("snooze", PID, inter)

    inter.response.edit_message.assert_not_awaited()
    inter.response.send_message.assert_awaited_once()
    kwargs = inter.response.send_message.call_args.kwargs
    assert kwargs["ephemeral"] is True
    assert any(getattr(child, "custom_id", "").startswith("ntask:snooze-select:")
               for child in kwargs["view"].children)


@pytest.mark.asyncio
async def test_open_thread_creates_discord_thread_and_binds_notion():
    notion = SimpleNamespace(
        get_page=AsyncMock(return_value=TASK_PAGE),
        set_thread_binding_verified=AsyncMock(return_value=TASK_PAGE),
    )
    ctrl = _ctrl(notion)
    adapter = SimpleNamespace(create_task_thread_from_message=AsyncMock(
        return_value={"success": True, "thread_id": "777", "thread_name": "Reply to Alice"}))
    ctrl.adapter = adapter
    inter = _interaction(content=f"Task https://notion.so/{PID}")
    inter.message.guild = SimpleNamespace(id="147")

    await ctrl.handle_open_thread(PID, inter)

    adapter.create_task_thread_from_message.assert_awaited_once()
    args = adapter.create_task_thread_from_message.await_args
    assert args.kwargs["name"] == "Reply to Alice"
    assert "Notion:" in args.kwargs["seed"]
    notion.set_thread_binding_verified.assert_awaited_once_with(
        PID,
        thread_id="777",
        thread_url="https://discord.com/channels/147/777",
        title_mode="auto",
        title_version=1,
    )
    inter.response.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_open_thread_returns_existing_binding_without_creating():
    bound_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                  "Discord Thread ID": {"type": "rich_text", "rich_text": [{"plain_text": "777"}]},
                  "Discord Thread URL": {"type": "url", "url": "https://discord.com/channels/147/777"}}}
    notion = SimpleNamespace(get_page=AsyncMock(return_value=bound_page))
    ctrl = _ctrl(notion)
    adapter = SimpleNamespace(create_task_thread_from_message=AsyncMock())
    ctrl.adapter = adapter
    inter = _task_clarify_interaction()

    await ctrl.handle_open_thread(PID, inter)

    adapter.create_task_thread_from_message.assert_not_awaited()
    inter.response.send_message.assert_not_awaited()
    inter.response.edit_message.assert_awaited_once()
    kwargs = inter.response.edit_message.call_args.kwargs
    labels = [_item(child).label for child in kwargs["view"].children]
    assert labels == ["1.", "2.", "3.", "Other", "🧵", "⏰", "⏸", "🗑", "✓"]
    assert _item(kwargs["view"].children[4]).url == "https://discord.com/channels/147/777"


@pytest.mark.asyncio
async def test_open_thread_existing_binding_non_clarify_card_keeps_fallback_notice():
    bound_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                  "Discord Thread ID": {"type": "rich_text", "rich_text": [{"plain_text": "777"}]},
                  "Discord Thread URL": {"type": "url", "url": "https://discord.com/channels/147/777"}}}
    notion = SimpleNamespace(get_page=AsyncMock(return_value=bound_page))
    ctrl = _ctrl(notion)
    ctrl.adapter = SimpleNamespace(create_task_thread_from_message=AsyncMock())
    inter = _interaction(content=f"Task https://notion.so/{PID}")

    await ctrl.handle_open_thread(PID, inter)

    ctrl.adapter.create_task_thread_from_message.assert_not_awaited()
    inter.response.edit_message.assert_not_awaited()
    inter.response.send_message.assert_awaited_once()
    assert "<#777>" in inter.response.send_message.call_args.args[0]


@pytest.mark.asyncio
async def test_primary_choice_opens_thread_with_selected_strategy_seed():
    notion = SimpleNamespace(
        get_page=AsyncMock(return_value=TASK_PAGE),
        set_thread_binding_verified=AsyncMock(return_value=TASK_PAGE),
    )
    ctrl = _ctrl(notion)
    adapter = SimpleNamespace(create_task_thread_from_message=AsyncMock(
        return_value={"success": True, "thread_id": "888", "thread_name": "Reply to Alice"}))
    ctrl.adapter = adapter
    inter = _interaction(content=f"Task https://notion.so/{PID}")
    inter.message.guild = SimpleNamespace(id="147")
    inter.message.embeds = [SimpleNamespace(
        description="1. **推荐：先开子区整理上下文** — 整理背景。\n2. **先起草回复/材料** — 先在子区里起草，不直接发送。\n3. **先梳理执行图** — 整理依赖。"
    )]

    await ctrl.handle_action("choice2", PID, inter)

    adapter.create_task_thread_from_message.assert_awaited_once()
    seed = adapter.create_task_thread_from_message.await_args.kwargs["seed"]
    assert "Selected option: 2. 先起草回复/材料" in seed
    assert "不直接发送" in seed
    notion.set_thread_binding_verified.assert_awaited_once_with(
        PID,
        thread_id="888",
        thread_url="https://discord.com/channels/147/888",
        title_mode="auto",
        title_version=1,
    )


@pytest.mark.asyncio
async def test_primary_choice_posts_strategy_seed_to_existing_thread():
    bound_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                  "Discord Thread ID": {"type": "rich_text", "rich_text": [{"plain_text": "777"}]},
                  "Discord Thread URL": {"type": "url", "url": "https://discord.com/channels/147/777"}}}
    notion = SimpleNamespace(get_page=AsyncMock(return_value=bound_page))
    thread = SimpleNamespace(send=AsyncMock())

    async def fetch_channel(channel_id):
        assert channel_id == "777"
        return thread

    ctrl = _ctrl(notion, fetch_channel=fetch_channel)
    adapter = SimpleNamespace(create_task_thread_from_message=AsyncMock())
    ctrl.adapter = adapter
    inter = _task_clarify_interaction(content=f"Task https://notion.so/{PID}")

    await ctrl.handle_action("choice1", PID, inter)

    adapter.create_task_thread_from_message.assert_not_awaited()
    thread.send.assert_awaited_once()
    sent = thread.send.await_args.args[0]
    assert "Selected option: 1. 推荐：先开子区整理上下文" in sent
    assert "整理背景" in sent
    inter.response.send_message.assert_not_awaited()
    inter.response.edit_message.assert_awaited_once()
    kwargs = inter.response.edit_message.call_args.kwargs
    assert "已选择：推荐：先开子区整理上下文" in kwargs["embed"].description
    assert "状态：已在子区继续" in kwargs["embed"].description
    labels = [_item(child).label for child in kwargs["view"].children]
    assert labels == ["🧵", "⏰", "⏸", "🗑", "✓"]
    assert _item(kwargs["view"].children[0]).url == "https://discord.com/channels/147/777"
    assert all(not str(getattr(_item(child), "custom_id", "")).startswith("ntask:v1:choice")
               for child in kwargs["view"].children)


@pytest.mark.asyncio
async def test_primary_choice_dispatches_synthetic_user_turn_to_existing_thread():
    bound_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                  "Discord Thread ID": {"type": "rich_text", "rich_text": [{"plain_text": "777"}]},
                  "Discord Thread URL": {"type": "url", "url": "https://discord.com/channels/147/777"}}}
    notion = SimpleNamespace(get_page=AsyncMock(return_value=bound_page))
    thread = SimpleNamespace(name="Reply thread", send=AsyncMock())

    async def fetch_channel(channel_id):
        assert channel_id == "777"
        return thread

    ctrl = _ctrl(notion, fetch_channel=fetch_channel)
    ctrl.adapter = SimpleNamespace(
        create_task_thread_from_message=AsyncMock(),
        dispatch_task_followthrough=AsyncMock(),
    )
    inter = _task_clarify_interaction(content=f"Task https://notion.so/{PID}")

    await ctrl.handle_action("choice2", PID, inter)

    ctrl.adapter.dispatch_task_followthrough.assert_awaited_once()
    kwargs = ctrl.adapter.dispatch_task_followthrough.await_args.kwargs
    assert kwargs["thread_id"] == "777"
    assert kwargs["thread_name"] == "Reply thread"
    assert "Selected option: 2. 先起草回复/材料" in kwargs["text"]
    assert "按这个方向继续" in kwargs["text"]


@pytest.mark.asyncio
async def test_other_direction_submit_opens_thread_with_custom_seed():
    notion = SimpleNamespace(
        get_page=AsyncMock(return_value=TASK_PAGE),
        set_thread_binding_verified=AsyncMock(return_value=TASK_PAGE),
    )
    ctrl = _ctrl(notion)
    adapter = SimpleNamespace(create_task_thread_from_message=AsyncMock(
        return_value={"success": True, "thread_id": "889", "thread_name": "Reply to Alice"}))
    ctrl.adapter = adapter
    inter = _interaction(content=f"Task https://notion.so/{PID}")
    inter.message.guild = SimpleNamespace(id="147")

    await ctrl.handle_other_direction_submit(PID, "先查旧邮件，再起草回复", inter)

    adapter.create_task_thread_from_message.assert_awaited_once()
    seed = adapter.create_task_thread_from_message.await_args.kwargs["seed"]
    assert "Selected option: Other" in seed
    assert "Custom direction: 先查旧邮件，再起草回复" in seed
    notion.set_thread_binding_verified.assert_awaited_once_with(
        PID,
        thread_id="889",
        thread_url="https://discord.com/channels/147/889",
        title_mode="auto",
        title_version=1,
    )


@pytest.mark.asyncio
async def test_other_direction_submit_posts_custom_seed_to_existing_thread():
    bound_page = {**TASK_PAGE, "properties": {**TASK_PAGE["properties"],
                  "Discord Thread ID": {"type": "rich_text", "rich_text": [{"plain_text": "777"}]},
                  "Discord Thread URL": {"type": "url", "url": "https://discord.com/channels/147/777"}}}
    notion = SimpleNamespace(get_page=AsyncMock(return_value=bound_page))
    thread = SimpleNamespace(send=AsyncMock())

    async def fetch_channel(channel_id):
        assert channel_id == "777"
        return thread

    ctrl = _ctrl(notion, fetch_channel=fetch_channel)
    ctrl.adapter = SimpleNamespace(create_task_thread_from_message=AsyncMock())
    inter = _task_clarify_interaction(content=f"Task https://notion.so/{PID}")

    await ctrl.handle_other_direction_submit(PID, "先查旧邮件，再起草回复", inter)

    ctrl.adapter.create_task_thread_from_message.assert_not_awaited()
    thread.send.assert_awaited_once()
    sent = thread.send.await_args.args[0]
    assert "Selected option: Other" in sent
    assert "Custom direction: 先查旧邮件，再起草回复" in sent
    inter.response.send_message.assert_not_awaited()
    inter.response.edit_message.assert_awaited_once()
    kwargs = inter.response.edit_message.call_args.kwargs
    assert "已选择：Other — 先查旧邮件，再起草回复" in kwargs["embed"].description
    assert "状态：已在子区继续" in kwargs["embed"].description
