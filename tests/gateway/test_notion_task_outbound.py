from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.discord.notion_tasks import outbound

PID = "1f17a58d229e816f839bef72f6f2ec72"          # matches DEFAULT_TASKS_IDS database_id
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


def _fake_client(monkeypatch, page=TASK_PAGE):
    from plugins.platforms.discord.notion_tasks import notion_client as nc

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def get_page(self, pid):
            return page

    monkeypatch.setattr(nc, "NotionClient", _FakeClient)


@pytest.mark.asyncio
async def test_detect_task_links_only_returns_task_pages():
    notion = SimpleNamespace(get_page=AsyncMock(side_effect=lambda pid: TASK_PAGE if pid == PID else NON_TASK))
    text = f"[Reply to Alice](https://notion.so/{PID}) and a [doc](https://notion.so/{'a' * 32})"
    tasks = await outbound.detect_task_links(text, notion=notion)
    assert tasks == [(PID, "Reply to Alice")]


@pytest.mark.asyncio
async def test_detect_task_links_accepts_app_notion_p_urls():
    notion = SimpleNamespace(get_page=AsyncMock(return_value=TASK_PAGE))
    text = f"Notion: https://app.notion.com/p/Reply-to-Alice-{PID}"
    tasks = await outbound.detect_task_links(text, notion=notion)
    assert tasks == [(PID, "Reply to Alice")]
    notion.get_page.assert_awaited_once_with(PID)


@pytest.mark.asyncio
async def test_detect_task_links_skips_get_page_failure():
    notion = SimpleNamespace(get_page=AsyncMock(side_effect=RuntimeError("boom")))
    tasks = await outbound.detect_task_links(f"[x](https://notion.so/{PID})", notion=notion)
    assert tasks == []


@pytest.mark.asyncio
async def test_detect_task_links_no_notion_substring_short_circuits():
    notion = SimpleNamespace(get_page=AsyncMock())
    assert await outbound.detect_task_links("plain text", notion=notion) == []
    notion.get_page.assert_not_awaited()


@pytest.mark.asyncio
async def test_standalone_payload_builds_numbered_buttons_and_card(monkeypatch):
    _fake_client(monkeypatch)
    rows, embed = await outbound.standalone_task_payload(f"[Reply](https://notion.so/{PID})")
    assert len(rows) == 1
    buttons = rows[0]["components"]
    assert [b["custom_id"] for b in buttons] == [
        f"ntask:v1:open_thread:{PID}", f"ntask:v1:done:{PID}",
        f"ntask:v1:hold:{PID}", f"ntask:v1:drop:{PID}", f"ntask:v1:snooze:{PID}"]
    assert buttons[0]["style"] == 1
    assert buttons[1]["style"] == 3
    # buttons carry only the row number; full title lives in the card embed
    assert [b["label"] for b in buttons] == ["🧵1", "✓1", "⏸1", "🗑1", "⏰1"]
    assert embed["title"] == "📋 任务"
    assert f"1️⃣ [Reply](https://www.notion.so/{PID})" in embed["description"]


@pytest.mark.asyncio
async def test_standalone_payload_uses_thread_url_for_existing_binding(monkeypatch):
    _fake_client(monkeypatch, page=BOUND_TASK_PAGE)

    rows, _embed = await outbound.standalone_task_payload(f"[Reply](https://notion.so/{PID})")

    buttons = rows[0]["components"]
    assert buttons[0] == {"type": 2, "style": 5, "label": "🧵1", "url": THREAD_URL}
    assert "custom_id" not in buttons[0]
    assert buttons[1]["custom_id"] == f"ntask:v1:done:{PID}"


@pytest.mark.asyncio
async def test_standalone_payload_builds_buttons_for_app_notion(monkeypatch):
    _fake_client(monkeypatch)
    rows, embed = await outbound.standalone_task_payload(
        f"Notion: https://app.notion.com/p/Reply-to-Alice-{PID}"
    )
    assert len(rows) == 1
    assert [b["custom_id"] for b in rows[0]["components"]] == [
        f"ntask:v1:open_thread:{PID}", f"ntask:v1:done:{PID}",
        f"ntask:v1:hold:{PID}", f"ntask:v1:drop:{PID}", f"ntask:v1:snooze:{PID}"]
    # bare URL (no anchor) -> row title falls back to the Notion page title
    assert f"1️⃣ [Reply to Alice](https://www.notion.so/{PID})" in embed["description"]


@pytest.mark.asyncio
async def test_standalone_payload_caps_full_workbench_controls_when_overflowing(monkeypatch):
    _fake_client(monkeypatch)
    pids = [f"{i:032x}" for i in range(25)]
    rows, embed = await outbound.standalone_task_payload(
        " ".join(f"https://notion.so/{pid}" for pid in pids))
    buttons = [button for row in rows for button in row["components"]]

    assert len(buttons) == 25
    assert [b["custom_id"] for b in buttons] == [
        f"ntask:v1:{action}:{pids[i]}"
        for i in range(5)
        for action in ("open_thread", "done", "hold", "drop", "snooze")
    ]
    # every one of the 25 tasks keeps a numbered card row even when controls cap.
    assert embed["description"].count("\n") == 24


@pytest.mark.asyncio
async def test_standalone_payload_packs_task_groups_onto_shared_rows(monkeypatch):
    """Two tasks × 5 buttons wrap by whole task group; controls stay adjacent."""
    _fake_client(monkeypatch)
    pid2 = "2f17a58d229e816f839bef72f6f2ec72"
    text = f"[A](https://notion.so/{PID}) 和 [B](https://notion.so/{pid2})"
    rows, embed = await outbound.standalone_task_payload(text)

    assert len(rows) == 2
    assert [b["custom_id"] for b in rows[0]["components"]] == [
        f"ntask:v1:open_thread:{PID}", f"ntask:v1:done:{PID}",
        f"ntask:v1:hold:{PID}", f"ntask:v1:drop:{PID}", f"ntask:v1:snooze:{PID}"]
    assert [b["custom_id"] for b in rows[1]["components"]] == [
        f"ntask:v1:open_thread:{pid2}", f"ntask:v1:done:{pid2}",
        f"ntask:v1:hold:{pid2}", f"ntask:v1:drop:{pid2}", f"ntask:v1:snooze:{pid2}"]
    # button numbers match the card row order
    assert [b["label"] for b in rows[0]["components"]] == [
        "🧵1", "✓1", "⏸1", "🗑1", "⏰1"]
    assert [b["label"] for b in rows[1]["components"]] == [
        "🧵2", "✓2", "⏸2", "🗑2", "⏰2"]
    assert f"1️⃣ [A](https://www.notion.so/{PID})" in embed["description"]
    assert f"2️⃣ [B](https://www.notion.so/{pid2})" in embed["description"]


@pytest.mark.asyncio
async def test_standalone_payload_empty_when_no_task_links(monkeypatch):
    _fake_client(monkeypatch, page=NON_TASK)
    assert await outbound.standalone_task_payload(f"[x](https://notion.so/{PID})") == ([], None)
    assert await outbound.standalone_task_payload("plain text") == ([], None)


@pytest.mark.asyncio
async def test_standalone_payload_never_raises_on_client_error(monkeypatch):
    from plugins.platforms.discord.notion_tasks import notion_client as nc

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no token")

    monkeypatch.setattr(nc, "NotionClient", _Boom)
    # must degrade to no attachments (message still sends), not raise into delivery
    assert await outbound.standalone_task_payload(f"[x](https://notion.so/{PID})") == ([], None)
