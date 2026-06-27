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
NON_TASK = {"parent": {"type": "page_id"}, "properties": {}}


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
async def test_standalone_components_builds_done_button(monkeypatch):
    from plugins.platforms.discord.notion_tasks import notion_client as nc

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def get_page(self, pid):
            return TASK_PAGE

    monkeypatch.setattr(nc, "NotionClient", _FakeClient)
    rows = await outbound.standalone_task_components(f"[Reply](https://notion.so/{PID})")
    assert len(rows) == 1
    btn = rows[0]["components"][0]
    assert btn["custom_id"] == f"ntask:done:{PID}" and btn["style"] == 3


@pytest.mark.asyncio
async def test_standalone_components_builds_done_button_for_app_notion(monkeypatch):
    from plugins.platforms.discord.notion_tasks import notion_client as nc

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def get_page(self, pid):
            return TASK_PAGE

    monkeypatch.setattr(nc, "NotionClient", _FakeClient)
    rows = await outbound.standalone_task_components(
        f"Notion: https://app.notion.com/p/Reply-to-Alice-{PID}"
    )
    assert len(rows) == 1
    assert rows[0]["components"][0]["custom_id"] == f"ntask:done:{PID}"


@pytest.mark.asyncio
async def test_standalone_components_never_raises_on_client_error(monkeypatch):
    from plugins.platforms.discord.notion_tasks import notion_client as nc

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no token")

    monkeypatch.setattr(nc, "NotionClient", _Boom)
    # must degrade to [] (message still sends), not raise into delivery
    assert await outbound.standalone_task_components(f"[x](https://notion.so/{PID})") == []
