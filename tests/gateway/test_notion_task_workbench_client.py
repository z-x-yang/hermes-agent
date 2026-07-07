import json

import httpx
import pytest

from plugins.platforms.discord.notion_tasks.notion_client import NotionClient

PID = "abc123abc123abc123abc123abc123ab"
TASKS_DATA_SOURCE_ID = "1f17a58d229e814496f3000b99bdcf95"


def _page(status="Hold", extra=None):
    props = {
        "Status": {"type": "status", "status": {"name": status}},
        "Next Check": {"type": "date", "date": {"start": "2026-07-07T09:30:00-04:00"}},
        "Hold Reason": {"type": "rich_text", "rich_text": [{"plain_text": "snoozed"}]},
    }
    if extra:
        props.update(extra)
    return {"id": PID, "properties": props}


def _text_prop(value):
    return {"type": "rich_text", "rich_text": [{"plain_text": value, "text": {"content": value}}]}


@pytest.mark.asyncio
async def test_query_data_source_posts_filter_body():
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path, json.loads(req.content or b"{}")))
        return httpx.Response(200, json={"results": []})

    client = NotionClient(api_key="secret", transport=httpx.MockTransport(handler), backoff=0)
    body = {
        "filter": {"property": "Discord Thread ID", "rich_text": {"equals": "1523"}},
        "page_size": 2,
    }

    result = await client.query_data_source(TASKS_DATA_SOURCE_ID, body)

    assert result == {"results": []}
    assert calls == [("POST", f"/v1/data_sources/{TASKS_DATA_SOURCE_ID}/query", body)]


@pytest.mark.asyncio
async def test_find_task_by_discord_thread_id_filters_tasks_source():
    bodies = []
    page = {
        "id": PID,
        "parent": {"type": "data_source_id", "data_source_id": TASKS_DATA_SOURCE_ID},
        "properties": {},
    }

    def handler(req):
        bodies.append(json.loads(req.content or b"{}"))
        return httpx.Response(200, json={"results": [page]})

    client = NotionClient(api_key="secret", transport=httpx.MockTransport(handler), backoff=0)

    pages = await client.find_task_by_discord_thread_id("1523", {TASKS_DATA_SOURCE_ID})

    assert pages == [page]
    assert bodies[0]["page_size"] == 2
    assert bodies[0]["filter"] == {
        "property": "Discord Thread ID",
        "rich_text": {"equals": "1523"},
    }


@pytest.mark.asyncio
async def test_set_hold_verified_patches_and_reads_back():
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path, json.loads(req.content or b"{}")))
        if req.method == "PATCH":
            return httpx.Response(200, json=_page("Hold"))
        if req.method == "GET":
            return httpx.Response(200, json=_page("Hold"))
        raise AssertionError(req)

    client = NotionClient(api_key="secret", transport=httpx.MockTransport(handler), backoff=0)
    page = await client.set_hold_verified(PID, next_check="2026-07-07T09:30:00-04:00", reason="snoozed", waiting_for=None)
    assert page["properties"]["Status"]["status"]["name"] == "Hold"
    patch_body = calls[0][2]
    assert patch_body["properties"]["Status"]["status"]["name"] == "Hold"
    assert patch_body["properties"]["Next Check"]["date"]["start"] == "2026-07-07T09:30:00-04:00"
    assert patch_body["properties"]["Hold Reason"]["rich_text"][0]["text"]["content"] == "snoozed"


@pytest.mark.asyncio
async def test_set_hold_verified_accepts_notion_minute_precision_readback():
    def handler(req):
        if req.method == "PATCH":
            return httpx.Response(200, json=_page("Hold"))
        if req.method == "GET":
            return httpx.Response(200, json=_page("Hold", {
                "Next Check": {"type": "date", "date": {"start": "2026-07-05T15:15:00.000+00:00"}},
                "Hold Reason": _text_prop("snoozed"),
            }))
        raise AssertionError(req)

    client = NotionClient(api_key="secret", transport=httpx.MockTransport(handler), backoff=0)
    page = await client.set_hold_verified(
        PID,
        next_check="2026-07-05T15:15:00",
        reason="snoozed",
        waiting_for=None,
    )
    assert page["properties"]["Next Check"]["date"]["start"] == "2026-07-05T15:15:00.000+00:00"


@pytest.mark.asyncio
async def test_set_hold_verified_fails_if_next_check_readback_mismatch():
    def handler(req):
        if req.method == "PATCH":
            return httpx.Response(200, json=_page("Hold"))
        if req.method == "GET":
            return httpx.Response(200, json=_page("Hold", {
                "Next Check": {"type": "date", "date": {"start": "2026-07-08T09:30:00-04:00"}},
                "Hold Reason": _text_prop("snoozed"),
            }))
        raise AssertionError(req)

    client = NotionClient(api_key="secret", transport=httpx.MockTransport(handler), backoff=0)
    with pytest.raises(Exception, match="Next Check read-back mismatch"):
        await client.set_hold_verified(PID, next_check="2026-07-07T09:30:00-04:00", reason="snoozed", waiting_for=None)


@pytest.mark.asyncio
async def test_set_status_verified_fails_if_readback_mismatch():
    def handler(req):
        if req.method == "PATCH":
            return httpx.Response(200, json=_page("Done"))
        if req.method == "GET":
            return httpx.Response(200, json=_page("To Do"))
        raise AssertionError(req)

    client = NotionClient(api_key="secret", transport=httpx.MockTransport(handler), backoff=0)
    with pytest.raises(Exception, match="read-back"):
        await client.set_status_verified(PID, "Done", "status")


@pytest.mark.asyncio
async def test_reopen_verified_writes_todo_and_clears_hold_fields():
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path, json.loads(req.content or b"{}")))
        if req.method == "PATCH":
            return httpx.Response(200, json=_page("To Do"))
        if req.method == "GET":
            return httpx.Response(200, json=_page("To Do", {
                "Next Check": {"type": "date", "date": None},
                "Hold Reason": {"type": "rich_text", "rich_text": []},
            }))
        raise AssertionError(req)

    client = NotionClient(api_key="secret", transport=httpx.MockTransport(handler), backoff=0)

    page = await client.reopen_verified(PID)

    assert page["properties"]["Status"]["status"]["name"] == "To Do"
    patch = calls[0][2]["properties"]
    assert patch["Status"]["status"]["name"] == "To Do"
    assert patch["Next Check"] == {"date": None}
    assert patch["Hold Reason"] == {"rich_text": []}


@pytest.mark.asyncio
async def test_set_thread_binding_verified_checks_thread_id_readback():
    def handler(req):
        if req.method == "PATCH":
            return httpx.Response(200, json=_page("To Do"))
        if req.method == "GET":
            return httpx.Response(200, json=_page("To Do", {
                "Discord Thread ID": {"type": "rich_text", "rich_text": [{"plain_text": "1523"}]},
                "Discord Thread URL": {"type": "url", "url": "https://discord.com/channels/g/c/1523"},
                "Thread Title Mode": {"type": "select", "select": {"name": "auto"}},
                "Thread Title Version": {"type": "number", "number": 1},
            }))
        raise AssertionError(req)

    client = NotionClient(api_key="secret", transport=httpx.MockTransport(handler), backoff=0)
    page = await client.set_thread_binding_verified(
        PID,
        thread_id="1523",
        thread_url="https://discord.com/channels/g/c/1523",
        title_mode="auto",
        title_version=1,
    )
    got = page["properties"]["Discord Thread ID"]["rich_text"][0]["plain_text"]
    assert got == "1523"


@pytest.mark.asyncio
async def test_set_thread_binding_verified_checks_url_readback():
    def handler(req):
        if req.method == "PATCH":
            return httpx.Response(200, json=_page("To Do"))
        if req.method == "GET":
            return httpx.Response(200, json=_page("To Do", {
                "Discord Thread ID": _text_prop("1523"),
                "Discord Thread URL": {"type": "url", "url": "https://discord.com/channels/g/c/wrong"},
            }))
        raise AssertionError(req)

    client = NotionClient(api_key="secret", transport=httpx.MockTransport(handler), backoff=0)
    with pytest.raises(Exception, match="Discord Thread URL read-back mismatch"):
        await client.set_thread_binding_verified(
            PID,
            thread_id="1523",
            thread_url="https://discord.com/channels/g/c/1523",
            title_mode="auto",
            title_version=1,
        )
