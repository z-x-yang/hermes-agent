import json

import httpx
import pytest

from plugins.platforms.discord.notion_tasks.notion_client import NotionClient, NotionError


def _client(handler, tmp_path):
    return NotionClient(api_key="ntn_test", transport=httpx.MockTransport(handler), backoff=0)


@pytest.mark.asyncio
async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    c = NotionClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with pytest.raises(NotionError):
        await c.get_page("p1")


@pytest.mark.asyncio
async def test_get_page_sends_bearer_and_version(tmp_path):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["ver"] = request.headers.get("notion-version")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"id": "p1", "parent": {}})

    c = _client(handler, tmp_path)
    page = await c.get_page("p1")
    assert page["id"] == "p1"
    assert seen["auth"] == "Bearer ntn_test"
    assert seen["ver"] == "2025-09-03"
    assert seen["url"].endswith("/v1/pages/p1")


@pytest.mark.asyncio
async def test_set_status_patches_properties(tmp_path):
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "p1"})

    c = _client(handler, tmp_path)
    await c.set_status("p1", "Done", "select")
    assert seen["method"] == "PATCH"
    assert seen["body"] == {"properties": {"Status": {"select": {"name": "Done"}}}}


@pytest.mark.asyncio
async def test_retries_then_raises_on_persistent_5xx(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(502, text="bad gateway")

    c = _client(handler, tmp_path)
    with pytest.raises(NotionError):
        await c.get_page("p1")
    assert calls["n"] == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_does_not_retry_on_404(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404, json={"message": "not found"})

    c = _client(handler, tmp_path)
    with pytest.raises(NotionError):
        await c.get_page("p1")
    assert calls["n"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_fetch_a_real_task_structure():
    """实测：从 Tasks 库查一个 page，打印 parent 与 Status 结构，确认判定/读取适配。

    The hermetic conftest fixture scrubs NOTION_API_KEY from os.environ, so read
    the real integration token straight from ~/.hermes/.env and pass it through.
    """
    import re
    from pathlib import Path
    from plugins.platforms.discord.notion_tasks import detection as d

    env_text = (Path.home() / ".hermes" / ".env").read_text(encoding="utf-8")
    m = re.search(r"^NOTION_API_KEY=(\S+)", env_text, re.M)
    assert m, "NOTION_API_KEY not found in ~/.hermes/.env"
    c = NotionClient(api_key=m.group(1))
    res = await c._request("POST", "/data_sources/1f17a58d-229e-8144-96f3-000b99bdcf95/query",
                           json_body={"page_size": 1})
    results = res.get("results") or []
    assert results, "Tasks data source query returned nothing"
    page = results[0]
    print("PARENT:", json.dumps(page.get("parent"), ensure_ascii=False))
    print("STATUS:", json.dumps((page.get("properties") or {}).get("Status"), ensure_ascii=False))
    # parent reports both database_id and data_source_id; matching either is enough
    assert d.is_task_page(page, {"1f17a58d229e816f839bef72f6f2ec72",
                                 "1f17a58d229e814496f3000b99bdcf95"}), "判定逻辑需按打印的 parent 适配"
    value, kind = d.read_status(page)
    assert kind in ("select", "status"), f"Status kind={kind}; 需按打印适配"
