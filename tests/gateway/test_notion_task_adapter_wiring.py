import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.discord.adapter import DiscordAdapter, _standalone_send
from plugins.platforms.discord.notion_tasks import outbound
from gateway.config import PlatformConfig

PID = "1f17a58d229e816f839bef72f6f2ec72"


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    """Point the tracker's state file at tmp so construction reads no real file."""
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)


def _make_adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._client = MagicMock()
    adapter._allowed_user_ids = {"42"}
    adapter._allowed_role_ids = set()
    adapter._is_forum_parent = lambda c: False  # mock channel is not a forum
    return adapter


@pytest.mark.asyncio
async def test_send_attaches_task_view_when_link_present():
    adapter = _make_adapter()
    channel = MagicMock()
    sent = MagicMock()
    sent.id = 123
    channel.send = AsyncMock(return_value=sent)
    adapter._client.get_channel = MagicMock(return_value=channel)
    fake_view = MagicMock()
    adapter._notion_controller.render_view_for_text = AsyncMock(return_value=fake_view)

    res = await adapter.send("9001", f"[T](https://notion.so/{PID})")
    assert res.success is True
    adapter._notion_controller.render_view_for_text.assert_awaited_once()
    assert channel.send.call_args.kwargs.get("view") is fake_view


@pytest.mark.asyncio
async def test_send_attaches_task_view_for_app_notion_links():
    adapter = _make_adapter()
    channel = MagicMock()
    sent = MagicMock()
    sent.id = 123
    channel.send = AsyncMock(return_value=sent)
    adapter._client.get_channel = MagicMock(return_value=channel)
    fake_view = MagicMock()
    adapter._notion_controller.render_view_for_text = AsyncMock(return_value=fake_view)

    res = await adapter.send("9001", f"Notion: https://app.notion.com/p/Reply-to-Alice-{PID}")
    assert res.success is True
    adapter._notion_controller.render_view_for_text.assert_awaited_once()
    assert channel.send.call_args.kwargs.get("view") is fake_view


@pytest.mark.asyncio
async def test_send_skips_render_without_notion_substring():
    adapter = _make_adapter()
    channel = MagicMock()
    sent = MagicMock()
    sent.id = 1
    channel.send = AsyncMock(return_value=sent)
    adapter._client.get_channel = MagicMock(return_value=channel)
    adapter._notion_controller.render_view_for_text = AsyncMock()
    await adapter.send("9001", "plain message no link")
    adapter._notion_controller.render_view_for_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_thread_create_delegates_to_controller():
    adapter = _make_adapter()
    adapter._notion_controller.on_thread_opened = AsyncMock()
    thread = SimpleNamespace(id=7, starter_message=None)
    await adapter._handle_thread_create(thread)
    adapter._notion_controller.on_thread_opened.assert_awaited_once_with(thread)


# --- the standalone HTTP path (`hermes send` / cron) MUST attach the button ---

class _FakeResp:
    def __init__(self, status=200, data=None):
        self.status = status
        self._data = data if data is not None else {"id": "999"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._data

    async def text(self):
        return ""


class _FakeSession:
    posts: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, url, **kwargs):
        _FakeSession.posts.append({"url": url, **kwargs})
        return _FakeResp(200, {"id": "999"})

    def get(self, url, **kwargs):
        return _FakeResp(200, {"type": 0})  # not a forum channel


def _install_fake_aiohttp(monkeypatch):
    fake = ModuleType("aiohttp")
    fake.ClientSession = _FakeSession
    fake.ClientTimeout = lambda **k: None
    fake.FormData = MagicMock
    monkeypatch.setitem(sys.modules, "aiohttp", fake)
    _FakeSession.posts = []


@pytest.mark.asyncio
async def test_standalone_send_attaches_components_for_task_link(monkeypatch):
    _install_fake_aiohttp(monkeypatch)
    monkeypatch.setattr(outbound, "standalone_task_components",
                        AsyncMock(return_value=[{"type": 1, "components": [
                            {"type": 2, "style": 3, "label": "✓ 完成",
                             "custom_id": f"ntask:done:{PID}"}]}]))
    pconfig = SimpleNamespace(token="bot_token")
    res = await _standalone_send(pconfig, "9001", f"do [T](https://notion.so/{PID})")
    assert res.get("success") is True
    # the message POST carried the button components
    msg_posts = [p for p in _FakeSession.posts if "/messages" in p["url"]]
    assert msg_posts, "no message POST captured"
    payload = msg_posts[-1]["json"]
    assert payload["components"][0]["components"][0]["custom_id"] == f"ntask:done:{PID}"


@pytest.mark.asyncio
async def test_standalone_send_no_components_for_plain_message(monkeypatch):
    _install_fake_aiohttp(monkeypatch)
    monkeypatch.setattr(outbound, "standalone_task_components", AsyncMock(return_value=[]))
    pconfig = SimpleNamespace(token="bot_token")
    await _standalone_send(pconfig, "9001", "plain message")
    msg_posts = [p for p in _FakeSession.posts if "/messages" in p["url"]]
    assert msg_posts and "components" not in msg_posts[-1]["json"]
