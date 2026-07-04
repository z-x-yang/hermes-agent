import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.discord.adapter import DiscordAdapter, _standalone_send
from plugins.platforms.discord.notion_tasks import outbound
from gateway.config import PlatformConfig

PID = "1f17a58d229e816f839bef72f6f2ec72"

CARD_EMBED = {"title": "📋 任务", "description": f"1️⃣ T", "color": 0x4E8CD8}


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
async def test_send_attaches_task_card_when_link_present():
    adapter = _make_adapter()
    channel = MagicMock()
    sent = MagicMock()
    sent.id = 123
    channel.send = AsyncMock(return_value=sent)
    adapter._client.get_channel = MagicMock(return_value=channel)
    fake_view, fake_embed = MagicMock(), MagicMock()
    adapter._notion_controller.render_send_attachments = AsyncMock(
        return_value=(fake_view, fake_embed))

    res = await adapter.send("9001", f"[T](https://notion.so/{PID})")
    assert res.success is True
    adapter._notion_controller.render_send_attachments.assert_awaited_once()
    assert channel.send.call_args.kwargs.get("view") is fake_view
    assert channel.send.call_args.kwargs.get("embed") is fake_embed


@pytest.mark.asyncio
async def test_send_attaches_task_card_for_app_notion_links():
    adapter = _make_adapter()
    channel = MagicMock()
    sent = MagicMock()
    sent.id = 123
    channel.send = AsyncMock(return_value=sent)
    adapter._client.get_channel = MagicMock(return_value=channel)
    fake_view, fake_embed = MagicMock(), MagicMock()
    adapter._notion_controller.render_send_attachments = AsyncMock(
        return_value=(fake_view, fake_embed))

    res = await adapter.send("9001", f"Notion: https://app.notion.com/p/Reply-to-Alice-{PID}")
    assert res.success is True
    adapter._notion_controller.render_send_attachments.assert_awaited_once()
    assert channel.send.call_args.kwargs.get("view") is fake_view
    assert channel.send.call_args.kwargs.get("embed") is fake_embed


@pytest.mark.asyncio
async def test_send_skips_render_without_notion_substring():
    adapter = _make_adapter()
    channel = MagicMock()
    sent = MagicMock()
    sent.id = 1
    channel.send = AsyncMock(return_value=sent)
    adapter._client.get_channel = MagicMock(return_value=channel)
    adapter._notion_controller.render_send_attachments = AsyncMock()
    await adapter.send("9001", "plain message no link")
    adapter._notion_controller.render_send_attachments.assert_not_awaited()
    assert "embed" not in channel.send.call_args.kwargs


@pytest.mark.asyncio
async def test_send_forum_starter_carries_card():
    adapter = _make_adapter()
    adapter._is_forum_parent = lambda c: True
    forum = MagicMock()
    thread = MagicMock()
    thread.id = 555
    thread.message = MagicMock(id=666)
    forum.create_thread = AsyncMock(return_value=thread)
    adapter._client.get_channel = MagicMock(return_value=forum)
    fake_view, fake_embed = MagicMock(), MagicMock()
    adapter._notion_controller.render_send_attachments = AsyncMock(
        return_value=(fake_view, fake_embed))

    res = await adapter.send("9001", f"[T](https://notion.so/{PID})")
    assert res.success is True
    kwargs = forum.create_thread.call_args.kwargs
    assert kwargs["view"] is fake_view
    assert kwargs["embed"] is fake_embed


@pytest.mark.asyncio
async def test_on_thread_create_delegates_to_controller():
    adapter = _make_adapter()
    adapter._notion_controller.on_thread_opened = AsyncMock()
    thread = SimpleNamespace(id=7, starter_message=None)
    await adapter._handle_thread_create(thread)
    adapter._notion_controller.on_thread_opened.assert_awaited_once_with(thread)


# --- the standalone HTTP path (`hermes send` / cron) MUST attach the card ---

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

class _FakeForumSession(_FakeSession):
    def get(self, url, **kwargs):
        return _FakeResp(200, {"type": 15})  # forum channel


def _install_fake_aiohttp(monkeypatch, session=_FakeSession):
    fake = ModuleType("aiohttp")
    fake.ClientSession = session
    fake.ClientTimeout = lambda **k: None
    fake.FormData = MagicMock
    monkeypatch.setitem(sys.modules, "aiohttp", fake)
    _FakeSession.posts = []


@pytest.mark.asyncio
async def test_standalone_send_attaches_card_for_task_link(monkeypatch):
    _install_fake_aiohttp(monkeypatch)
    components = [{"type": 1, "components": [
        {"type": 2, "style": 3, "label": "✓ 1", "custom_id": f"ntask:done:{PID}"}]}]
    monkeypatch.setattr(outbound, "standalone_task_payload",
                        AsyncMock(return_value=(components, CARD_EMBED)))
    pconfig = SimpleNamespace(token="bot_token")
    res = await _standalone_send(pconfig, "9001", f"do [T](https://notion.so/{PID})")
    assert res.get("success") is True
    # the message POST carried the button components AND the card embed
    msg_posts = [p for p in _FakeSession.posts if "/messages" in p["url"]]
    assert msg_posts, "no message POST captured"
    payload = msg_posts[-1]["json"]
    assert payload["components"][0]["components"][0]["custom_id"] == f"ntask:done:{PID}"
    assert payload["embeds"] == [CARD_EMBED]


@pytest.mark.asyncio
async def test_standalone_send_no_card_for_plain_message(monkeypatch):
    _install_fake_aiohttp(monkeypatch)
    monkeypatch.setattr(outbound, "standalone_task_payload",
                        AsyncMock(return_value=([], None)))
    pconfig = SimpleNamespace(token="bot_token")
    await _standalone_send(pconfig, "9001", "plain message")
    msg_posts = [p for p in _FakeSession.posts if "/messages" in p["url"]]
    assert msg_posts
    assert "components" not in msg_posts[-1]["json"]
    assert "embeds" not in msg_posts[-1]["json"]


@pytest.mark.asyncio
async def test_standalone_send_forum_starter_carries_card(monkeypatch):
    _install_fake_aiohttp(monkeypatch, session=_FakeForumSession)
    components = [{"type": 1, "components": [
        {"type": 2, "style": 3, "label": "✓ 1", "custom_id": f"ntask:done:{PID}"}]}]
    monkeypatch.setattr(outbound, "standalone_task_payload",
                        AsyncMock(return_value=(components, CARD_EMBED)))
    pconfig = SimpleNamespace(token="bot_token")
    # fresh chat_id: the forum probe result is cached per-channel in-process,
    # and earlier tests already cached "9001" as not-a-forum
    res = await _standalone_send(pconfig, "9002", f"do [T](https://notion.so/{PID})")
    assert res.get("success") is True
    thread_posts = [p for p in _FakeSession.posts if p["url"].endswith("/threads")]
    assert thread_posts, "no forum thread POST captured"
    starter = thread_posts[-1]["json"]["message"]
    assert starter["components"] == components
    assert starter["embeds"] == [CARD_EMBED]
