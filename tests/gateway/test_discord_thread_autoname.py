"""Tests for naming a freshly-engaged user-created Discord thread.

When a user manually creates a thread off a Hermes message and posts in it,
Hermes should name the thread from the user's first message immediately —
mirroring the auto-thread path's creation-time naming — so the thread title
aligns with the conversation even before (or without) an LLM-generated
session title.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

from tests.gateway.test_discord_send import _ensure_discord_mock

_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


def _adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    # Replace the persistent tracker with a plain set so tests never touch the
    # real ~/.hermes/discord_threads.json. The method under test only reads
    # membership via ``in``; participation marking happens at the call site.
    adapter._threads = set()
    return adapter


class TestDeriveThreadName:
    def test_strips_mentions_and_collapses_whitespace(self):
        adapter = _adapter()
        assert adapter._derive_thread_name("  <@123>  hello   world ") == "hello world"

    def test_truncates_to_80_with_ellipsis(self):
        adapter = _adapter()
        name = adapter._derive_thread_name("x" * 100)
        assert len(name) == 80
        assert name.endswith("...")

    def test_empty_after_stripping_returns_empty(self):
        adapter = _adapter()
        assert adapter._derive_thread_name("   <@123> <#456> ") == ""


class TestResolveThreadParent:
    def test_prefers_hydrated_parent(self):
        adapter = _adapter()
        parent = object()
        channel = SimpleNamespace(parent=parent, parent_id=999)
        assert adapter._resolve_thread_parent(channel) is parent

    def test_falls_back_to_parent_id_cache_lookup(self):
        adapter = _adapter()
        forum = object()
        adapter._client = SimpleNamespace(get_channel=lambda cid: forum if cid == 999 else None)
        channel = SimpleNamespace(parent=None, parent_id=999)
        assert adapter._resolve_thread_parent(channel) is forum

    def test_returns_channel_when_no_parent_resolvable(self):
        adapter = _adapter()
        adapter._client = SimpleNamespace(get_channel=lambda cid: None)
        channel = SimpleNamespace(parent=None, parent_id=None)
        assert adapter._resolve_thread_parent(channel) is channel


@pytest.mark.asyncio
class TestMaybeNameNewThread:
    async def test_renames_fresh_user_thread_from_first_message(self):
        adapter = _adapter()
        adapter.rename_thread = AsyncMock()

        await adapter._maybe_name_new_thread(
            "800", "面试评价这个事情", is_thread=True, auto_created=False
        )

        adapter.rename_thread.assert_awaited_once_with("800", "面试评价这个事情")

    async def test_skips_auto_created_thread(self):
        # Auto-threads are already named at creation time — don't double-rename.
        adapter = _adapter()
        adapter.rename_thread = AsyncMock()

        await adapter._maybe_name_new_thread(
            "800", "hello", is_thread=True, auto_created=True
        )

        adapter.rename_thread.assert_not_awaited()

    async def test_skips_forum_posts(self):
        # Forum post titles are authored by the user who opened the post —
        # renaming from the first reply would clobber a deliberate title.
        adapter = _adapter()
        adapter.rename_thread = AsyncMock()

        await adapter._maybe_name_new_thread(
            "800", "hello", is_thread=True, auto_created=False, is_forum=True
        )

        adapter.rename_thread.assert_not_awaited()

    async def test_skips_slash_command_first_message(self):
        adapter = _adapter()
        adapter.rename_thread = AsyncMock()

        await adapter._maybe_name_new_thread(
            "800", "/status", is_thread=True, auto_created=False, is_command=True
        )

        adapter.rename_thread.assert_not_awaited()

    async def test_skips_empty_message(self):
        adapter = _adapter()
        adapter.rename_thread = AsyncMock()

        await adapter._maybe_name_new_thread(
            "800", "   ", is_thread=True, auto_created=False
        )

        adapter.rename_thread.assert_not_awaited()

    async def test_skips_non_thread_messages(self):
        adapter = _adapter()
        adapter.rename_thread = AsyncMock()

        await adapter._maybe_name_new_thread(
            "800", "hello", is_thread=False, auto_created=False
        )

        adapter.rename_thread.assert_not_awaited()
