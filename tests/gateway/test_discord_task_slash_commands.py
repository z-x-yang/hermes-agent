"""Discord native slash commands for Notion task fast paths."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.Interaction = object

        class _FakeCommand:
            def __init__(self, *, name, description, callback, parent=None):
                self.name = name
                self.description = description
                self.callback = callback
                self.parent = parent

        class _FakeGroup:
            def __init__(self, *, name, description, parent=None):
                self.name = name
                self.description = description
                self.parent = parent
                self._children = {}
                if parent is not None:
                    parent.add_command(self)

            def add_command(self, cmd):
                self._children[cmd.name] = cmd

        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            autocomplete=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
            Command=_FakeCommand,
            Group=_FakeGroup,
        )
        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod
        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn
        return decorator

    def add_command(self, cmd):
        self.commands[cmd.name] = cmd

    def get_commands(self):
        return [SimpleNamespace(name=n) for n in self.commands]


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    adapter._notion_controller = SimpleNamespace(
        handle_slash_done=AsyncMock(),
        handle_slash_hold=AsyncMock(),
        handle_slash_snooze=AsyncMock(),
        handle_slash_reopen=AsyncMock(),
        handle_slash_bind=AsyncMock(),
    )
    return adapter


@pytest.mark.asyncio
async def test_registers_top_level_task_slash_commands(adapter):
    adapter._register_slash_commands()

    names = set(adapter._client.tree.commands)

    assert {"task-done", "task-hold", "task-snooze", "task-reopen", "task-bind"}.issubset(names)
    assert "task" not in names


@pytest.mark.asyncio
async def test_task_done_slash_auth_gate_before_controller(adapter):
    adapter._check_slash_authorization = AsyncMock(return_value=False)
    adapter._register_slash_commands()
    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))

    await adapter._client.tree.commands["task-done"](interaction)

    adapter._check_slash_authorization.assert_awaited_once_with(interaction, "/task-done")
    adapter._notion_controller.handle_slash_done.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "handler", "args"),
    [
        ("task-done", "handle_slash_done", ()),
        ("task-hold", "handle_slash_hold", ()),
        ("task-snooze", "handle_slash_snooze", ()),
        ("task-reopen", "handle_slash_reopen", ()),
        ("task-bind", "handle_slash_bind", ("https://notion.so/example",)),
    ],
)
async def test_task_slash_commands_dispatch_to_controller(adapter, command, handler, args):
    adapter._register_slash_commands()
    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))

    await adapter._client.tree.commands[command](interaction, *args)

    adapter._check_slash_authorization.assert_awaited_with(interaction, f"/{command}")
    getattr(adapter._notion_controller, handler).assert_awaited_once_with(interaction, *args)
