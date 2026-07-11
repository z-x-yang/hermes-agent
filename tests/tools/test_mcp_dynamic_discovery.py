"""Tests for MCP dynamic tool discovery (notifications/tools/list_changed)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _register_server_tools
from tools.registry import ToolRegistry
from tools.tool_effects import ResultRetention, ToolEffect


def _make_mcp_tool(name: str, desc: str = "", **extra):
    return SimpleNamespace(name=name, description=desc, inputSchema=None, **extra)


class TestRegisterServerTools:
    """Tests for the extracted _register_server_tools helper."""

    @pytest.fixture
    def mock_registry(self):
        return ToolRegistry()

    def test_exposes_live_server_aliases(self, mock_registry):
        """Registered MCP tools are reachable via live raw-server aliases."""
        server = MCPServerTask("my_srv")
        server._tools = [_make_mcp_tool("my_tool", "desc")]
        server.session = MagicMock()
        from toolsets import resolve_toolset, validate_toolset

        with patch("tools.registry.registry", mock_registry):
            registered = _register_server_tools("my_srv", server, {})
            assert "mcp_my_srv_my_tool" in registered
            assert "mcp_my_srv_my_tool" in mock_registry.get_all_tool_names()
            assert validate_toolset("my_srv") is True
            assert "mcp_my_srv_my_tool" in resolve_toolset("my_srv")

    def test_mcp_identity_binds_server_info_and_strips_transport_secrets(self, mock_registry):
        server = MCPServerTask("secure_srv")
        server._tools = [
            _make_mcp_tool(
                "lookup",
                annotations=SimpleNamespace(readOnlyHint=True),
            )
        ]
        server.session = MagicMock()
        server.initialize_result = SimpleNamespace(
            serverInfo=SimpleNamespace(name="remote-index", version="2.4.1")
        )
        config = {
            "url": "https://user:password@example.test/mcp?token=top-secret",
            "headers": {"Authorization": "Bearer top-secret"},
        }

        with patch("tools.registry.registry", mock_registry):
            _register_server_tools("secure_srv", server, config)

        entry = mock_registry.get_entry("mcp_secure_srv_lookup")
        identity = entry.policy_descriptor.source_identity
        assert entry.policy_descriptor.effects == frozenset({ToolEffect.UNKNOWN})
        assert "remote-index" not in identity
        assert "2.4.1" not in identity
        assert "password" not in identity
        assert "top-secret" not in identity
        assert "Bearer" not in identity

        upgraded_registry = ToolRegistry()
        server.initialize_result = SimpleNamespace(
            serverInfo=SimpleNamespace(name="remote-index", version="2.4.2")
        )
        with patch("tools.registry.registry", upgraded_registry):
            _register_server_tools("secure_srv", server, config)
        upgraded_identity = upgraded_registry.get_entry(
            "mcp_secure_srv_lookup"
        ).policy_descriptor.source_identity
        assert upgraded_identity != identity

    def test_notion_ai_ask_is_read_only_only_when_mode_is_readonly(self, mock_registry):
        server = MCPServerTask("notion-ai")
        server._tools = [_make_mcp_tool("notion_ai_ask")]
        server.session = MagicMock()

        with patch("tools.registry.registry", mock_registry):
            _register_server_tools("notion-ai", server, {})

        descriptor = mock_registry.get_entry(
            "mcp_notion_ai_notion_ai_ask"
        ).policy_descriptor
        assert descriptor.argument_resolver is not None
        assert descriptor.argument_resolver({"mode": "readonly"}) == frozenset(
            {ToolEffect.READ_REMOTE}
        )
        assert descriptor.argument_resolver({"mode": "write"}) == frozenset(
            {ToolEffect.WRITE_REMOTE}
        )
        assert descriptor.argument_resolver({}) == frozenset(
            {ToolEffect.WRITE_REMOTE}
        )
        assert descriptor.retention is ResultRetention.HANDLE_ONLY

    def test_only_named_apple_mail_read_tools_gain_read_effects(self, mock_registry):
        server = MCPServerTask("apple-mail")
        server._tools = [
            _make_mcp_tool("search_messages"),
            _make_mcp_tool("fetch_attachment"),
            _make_mcp_tool("send_email"),
            _make_mcp_tool("delete_message"),
        ]
        server.session = MagicMock()

        with patch("tools.registry.registry", mock_registry):
            _register_server_tools("apple-mail", server, {})

        for name in ("search_messages", "fetch_attachment"):
            descriptor = mock_registry.get_entry(
                f"mcp_apple_mail_{name}"
            ).policy_descriptor
            assert descriptor.effects == frozenset({ToolEffect.READ_REMOTE})
            assert descriptor.retention is ResultRetention.HANDLE_ONLY
        for name in ("send_email", "delete_message"):
            descriptor = mock_registry.get_entry(
                f"mcp_apple_mail_{name}"
            ).policy_descriptor
            assert descriptor.effects == frozenset({ToolEffect.UNKNOWN})
            assert descriptor.retention is ResultRetention.DEFAULT

    def test_explore_policy_exposes_existing_data_reads_but_not_mail_writes(
        self, mock_registry
    ):
        from agent.subagent_tool_policy import build_child_tool_policy
        from tools.subagent_profiles import get_subagent_profile
        from tools.tool_effects import build_authority_snapshot

        notion = MCPServerTask("notion-ai")
        notion._tools = [_make_mcp_tool("notion_ai_ask")]
        notion.session = MagicMock()
        mail = MCPServerTask("apple-mail")
        mail._tools = [
            _make_mcp_tool("search_messages"),
            _make_mcp_tool("get_message"),
            _make_mcp_tool("send_email"),
        ]
        mail.session = MagicMock()

        with patch("tools.registry.registry", mock_registry):
            _register_server_tools("notion-ai", notion, {})
            _register_server_tools("apple-mail", mail, {})
            for entry in mock_registry._tools.values():
                entry.check_fn = lambda: True
            identities = {
                entry.policy_identity
                for entry in mock_registry._tools.values()
            }
            snapshot = build_authority_snapshot(
                identities,
                registry_generation=mock_registry._generation,
            )
            profile = get_subagent_profile("Explore")
            policy = build_child_tool_policy(
                child=SimpleNamespace(_parent_tool_authority_snapshot=snapshot),
                parent=SimpleNamespace(_parent_tool_authority_snapshot=snapshot),
                profile_name="Explore",
                profile_allowed_names=profile.allowed_tool_names,
            )

        assert policy.allowed_names is not None
        assert "mcp_notion_ai_notion_ai_ask" in policy.allowed_names
        assert "mcp_apple_mail_search_messages" in policy.allowed_names
        assert "mcp_apple_mail_get_message" in policy.allowed_names
        assert "mcp_apple_mail_send_email" not in policy.allowed_names

    def test_same_name_refresh_gets_new_policy_identity(self, mock_registry):
        server = MCPServerTask("refresh_srv")
        server._tools = [_make_mcp_tool("lookup")]
        server.session = MagicMock()
        server.initialize_result = SimpleNamespace(
            serverInfo=SimpleNamespace(name="remote-index", version="1.0")
        )
        with patch("tools.registry.registry", mock_registry):
            _register_server_tools("refresh_srv", server, {"command": "safe-server"})
            first = mock_registry.get_entry("mcp_refresh_srv_lookup")
            _register_server_tools("refresh_srv", server, {"command": "safe-server"})
            second = mock_registry.get_entry("mcp_refresh_srv_lookup")

        assert second.entry_generation > first.entry_generation
        assert second.policy_identity != first.policy_identity

    def test_missing_server_version_is_explicit_unknown(self, mock_registry):
        server = MCPServerTask("legacy_srv")
        server._tools = [_make_mcp_tool("lookup")]
        server.session = MagicMock()
        server.initialize_result = SimpleNamespace(
            serverInfo=SimpleNamespace(name="legacy-index")
        )

        with patch("tools.registry.registry", mock_registry):
            _register_server_tools("legacy_srv", server, {"command": "safe-server"})

        identity = mock_registry.get_entry(
            "mcp_legacy_srv_lookup"
        ).policy_descriptor.source_identity
        assert "legacy-index" not in identity
        assert "unknown" not in identity

        explicit_unknown_registry = ToolRegistry()
        server.initialize_result = SimpleNamespace(
            serverInfo=SimpleNamespace(name="legacy-index", version="unknown")
        )
        with patch("tools.registry.registry", explicit_unknown_registry):
            _register_server_tools("legacy_srv", server, {"command": "safe-server"})
        explicit_unknown_identity = explicit_unknown_registry.get_entry(
            "mcp_legacy_srv_lookup"
        ).policy_descriptor.source_identity
        assert explicit_unknown_identity == identity

        versioned_registry = ToolRegistry()
        server.initialize_result = SimpleNamespace(
            serverInfo=SimpleNamespace(name="legacy-index", version="1.0")
        )
        with patch("tools.registry.registry", versioned_registry):
            _register_server_tools("legacy_srv", server, {"command": "safe-server"})
        versioned_identity = versioned_registry.get_entry(
            "mcp_legacy_srv_lookup"
        ).policy_descriptor.source_identity
        assert versioned_identity != identity


class TestRefreshTools:
    """Tests for MCPServerTask._refresh_tools nuke-and-repave cycle."""

    @pytest.fixture
    def mock_registry(self):
        return ToolRegistry()

    @pytest.mark.asyncio
    async def test_nuke_and_repave(self, mock_registry):
        """Old tools are removed and new tools registered on refresh."""
        server = MCPServerTask("live_srv")
        server._refresh_lock = asyncio.Lock()
        server._config = {}
        from toolsets import resolve_toolset

        # Seed initial state: one old tool registered
        mock_registry.register(
            name="mcp_live_srv_old_tool", toolset="mcp-live_srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )
        server._registered_tool_names = ["mcp_live_srv_old_tool"]

        # New tool list from server
        new_tool = _make_mcp_tool("new_tool", "new behavior")
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(tools=[new_tool])
            )
        )

        with patch("tools.registry.registry", mock_registry):
            await server._refresh_tools()
            assert "mcp_live_srv_old_tool" not in mock_registry.get_all_tool_names()
            assert "mcp_live_srv_old_tool" not in resolve_toolset("live_srv")
            assert "mcp_live_srv_new_tool" in mock_registry.get_all_tool_names()
            assert "mcp_live_srv_new_tool" in resolve_toolset("live_srv")
            assert server._registered_tool_names == ["mcp_live_srv_new_tool"]


class TestMessageHandler:
    """Tests for MCPServerTask._make_message_handler dispatch."""

    @pytest.mark.asyncio
    async def test_dispatches_tool_list_changed(self):
        from tools.mcp_tool import _MCP_NOTIFICATION_TYPES
        if not _MCP_NOTIFICATION_TYPES:
            pytest.skip("MCP SDK ToolListChangedNotification not available")

        from mcp.types import ServerNotification, ToolListChangedNotification

        server = MCPServerTask("notif_srv")
        # Product now schedules the refresh as a background task (see
        # _schedule_tools_refresh in mcp_tool.py ~L918) rather than awaiting
        # it directly, to avoid wedging the stdio JSON-RPC stream. Patch at
        # the scheduler seam so we can still assert dispatch happened without
        # reaching into asyncio.create_task internals.
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule:
            handler = server._make_message_handler()
            notification = ServerNotification(
                root=ToolListChangedNotification(method="notifications/tools/list_changed")
            )
            await handler(notification)
            mock_schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_ignores_exceptions_and_other_messages(self):
        server = MCPServerTask("notif_srv")
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule:
            handler = server._make_message_handler()
            # Exceptions should not trigger refresh
            await handler(RuntimeError("connection dead"))
            # Unknown message types should not trigger refresh
            await handler({"jsonrpc": "2.0", "result": "ok"})
            mock_schedule.assert_not_called()


class TestDeregister:
    """Tests for ToolRegistry.deregister."""

    def test_removes_tool(self):
        reg = ToolRegistry()
        reg.register(name="foo", toolset="ts1", schema={}, handler=lambda x: x)
        assert "foo" in reg.get_all_tool_names()
        reg.deregister("foo")
        assert "foo" not in reg.get_all_tool_names()

    def test_cleans_up_toolset_check(self):
        reg = ToolRegistry()
        check = lambda: True  # noqa: E731
        reg.register(name="foo", toolset="ts1", schema={}, handler=lambda x: x, check_fn=check)
        assert reg.is_toolset_available("ts1")
        reg.deregister("foo")
        # Toolset check should be gone since no tools remain
        assert "ts1" not in reg._toolset_checks

    def test_preserves_toolset_check_if_other_tools_remain(self):
        reg = ToolRegistry()
        check = lambda: True  # noqa: E731
        reg.register(name="foo", toolset="ts1", schema={}, handler=lambda x: x, check_fn=check)
        reg.register(name="bar", toolset="ts1", schema={}, handler=lambda x: x)
        reg.deregister("foo")
        # bar still in ts1, so check should remain
        assert "ts1" in reg._toolset_checks

    def test_removes_toolset_alias_when_last_tool_is_removed(self):
        reg = ToolRegistry()
        reg.register(name="foo", toolset="mcp-srv", schema={}, handler=lambda x: x)
        reg.register_toolset_alias("srv", "mcp-srv")

        reg.deregister("foo")

        assert reg.get_toolset_alias_target("srv") is None

    def test_preserves_toolset_alias_while_toolset_still_exists(self):
        reg = ToolRegistry()
        reg.register(name="foo", toolset="mcp-srv", schema={}, handler=lambda x: x)
        reg.register(name="bar", toolset="mcp-srv", schema={}, handler=lambda x: x)
        reg.register_toolset_alias("srv", "mcp-srv")

        reg.deregister("foo")

        assert reg.get_toolset_alias_target("srv") == "mcp-srv"

    def test_noop_for_unknown_tool(self):
        reg = ToolRegistry()
        reg.deregister("nonexistent")  # Should not raise
