from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.subagent_tool_policy import (
    ToolAuthorizationError,
    apply_tool_policy_to_agent,
    authorize_subagent_call,
    build_child_tool_policy,
)
from tools.mcp_tool import MCPServerTask, _register_server_tools
from tools.registry import ToolRegistry
from tools.subagent_profiles import get_subagent_profile
from tools.tool_effects import ResultRetention, ToolEffect, build_authority_snapshot


def _mcp_tool(name: str):
    return SimpleNamespace(name=name, description=name, inputSchema=None)


def test_existing_notion_and_mail_readers_form_the_read_profile_data_plane():
    registry = ToolRegistry()
    notion = MCPServerTask("notion-ai")
    notion._tools = [_mcp_tool("notion_ai_ask")]
    notion.session = MagicMock()
    mail = MCPServerTask("apple-mail")
    mail._tools = [
        _mcp_tool("search_messages"),
        _mcp_tool("get_message"),
        _mcp_tool("fetch_attachment"),
        _mcp_tool("send_email"),
        _mcp_tool("reply_to_message"),
        _mcp_tool("forward_message"),
        _mcp_tool("move_message"),
        _mcp_tool("delete_message"),
        _mcp_tool("flag_message"),
        _mcp_tool("mark_as_read"),
        _mcp_tool("save_attachment"),
    ]
    mail.session = MagicMock()

    with patch("tools.registry.registry", registry):
        _register_server_tools("notion-ai", notion, {})
        _register_server_tools("apple-mail", mail, {})
        for entry in registry._tools.values():
            entry.check_fn = lambda: True
        snapshot = build_authority_snapshot(
            {entry.policy_identity for entry in registry._tools.values()},
            registry_generation=registry._generation,
        )
        profile = get_subagent_profile("Explore")
        policy = build_child_tool_policy(
            child=SimpleNamespace(_parent_tool_authority_snapshot=snapshot),
            parent=SimpleNamespace(_parent_tool_authority_snapshot=snapshot),
            profile_name="Explore",
            profile_allowed_names=profile.allowed_tool_names,
        )
        child = SimpleNamespace(
            valid_tool_names=set(registry._tools),
            tools=[
                {"type": "function", "function": entry.schema}
                for entry in registry._tools.values()
            ],
        )
        apply_tool_policy_to_agent(child, policy)

        readonly_call = authorize_subagent_call(
            child,
            "mcp_notion_ai_notion_ai_ask",
            {"prompt": "lookup", "mode": "readonly"},
        )
        assert readonly_call.effects == frozenset({ToolEffect.READ_REMOTE})

        for args in (
            {"prompt": "mutate", "mode": "write"},
            {"prompt": "missing mode"},
        ):
            with pytest.raises(ToolAuthorizationError):
                authorize_subagent_call(
                    child,
                    "mcp_notion_ai_notion_ai_ask",
                    args,
                )

        mail_write_names = (
            "mcp_apple_mail_send_email",
            "mcp_apple_mail_reply_to_message",
            "mcp_apple_mail_forward_message",
            "mcp_apple_mail_move_message",
            "mcp_apple_mail_delete_message",
            "mcp_apple_mail_flag_message",
            "mcp_apple_mail_mark_as_read",
            "mcp_apple_mail_save_attachment",
        )
        for name in mail_write_names:
            with pytest.raises(ToolAuthorizationError):
                authorize_subagent_call(child, name, {"id": "message-1"})

        assert notion.session.call_tool.call_count == 0
        assert mail.session.call_tool.call_count == 0

    assert policy.allowed_names is not None
    assert {
        "mcp_notion_ai_notion_ai_ask",
        "mcp_apple_mail_search_messages",
        "mcp_apple_mail_get_message",
        "mcp_apple_mail_fetch_attachment",
    }.issubset(policy.allowed_names)
    assert "mcp_apple_mail_send_email" not in policy.allowed_names
    assert "mcp_apple_mail_delete_message" not in policy.allowed_names
    assert "mode=readonly" in profile.system_instructions
    assert "Never send, reply, forward, move, delete, flag, or mark mail" in (
        profile.system_instructions
    )

    notion_descriptor = registry.get_entry(
        "mcp_notion_ai_notion_ai_ask"
    ).policy_descriptor
    assert notion_descriptor.argument_resolver is not None
    assert notion_descriptor.argument_resolver({"mode": "readonly"}) == frozenset(
        {ToolEffect.READ_REMOTE}
    )
    assert notion_descriptor.argument_resolver({"mode": "write"}) == frozenset(
        {ToolEffect.WRITE_REMOTE}
    )
    assert notion_descriptor.retention is ResultRetention.HANDLE_ONLY

    for name in (
        "mcp_apple_mail_search_messages",
        "mcp_apple_mail_get_message",
        "mcp_apple_mail_fetch_attachment",
    ):
        descriptor = registry.get_entry(name).policy_descriptor
        assert descriptor.effects == frozenset({ToolEffect.READ_REMOTE})
        assert descriptor.retention is ResultRetention.HANDLE_ONLY

    for name in ("mcp_apple_mail_send_email", "mcp_apple_mail_delete_message"):
        descriptor = registry.get_entry(name).policy_descriptor
        assert descriptor.effects == frozenset({ToolEffect.UNKNOWN})
        assert descriptor.retention is ResultRetention.DEFAULT
