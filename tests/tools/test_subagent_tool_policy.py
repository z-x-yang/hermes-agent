from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.subagent_tool_policy import (
    ToolNamePolicy,
    apply_tool_policy_to_agent,
    tool_policy_block_message,
)
from agent.tool_executor import (
    execute_tool_calls_concurrent,
    execute_tool_calls_sequential,
)
from run_agent import AIAgent
from tools.delegate_tool import _build_child_agent
from tools.subagent_profiles import get_subagent_profile


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": name, "parameters": {}},
    }


def _parent_agent() -> MagicMock:
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _child_agent() -> MagicMock:
    child = MagicMock()
    child.tools = [
        _tool("read_file"),
        _tool("write_file"),
        _tool("terminal"),
        _tool("mcp_apple_mail_send_email"),
    ]
    child.valid_tool_names = {
        "read_file",
        "write_file",
        "terminal",
        "mcp_apple_mail_send_email",
    }
    child._skip_mcp_refresh = False
    child._session_init_model_config = None
    child.session_id = "child-session"
    return child


def test_explore_definitions_exclude_write_and_external_tools():
    agent = SimpleNamespace(
        tools=[
            _tool("read_file"),
            _tool("write_file"),
            _tool("terminal"),
            _tool("mcp_apple_mail_send_email"),
            _tool("cronjob"),
        ],
        valid_tool_names={
            "read_file",
            "write_file",
            "terminal",
            "mcp_apple_mail_send_email",
            "cronjob",
        },
        _skip_mcp_refresh=False,
    )
    policy = ToolNamePolicy(allowed_names=frozenset({"read_file"}))

    apply_tool_policy_to_agent(agent, policy)

    assert agent.valid_tool_names == {"read_file"}
    assert [tool["function"]["name"] for tool in agent.tools] == ["read_file"]
    assert agent._subagent_tool_policy is policy
    assert agent._skip_mcp_refresh is True


def test_direct_hallucinated_tool_name_is_execution_blocked():
    agent = SimpleNamespace(
        _subagent_tool_policy=ToolNamePolicy(
            allowed_names=frozenset({"read_file"})
        )
    )

    message = tool_policy_block_message(agent, "write_file")

    assert "blocked by subagent capability policy" in message


def test_general_purpose_blocks_named_external_tools_but_keeps_raw_shell():
    profile = get_subagent_profile("general-purpose")
    policy = ToolNamePolicy(allowed_names=profile.allowed_tool_names)
    agent = SimpleNamespace(_subagent_tool_policy=policy)

    for repo_local_name in ("write_file", "patch", "terminal", "process"):
        assert tool_policy_block_message(agent, repo_local_name) is None
    assert tool_policy_block_message(agent, "mcp_notion_ai_notion_ai_ask")


def test_profiled_child_filters_visible_and_runtime_tool_names():
    parent = _parent_agent()
    child = _child_agent()

    with patch("run_agent.AIAgent", return_value=child):
        built = _build_child_agent(
            task_index=0,
            goal="Inspect the repository",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            profile=get_subagent_profile("Explore"),
        )

    assert built is child
    assert child.valid_tool_names == {"read_file"}
    assert [tool["function"]["name"] for tool in child.tools] == ["read_file"]
    assert child._subagent_tool_policy.allowed_names == frozenset(
        {"read_file", "search_files", "web_search", "web_extract"}
    )
    assert child._skip_mcp_refresh is True


def test_legacy_child_has_no_hard_allowlist_policy():
    parent = _parent_agent()
    child = _child_agent()

    with patch("run_agent.AIAgent", return_value=child):
        _build_child_agent(
            task_index=0,
            goal="Legacy delegation",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            profile=None,
        )

    assert "_subagent_tool_policy" not in child.__dict__
    assert child.valid_tool_names == {
        "read_file",
        "write_file",
        "terminal",
        "mcp_apple_mail_send_email",
    }
    assert child._skip_mcp_refresh is False


@pytest.fixture
def executor_agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[
                _tool("read_file"),
                _tool("write_file"),
                _tool("tool_call"),
            ],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    apply_tool_policy_to_agent(
        agent,
        ToolNamePolicy(allowed_names=frozenset({"read_file"})),
    )
    return agent


def _assistant_tool_call(name: str, arguments: dict, call_id: str = "call-1"):
    return SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(arguments),
                ),
            )
        ]
    )


def _capture_terminal_events(monkeypatch):
    hook_calls = []
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
    )
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    return hook_calls


def _assert_capability_block(messages, hook_calls, *, tool_name: str, call_id: str):
    assert len(messages) == 1
    assert messages[0]["tool_call_id"] == call_id
    assert "blocked by subagent capability policy" in messages[0]["content"]
    post_call = next(call for call in hook_calls if call[0] == "post_tool_call")
    assert post_call[1]["tool_name"] == tool_name
    assert post_call[1]["tool_call_id"] == call_id
    assert post_call[1]["status"] == "blocked"
    assert post_call[1]["error_type"] == "subagent_capability_block"


def test_concurrent_executor_blocks_denied_direct_tool_before_hooks_and_invoke(
    executor_agent,
    monkeypatch,
):
    messages = []
    hook_calls = _capture_terminal_events(monkeypatch)
    executor_agent._invoke_tool = MagicMock(
        side_effect=AssertionError("denied tool reached _invoke_tool")
    )
    executor_agent._tool_guardrails.before_call = MagicMock(
        side_effect=AssertionError("capability block must precede guardrails")
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_pre_tool_call_block_message",
        MagicMock(side_effect=AssertionError("capability block must precede plugins")),
    )

    execute_tool_calls_concurrent(
        executor_agent,
        _assistant_tool_call(
            "write_file",
            {"path": "denied.txt", "content": "no"},
            "concurrent-denied",
        ),
        messages,
        "task-1",
    )

    executor_agent._invoke_tool.assert_not_called()
    executor_agent._tool_guardrails.before_call.assert_not_called()
    _assert_capability_block(
        messages,
        hook_calls,
        tool_name="write_file",
        call_id="concurrent-denied",
    )


def test_sequential_executor_blocks_hallucinated_tool_before_hooks_and_invoke(
    executor_agent,
    monkeypatch,
):
    messages = []
    hook_calls = _capture_terminal_events(monkeypatch)
    executor_agent._invoke_tool = MagicMock(
        side_effect=AssertionError("denied tool reached _invoke_tool")
    )
    executor_agent._tool_guardrails.before_call = MagicMock(
        side_effect=AssertionError("capability block must precede guardrails")
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_pre_tool_call_block_message",
        MagicMock(side_effect=AssertionError("capability block must precede plugins")),
    )

    execute_tool_calls_sequential(
        executor_agent,
        _assistant_tool_call(
            "mcp_fake_send_money",
            {"amount": 100},
            "sequential-denied",
        ),
        messages,
        "task-1",
    )

    executor_agent._invoke_tool.assert_not_called()
    executor_agent._tool_guardrails.before_call.assert_not_called()
    _assert_capability_block(
        messages,
        hook_calls,
        tool_name="mcp_fake_send_money",
        call_id="sequential-denied",
    )


def test_tool_call_bridge_blocks_denied_underlying_external_tool_after_unwrap(
    executor_agent,
    monkeypatch,
):
    external_tool_name = "mcp_apple_mail_send_email"

    class ExternalToolManager:
        def get_all_tool_schemas(self):
            return [
                {
                    "name": external_tool_name,
                    "description": "Send external email",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]

    executor_agent.enabled_toolsets = None
    executor_agent._memory_manager = ExternalToolManager()
    # Keep this focused on the real agent-local Tool Search resolver without
    # rebuilding the process-global registry (which can probe optional tools).
    monkeypatch.setattr(
        "agent.tool_executor._tool_search_scoped_names",
        lambda agent: frozenset(),
    )
    executor_agent._invoke_tool = MagicMock(
        side_effect=AssertionError("unwrapped denied tool reached _invoke_tool")
    )
    executor_agent._tool_guardrails.before_call = MagicMock(
        side_effect=AssertionError("capability block must precede guardrails")
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_pre_tool_call_block_message",
        MagicMock(side_effect=AssertionError("capability block must precede plugins")),
    )
    hook_calls = _capture_terminal_events(monkeypatch)
    messages = []

    execute_tool_calls_concurrent(
        executor_agent,
        _assistant_tool_call(
            "tool_call",
            {
                "name": external_tool_name,
                "arguments": {
                    "to": ["outside@example.com"],
                    "subject": "Denied",
                    "body": "No external side effects",
                },
            },
            "bridge-denied",
        ),
        messages,
        "task-1",
    )

    executor_agent._invoke_tool.assert_not_called()
    executor_agent._tool_guardrails.before_call.assert_not_called()
    _assert_capability_block(
        messages,
        hook_calls,
        tool_name=external_tool_name,
        call_id="bridge-denied",
    )
