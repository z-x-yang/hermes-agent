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
from tools.registry import registry
from tools.subagent_profiles import get_subagent_profile
from tools.tool_effects import build_authority_snapshot


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": name, "parameters": {}},
    }


def _set_authority(agent, names) -> set[str]:
    # Core registrations are side effects of importing model_tools in the real
    # agent-init path; mirror that ordering before resolving test identities.
    import model_tools  # noqa: F401

    resolved = {
        identity
        for name in names
        if isinstance((identity := registry.resolved_policy_identity(name)), str)
    }
    agent._parent_tool_authority_snapshot = build_authority_snapshot(
        resolved, registry_generation=registry._generation
    )
    return resolved


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
    parent.valid_tool_names = {
        "read_file",
        "write_file",
        "terminal",
        "mcp_apple_mail_send_email",
    }
    _set_authority(parent, parent.valid_tool_names)
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
    _set_authority(child, child.valid_tool_names)
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
    # Governed children may refresh, but refresh is filtered by their original
    # exact authority instead of being disabled wholesale.
    assert agent._skip_mcp_refresh is False


def test_direct_hallucinated_tool_name_is_execution_blocked():
    agent = SimpleNamespace(
        _subagent_tool_policy=ToolNamePolicy(
            allowed_names=frozenset({"read_file"})
        )
    )

    message = tool_policy_block_message(agent, "write_file")

    assert "blocked by subagent capability policy" in message


def test_general_purpose_name_policy_does_not_remove_parent_authorized_actions():
    profile = get_subagent_profile("general-purpose")
    policy = ToolNamePolicy(allowed_names=profile.allowed_tool_names)
    agent = SimpleNamespace(_subagent_tool_policy=policy)

    for tool_name in (
        "write_file",
        "patch",
        "terminal",
        "process",
        "mcp_notion_ai_notion_ai_ask",
    ):
        assert tool_policy_block_message(agent, tool_name) is None


def test_profiled_child_filters_visible_and_runtime_tool_names():
    parent = _parent_agent()
    child = _child_agent()

    with patch("run_agent.AIAgent", return_value=child):
        built = _build_child_agent(
            task_index=0,
            description="Inspect the repository",
            prompt="",
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
    assert child._subagent_tool_policy.allowed_names == frozenset({"read_file"})
    assert child._skip_mcp_refresh is False


@pytest.mark.parametrize("profile_name", [None, "general-purpose"])
def test_initial_child_intersects_exact_current_parent_tool_names(profile_name):
    parent = _parent_agent()
    parent.enabled_toolsets = ["file"]
    parent.valid_tool_names = {"read_file"}
    _set_authority(parent, parent.valid_tool_names)
    child = _child_agent()
    profile = get_subagent_profile(profile_name) if profile_name else None

    with patch("run_agent.AIAgent", return_value=child):
        built = _build_child_agent(
            task_index=0,
            description="Stay within the current parent ceiling",
            prompt="",
            toolsets=None,
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            profile=profile,
        )

    assert built.valid_tool_names == {"read_file"}
    assert [tool["function"]["name"] for tool in built.tools] == ["read_file"]
    assert built._subagent_tool_policy.allowed_names == frozenset({"read_file"})


@pytest.mark.parametrize("profile_name", [None, "general-purpose"])
def test_initial_child_keeps_exact_empty_parent_tool_names_fail_closed(profile_name):
    parent = _parent_agent()
    parent.enabled_toolsets = ["file"]
    parent.valid_tool_names = set()
    _set_authority(parent, parent.valid_tool_names)
    child = _child_agent()
    profile = get_subagent_profile(profile_name) if profile_name else None

    with patch("run_agent.AIAgent", return_value=child):
        built = _build_child_agent(
            task_index=0,
            description="Have no tools",
            prompt="",
            toolsets=None,
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            profile=profile,
        )

    assert built.valid_tool_names == set()
    assert built.tools == []
    assert built._subagent_tool_policy.allowed_names == frozenset()


def test_general_purpose_gets_delegate_task_only_when_runtime_gates_allow():
    parent = _parent_agent()
    parent.enabled_toolsets = ["file"]
    parent.valid_tool_names = {"read_file", "delegate_task"}
    child = _child_agent()
    child.valid_tool_names |= {"delegate_task", "delegate_continue"}
    child.tools.extend([_tool("delegate_task"), _tool("delegate_continue")])

    with (
        patch("run_agent.AIAgent", return_value=child),
        patch("tools.delegate_tool._get_orchestrator_enabled", return_value=True),
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=2),
    ):
        _set_authority(parent, parent.valid_tool_names)
        _set_authority(child, child.valid_tool_names)
        built = _build_child_agent(
            task_index=0,
            description="Orchestrate one worker layer",
            prompt="",
            toolsets=None,
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,

            profile=get_subagent_profile("general-purpose"),
        )

    assert "_delegate_role" not in built.__dict__
    assert built.valid_tool_names == {"read_file", "delegate_task"}
    assert built._subagent_tool_policy.allowed_names == frozenset(
        {"read_file", "delegate_task"}
    )
    assert "delegate_continue" not in built.valid_tool_names


@pytest.mark.parametrize("bounded_gate", ["kill_switch", "depth"])
def test_general_purpose_delegation_requires_live_runtime_gates(
    bounded_gate,
):
    parent = _parent_agent()
    parent.enabled_toolsets = ["file"]
    parent.valid_tool_names = {"read_file", "delegate_task"}
    _set_authority(parent, parent.valid_tool_names)
    child = _child_agent()
    child.valid_tool_names |= {"delegate_task", "delegate_continue"}
    child.tools.extend([_tool("delegate_task"), _tool("delegate_continue")])
    _set_authority(child, child.valid_tool_names)
    orchestrator_enabled = bounded_gate != "kill_switch"
    max_spawn_depth = 1 if bounded_gate == "depth" else 2

    with (
        patch("run_agent.AIAgent", return_value=child),
        patch(
            "tools.delegate_tool._get_orchestrator_enabled",
            return_value=orchestrator_enabled,
        ),
        patch(
            "tools.delegate_tool._get_max_spawn_depth",
            return_value=max_spawn_depth,
        ),
    ):
        built = _build_child_agent(
            task_index=0,
            description="Bounded orchestrator",
            prompt="",
            toolsets=None,
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,

            profile=get_subagent_profile("general-purpose"),
        )

    assert "_delegate_role" not in built.__dict__
    assert built.valid_tool_names == {"read_file"}
    assert built._subagent_tool_policy.allowed_names == frozenset({"read_file"})


def test_initial_child_can_skip_parent_lifecycle_registration():
    parent = _parent_agent()
    child = _child_agent()

    with patch("run_agent.AIAgent", return_value=child):
        built = _build_child_agent(
            task_index=0,
            description="Async-owned child",
            prompt="",
            toolsets=None,
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            register_with_parent=False,
        )

    assert built is child
    assert child not in parent._active_children


def test_omitted_profile_uses_general_purpose_policy_not_fourth_legacy_policy():
    parent = _parent_agent()
    parent.valid_tool_names = {
        "read_file",
        "write_file",
        "terminal",
        "mcp_apple_mail_send_email",
    }
    _set_authority(parent, parent.valid_tool_names)
    child = _child_agent()

    with patch("run_agent.AIAgent", return_value=child):
        _build_child_agent(
            task_index=0,
            description="Legacy delegation",
            prompt="",
            toolsets=None,
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            profile=None,
        )

    assert child._subagent_tool_policy.profile_name == "general-purpose"
    assert child._subagent_profile.name == "general-purpose"
    assert child.valid_tool_names == {
        "read_file",
        "write_file",
        "terminal",
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


@pytest.mark.parametrize(
    "executor",
    [execute_tool_calls_sequential, execute_tool_calls_concurrent],
)
def test_readonly_executor_dispatches_authorized_read_once_without_middleware(
    executor_agent, monkeypatch, executor
):
    from tools.registry import registry
    from tools.tool_effects import (
        ResultRetention,
        ToolEffect,
        build_authority_snapshot,
        builtin_policy_descriptor,
    )

    name = "task6_executor_read"
    schema = _tool(name)["function"]
    backend_calls = []

    def handler(args, **kwargs):
        backend_calls.append(dict(args))
        return json.dumps({"ok": True, "value": args.get("value")})

    registry.register(
        name=name,
        toolset="task6-test",
        schema=schema,
        handler=handler,
        descriptor=builtin_policy_descriptor(
            name=name,
            schema=schema,
            handler=handler,
            effects={ToolEffect.READ_LOCAL},
            retention=ResultRetention.NO_SPILL,
        ),
        override=True,
    )
    try:
        identity = registry.resolved_policy_identity(name)
        assert isinstance(identity, str)
        executor_agent.tools.append(_tool(name))
        executor_agent.valid_tool_names.add(name)
        apply_tool_policy_to_agent(
            executor_agent,
            ToolNamePolicy(
                allowed_names=frozenset({name}),
                allowed_effects=frozenset(
                    {ToolEffect.READ_LOCAL, ToolEffect.READ_REMOTE}
                ),
                authority_snapshot=build_authority_snapshot(
                    {identity}, registry_generation=registry._generation
                ),
                profile_name="Explore",
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.middleware.apply_tool_request_middleware",
            MagicMock(side_effect=AssertionError("request middleware ran")),
        )
        monkeypatch.setattr(
            "hermes_cli.middleware.run_tool_execution_middleware",
            MagicMock(side_effect=AssertionError("execution middleware ran")),
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_tool_call_block_message",
            lambda *args, **kwargs: None,
        )
        messages = []

        executor(
            executor_agent,
            _assistant_tool_call(name, {"value": "safe"}, "read-call"),
            messages,
            "task-1",
        )

        assert backend_calls == [{"value": "safe"}]
        assert (
            executor_agent._subagent_tool_result_retention_by_call_id[
                "read-call"
            ]
            is ResultRetention.NO_SPILL
        )
        assert len(messages) == 1
        assert json.loads(messages[0]["content"])["ok"] is True
    finally:
        registry.deregister(name)


@pytest.mark.parametrize(
    "executor",
    [execute_tool_calls_sequential, execute_tool_calls_concurrent],
)
def test_effect_denial_stops_before_backend_hooks_checkpoint_and_persistence(
    executor_agent, monkeypatch, executor
):
    from tools.registry import registry
    from tools.tool_effects import (
        ResultRetention,
        ToolEffect,
        build_authority_snapshot,
        builtin_policy_descriptor,
    )

    name = "task6_executor_unknown"
    schema = _tool(name)["function"]
    backend_calls = []

    def handler(args, **kwargs):
        backend_calls.append(dict(args))
        return "must not run"

    registry.register(
        name=name,
        toolset="task6-test",
        schema=schema,
        handler=handler,
        descriptor=builtin_policy_descriptor(
            name=name,
            schema=schema,
            handler=handler,
            effects={ToolEffect.UNKNOWN},
            retention=ResultRetention.NO_SPILL,
        ),
        override=True,
    )
    try:
        identity = registry.resolved_policy_identity(name)
        assert isinstance(identity, str)
        executor_agent.tools.append(_tool(name))
        executor_agent.valid_tool_names.add(name)
        apply_tool_policy_to_agent(
            executor_agent,
            ToolNamePolicy(
                allowed_names=frozenset({name}),
                allowed_effects=frozenset(
                    {ToolEffect.READ_LOCAL, ToolEffect.READ_REMOTE}
                ),
                authority_snapshot=build_authority_snapshot(
                    {identity}, registry_generation=registry._generation
                ),
                profile_name="Explore",
            ),
        )
        pre_hook = MagicMock(
            side_effect=AssertionError("capability denial reached pre hook")
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_tool_call_block_message", pre_hook
        )
        persist_result = MagicMock(
            side_effect=lambda **kwargs: kwargs["content"]
        )
        monkeypatch.setattr(
            "agent.tool_executor.maybe_persist_tool_result", persist_result
        )
        executor_agent._flush_messages_to_session_db = MagicMock()
        executor_agent._checkpoint_mgr.ensure_checkpoint = MagicMock(
            side_effect=AssertionError("capability denial reached checkpoint")
        )
        messages = []

        executor(
            executor_agent,
            _assistant_tool_call(name, {"value": "blocked"}, "denied-call"),
            messages,
            "task-1",
        )

        assert backend_calls == []
        pre_hook.assert_not_called()
        executor_agent._checkpoint_mgr.ensure_checkpoint.assert_not_called()
        persist_result.assert_not_called()
        executor_agent._flush_messages_to_session_db.assert_not_called()
        assert len(messages) == 1
        assert "effect" in messages[0]["content"]
    finally:
        registry.deregister(name)


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
    # Capability denial is fail-closed before every third-party hook,
    # including observational post hooks.
    assert hook_calls == []


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
