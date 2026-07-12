"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _build_child_agent,
    _emit_parent_console,
    _strip_blocked_tools,
)
from tools.registry import registry
from tools.subagent_profiles import get_subagent_profile
from tools.tool_effects import build_authority_snapshot


def _set_authority(agent, names) -> None:
    import model_tools  # noqa: F401

    identities = {
        identity
        for name in names
        if isinstance((identity := registry.resolved_policy_identity(name)), str)
    }
    agent._parent_tool_authority_snapshot = build_authority_snapshot(
        identities, registry_generation=registry._generation
    )


class TestToolsetIntersection:
    """Subagent toolsets must be a subset of parent's enabled_toolsets."""

    def test_requested_toolsets_intersected_with_parent(self):
        """LLM requests toolsets parent doesn't have — extras are dropped."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file"])

        # Simulate the intersection logic from _build_child_agent
        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "file", "web", "browser", "rl"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert sorted(scoped) == ["file", "terminal"]
        assert "web" not in scoped
        assert "browser" not in scoped
        assert "rl" not in scoped

    def test_all_requested_toolsets_available_on_parent(self):
        """LLM requests subset of parent tools — all pass through."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file", "web", "browser"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "web"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert sorted(scoped) == ["terminal", "web"]

    def test_no_toolsets_requested_inherits_parent(self):
        """When toolsets is None/empty, child inherits parent's set."""
        parent_toolsets = ["terminal", "file", "web"]
        child = _strip_blocked_tools(parent_toolsets)
        assert "terminal" in child
        assert "file" in child
        assert "web" in child

    def test_strip_blocked_removes_control_plane_only(self):
        """Leaf control-plane toolsets are removed without shrinking GP actions."""
        child = _strip_blocked_tools(
            ["terminal", "delegation", "clarify", "memory", "code_execution"]
        )
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" in child
        assert "code_execution" in child
        assert "terminal" in child

    def test_empty_intersection_yields_empty_toolsets(self):
        """If parent has no overlap with requested, child gets nothing extra."""
        parent = SimpleNamespace(enabled_toolsets=["terminal"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["web", "browser"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert scoped == []

    def test_builder_normalizes_omitted_profile_to_general_purpose(
        self, monkeypatch
    ):
        parent = MagicMock()
        parent._delegate_depth = 0
        parent.valid_tool_names = {"read_file"}
        parent.enabled_toolsets = {"file"}
        parent._active_children = []
        parent._active_children_lock = None
        child = MagicMock()
        child.valid_tool_names = {"read_file", "write_file"}
        child.tools = [
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in sorted(child.valid_tool_names)
        ]
        _set_authority(parent, parent.valid_tool_names)
        _set_authority(child, child.valid_tool_names)

        with patch("run_agent.AIAgent", return_value=child):
            built = _build_child_agent(
                task_index=0,
                description="inspect",
                prompt=None,
                toolsets=None,
                model=None,
                max_iterations=5,
                task_count=1,
                parent_agent=parent,
                profile=None,
            )

        assert built._subagent_profile.name == "general-purpose"
        assert built.valid_tool_names == {"read_file"}

    def test_general_purpose_automatically_gets_delegate_task_only_when_all_gates_allow(
        self, monkeypatch
    ):
        import tools.delegate_tool as dt

        parent = MagicMock()
        parent._delegate_depth = 0
        parent.valid_tool_names = {"read_file", "delegate_task"}
        parent.enabled_toolsets = {"file", "delegation"}
        parent._active_children = []
        parent._active_children_lock = None
        child = MagicMock()
        child.valid_tool_names = {"read_file", "delegate_task"}
        child.tools = [
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in sorted(child.valid_tool_names)
        ]
        _set_authority(parent, parent.valid_tool_names)
        _set_authority(child, child.valid_tool_names)
        monkeypatch.setattr(dt, "_get_orchestrator_enabled", lambda: True)
        monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 2)

        with patch("run_agent.AIAgent", return_value=child):
            built = _build_child_agent(
                task_index=0,
                description="decompose work",
                prompt="split this task",
                toolsets=None,
                model=None,
                max_iterations=5,
                task_count=1,
                parent_agent=parent,
                profile=get_subagent_profile("general-purpose"),
            )

        assert "delegate_task" in built.valid_tool_names
        assert "delegate_continue" not in built.valid_tool_names
        assert "clarify" not in built.valid_tool_names
        assert "_delegate_role" not in built.__dict__

    def test_automatic_delegation_fails_closed_when_any_gate_is_missing(
        self, monkeypatch
    ):
        import tools.delegate_tool as dt

        cases = [
            ("Explore", True, 2, {"read_file", "delegate_task"}),
            ("Plan", True, 2, {"read_file", "delegate_task"}),
            ("general-purpose", False, 2, {"read_file", "delegate_task"}),
            ("general-purpose", True, 1, {"read_file", "delegate_task"}),
            ("general-purpose", True, 2, {"read_file"}),
        ]
        for profile_name, enabled, max_depth, parent_names in cases:
            parent = MagicMock()
            parent._delegate_depth = 0
            parent.valid_tool_names = set(parent_names)
            parent.enabled_toolsets = {"file", "delegation"}
            parent._active_children = []
            parent._active_children_lock = None
            child = MagicMock()
            child.valid_tool_names = {
                "read_file",
                "delegate_task",
                "delegate_continue",
                "clarify",
            }
            child.tools = [
                {"type": "function", "function": {"name": name, "parameters": {}}}
                for name in sorted(child.valid_tool_names)
            ]
            _set_authority(parent, parent.valid_tool_names)
            _set_authority(child, child.valid_tool_names)
            monkeypatch.setattr(
                dt, "_get_orchestrator_enabled", lambda value=enabled: value
            )
            monkeypatch.setattr(
                dt, "_get_max_spawn_depth", lambda value=max_depth: value
            )
            with patch("run_agent.AIAgent", return_value=child):
                built = _build_child_agent(
                    task_index=0,
                    description="inspect work",
                    prompt="inspect this task",
                    toolsets=None,
                    model=None,
                    max_iterations=5,
                    task_count=1,
                    parent_agent=parent,
                    profile=get_subagent_profile(profile_name),
                )
            assert {
                "delegate_task",
                "delegate_continue",
                "clarify",
            }.isdisjoint(built.valid_tool_names)
    def test_general_purpose_without_parent_exact_names_fails_closed(
        self, monkeypatch
    ):
        parent = MagicMock()
        parent._delegate_depth = 0
        parent.enabled_toolsets = {"terminal"}
        parent._active_children = []
        parent._active_children_lock = None
        child = MagicMock()
        child.valid_tool_names = {"terminal"}

        with patch("run_agent.AIAgent", return_value=child):
            built = _build_child_agent(
                task_index=0,
                description="inspect",
                prompt=None,
                toolsets=None,
                model=None,
                max_iterations=5,
                task_count=1,
                parent_agent=parent,
                profile=get_subagent_profile("general-purpose"),
            )

        assert built.valid_tool_names == set()


class TestEmitParentConsole:
    """Progress lines (e.g. ``✓ [N/M] …``) must route through the parent's
    configured ``_safe_print`` in headless stdio hosts (ACP, gateway) so
    they don't land on stdout and corrupt JSON-RPC frames. Regression for a
    bug where delegate_task completion lines pushed to stdout caused
    ``Failed to parse JSON message: ✓ [3/3] …`` errors in the ACP adapter."""

    def test_routes_through_parent_safe_print_when_available(self, capsys):
        captured_lines = []
        parent = SimpleNamespace(_safe_print=lambda line: captured_lines.append(line))

        _emit_parent_console(parent, "  ✓ [1/3] Research done  (11.55s)")

        assert captured_lines == ["  ✓ [1/3] Research done  (11.55s)"]
        stdout_stderr = capsys.readouterr()
        assert stdout_stderr.out == ""
        assert stdout_stderr.err == ""

    def test_falls_back_to_stdout_when_no_safe_print(self, capsys):
        parent = SimpleNamespace()
        _emit_parent_console(parent, "  ✓ [1/3] fallback path")
        captured = capsys.readouterr()
        assert "fallback path" in captured.out

    def test_falls_back_to_stdout_when_safe_print_raises(self, capsys):
        def raiser(_line):
            raise RuntimeError("boom")

        parent = SimpleNamespace(_safe_print=raiser)
        _emit_parent_console(parent, "  ✓ [2/3] fallback on exception")
        captured = capsys.readouterr()
        assert "fallback on exception" in captured.out

    def test_non_callable_safe_print_is_ignored(self, capsys):
        """Defensive: if _safe_print is set but not callable, fall back."""
        parent = SimpleNamespace(_safe_print="not-a-function")
        _emit_parent_console(parent, "  ✓ [3/3] non-callable guard")
        captured = capsys.readouterr()
        assert "non-callable guard" in captured.out
