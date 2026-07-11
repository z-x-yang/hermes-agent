"""Tests for the subagent_stop hook event.

Covers wire-up from tools.delegate_tool.delegate_task:
  * fires once per child in both single-task and batch modes
  * runs on the invoking thread for nested foreground delegation
  * carries parent/child session and completion metadata
  * does not leak internal child-cost accounting fields
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import plugins
from tools.delegate_tool import delegate_task


def _make_parent(depth: int = 1, session_id: str = "parent-1"):
    """Use nested depth so run_in_background=False executes synchronously."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._memory_manager = None
    parent.session_id = session_id
    return parent


@pytest.fixture(autouse=True)
def _fresh_plugin_manager():
    """Each test gets a fresh PluginManager so callbacks do not leak."""
    original = plugins._plugin_manager
    plugins._plugin_manager = plugins.PluginManager()
    yield
    plugins._plugin_manager = original


@pytest.fixture(autouse=True)
def _stub_child_builder(monkeypatch):
    """Avoid importing heavyweight runtime dependencies."""

    def _fake_build_child(task_index, **kwargs):
        child = MagicMock()
        child.session_id = f"child-{task_index}"
        child._delegate_saved_tool_names = []
        child._credential_pool = None
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", _fake_build_child)
    monkeypatch.setattr("tools.delegate_tool._get_max_spawn_depth", lambda: 2)


def _register_capturing_hook():
    captured = []

    def _cb(**kwargs):
        kwargs["_thread"] = threading.current_thread()
        captured.append(kwargs)

    mgr = plugins.get_plugin_manager()
    mgr._hooks.setdefault("subagent_stop", []).append(_cb)
    return captured


def _completed(index: int, summary: str, duration: float = 0.1):
    return {
        "task_index": index,
        "status": "completed",
        "summary": summary,
        "api_calls": 1,
        "duration_seconds": duration,
    }


class TestSingleTask:
    def test_fires_once(self):
        captured = _register_capturing_hook()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = _completed(0, "Done!", 5.0)
            delegate_task(
                description="do X",
                prompt="do X",
                run_in_background=False,
                parent_agent=_make_parent(),
            )

        assert len(captured) == 1
        payload = captured[0]
        assert payload["child_status"] == "completed"
        assert payload["child_summary"] == "Done!"
        assert payload["duration_ms"] == 5000
        assert payload["child_session_id"] == "child-0"
        assert "child_role" not in payload

    def test_fires_on_invoking_thread(self):
        captured = _register_capturing_hook()
        main_thread = threading.current_thread()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = _completed(0, "x")
            delegate_task(
                description="go",
                prompt="go",
                run_in_background=False,
                parent_agent=_make_parent(),
            )

        assert captured[0]["_thread"] is main_thread

    def test_payload_includes_parent_session_id(self):
        captured = _register_capturing_hook()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.return_value = _completed(0, "x")
            delegate_task(
                description="go",
                prompt="go",
                run_in_background=False,
                parent_agent=_make_parent(session_id="sess-xyz"),
            )

        assert captured[0]["parent_session_id"] == "sess-xyz"


class TestBatchMode:
    def test_fires_per_child(self):
        captured = _register_capturing_hook()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [
                _completed(0, "A", 1.0),
                _completed(1, "B", 2.0),
                _completed(2, "C", 3.0),
            ]
            delegate_task(
                tasks=[
                    {"description": "A", "prompt": "A"},
                    {"description": "B", "prompt": "B"},
                    {"description": "C", "prompt": "C"},
                ],
                run_in_background=False,
                parent_agent=_make_parent(),
            )

        assert len(captured) == 3
        assert sorted(c["child_summary"] for c in captured) == ["A", "B", "C"]
        assert all("child_role" not in c for c in captured)

    def test_all_fire_on_invoking_thread(self):
        captured = _register_capturing_hook()
        main_thread = threading.current_thread()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            mock_run.side_effect = [_completed(0, "A"), _completed(1, "B")]
            delegate_task(
                tasks=[
                    {"description": "A", "prompt": "A"},
                    {"description": "B", "prompt": "B"},
                ],
                run_in_background=False,
                parent_agent=_make_parent(),
            )

        assert all(payload["_thread"] is main_thread for payload in captured)


class TestPayloadShape:
    def test_internal_cost_field_does_not_leak(self):
        _register_capturing_hook()

        with patch("tools.delegate_tool._run_single_child") as mock_run:
            result = _completed(0, "x")
            result["_child_cost_usd"] = 1.25
            mock_run.return_value = result
            raw = delegate_task(
                description="do X",
                prompt="do X",
                run_in_background=False,
                parent_agent=_make_parent(),
            )

        parsed = json.loads(raw)
        assert "results" in parsed
        assert "_child_cost_usd" not in parsed["results"][0]
