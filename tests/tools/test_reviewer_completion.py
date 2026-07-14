import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from tools.delegate_tool import _run_single_child


def _parent():
    return SimpleNamespace(
        _current_task_id=None,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        session_id="parent-session",
    )


def _child(final_response: str):
    child = MagicMock()
    child._subagent_id = "sa-review-test"
    child._delegate_saved_tool_names = []
    child._credential_pool = None
    child.tool_progress_callback = None
    child.model = "review-model"
    child.max_iterations = 150
    child.session_prompt_tokens = 10
    child.session_completion_tokens = 5
    child.session_reasoning_tokens = 0
    child.session_estimated_cost_usd = 0.0
    child.run_conversation.return_value = {
        "final_response": final_response,
        "completed": True,
        "api_calls": 1,
        "messages": [],
    }
    return child


def test_reviewer_completion_returns_normal_final_response(tmp_path):
    final_response = (
        "### [Important] `src/a.py:7` — reachable incorrect branch\n"
        "Concrete failure scenario and evidence."
    )
    child = _child(final_response)
    parent = _parent()
    parent._active_children.append(child)

    result = _run_single_child(
        task_index=0,
        description="review code",
        child=child,
        parent_agent=parent,
        prompt="Review src/a.py and report high-signal blockers.",
        subagent_type="Reviewer",
        workspace_path=str(tmp_path),
    )

    assert result["status"] == "completed"
    assert result["exit_reason"] == "completed"
    assert result["summary"] == final_response
    assert "review_report" not in result


def test_reviewer_no_finding_final_response_is_valid_completion(tmp_path):
    final_response = "No high-signal candidate blockers found in the scoped change."
    child = _child(final_response)
    parent = _parent()
    parent._active_children.append(child)

    result = _run_single_child(
        task_index=0,
        description="review code",
        child=child,
        parent_agent=parent,
        prompt="Review the scoped change.",
        subagent_type="Reviewer",
        workspace_path=str(tmp_path),
    )

    assert result["status"] == "completed"
    assert result["summary"] == final_response
