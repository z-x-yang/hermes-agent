import json
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools.delegate_tool import _run_single_child
from tools.review_tools import (
    ReviewToolError,
    get_review_context,
    parse_review_capsule,
    report_review_findings,
    review_git,
)


def _capsule():
    return {
        "original_ask_or_approved_contract": "Review the implementation.",
        "acceptance_criteria_and_invariants": ["Return structured findings."],
        "relevant_repo_rules": [],
        "review_target": {"mode": "uncommitted", "paths": ["src"]},
        "verification_evidence": [{"command": "pytest -q", "result": "1 passed", "status": "pass"}],
        "known_baseline_failures": [],
        "external_reference_scope": "none",
    }


def _report():
    return {
        "findings": [],
        "overall_correctness": "correct",
        "explanation": "No evidence-backed blockers were found.",
        "confidence": 0.8,
        "sources_used": {"local_paths": [], "git_operations": ["diff"], "web_urls": []},
    }


def _repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "reviewer@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Reviewer Fixture"], cwd=tmp_path, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/a.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)


def _parent():
    return SimpleNamespace(
        _current_task_id=None,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        session_id="parent-session",
    )


def _child(context, run_conversation):
    child = MagicMock()
    child._subagent_id = "sa-review-test"
    child._review_context = context
    child._delegate_saved_tool_names = []
    child._credential_pool = None
    child.tool_progress_callback = None
    child.model = "review-model"
    child.max_iterations = 150
    child.session_prompt_tokens = 10
    child.session_completion_tokens = 5
    child.session_reasoning_tokens = 0
    child.session_estimated_cost_usd = 0.0
    child.run_conversation.side_effect = run_conversation
    return child


def test_reviewer_completion_requires_and_returns_validated_report(tmp_path):
    _repo(tmp_path)
    context = parse_review_capsule(json.dumps(_capsule()), root=tmp_path)

    def run_conversation(**kwargs):
        review_git({"operation": "diff"}, task_id=kwargs["task_id"])
        report_review_findings(_report(), task_id=kwargs["task_id"])
        return {
            "final_response": "free text must not become the review result",
            "completed": True,
            "api_calls": 1,
            "messages": [],
        }

    child = _child(context, run_conversation)
    parent = _parent()
    parent._active_children.append(child)
    result = _run_single_child(
        task_index=0,
        description="sealed review",
        child=child,
        parent_agent=parent,
        prompt=json.dumps(_capsule()),
        subagent_type="Reviewer",
        workspace_path=str(tmp_path),
    )
    assert result["status"] == "completed"
    assert result["exit_reason"] == "completed"
    assert result["review_report"] == context.report
    assert json.loads(result["summary"]) == context.report
    assert "free text" not in result["summary"]
    with pytest.raises(ReviewToolError):
        get_review_context("sa-review-test")


def test_reviewer_free_text_without_completion_tool_fails(tmp_path):
    _repo(tmp_path)
    context = parse_review_capsule(json.dumps(_capsule()), root=tmp_path)

    def run_conversation(**_kwargs):
        return {
            "final_response": "looks good",
            "completed": True,
            "api_calls": 1,
            "messages": [],
        }

    child = _child(context, run_conversation)
    parent = _parent()
    parent._active_children.append(child)
    result = _run_single_child(
        task_index=0,
        description="sealed review",
        child=child,
        parent_agent=parent,
        prompt=json.dumps(_capsule()),
        subagent_type="Reviewer",
        workspace_path=str(tmp_path),
    )
    assert result["status"] == "failed"
    assert result["exit_reason"] == "error"
    assert result["error"] == "review_report_missing_or_invalid"
    assert "review_report" not in result


def test_reviewer_invalidates_report_when_scoped_target_drifts(tmp_path):
    _repo(tmp_path)
    context = parse_review_capsule(json.dumps(_capsule()), root=tmp_path)

    def run_conversation(**kwargs):
        review_git({"operation": "diff"}, task_id=kwargs["task_id"])
        report_review_findings(_report(), task_id=kwargs["task_id"])
        (tmp_path / "src" / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
        return {
            "final_response": "done",
            "completed": True,
            "api_calls": 1,
            "messages": [],
        }

    child = _child(context, run_conversation)
    parent = _parent()
    parent._active_children.append(child)
    result = _run_single_child(
        task_index=0,
        description="sealed review",
        child=child,
        parent_agent=parent,
        prompt=json.dumps(_capsule()),
        subagent_type="Reviewer",
        workspace_path=str(tmp_path),
    )
    assert result["status"] == "failed"
    assert result["error"] == "review_target_drifted"
    assert "review_report" not in result
