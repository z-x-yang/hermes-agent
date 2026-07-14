import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.review_tools import (
    ReviewCapsuleError,
    ReviewToolError,
    clear_review_context,
    parse_review_capsule,
    register_review_context,
    report_review_findings,
    review_git,
    review_read_file,
    review_search_files,
)


def _capsule(paths=None, *, external_reference_scope="none", mode="uncommitted"):
    target = {"mode": mode, "paths": paths or ["src"]}
    if mode == "base":
        target["base"] = "HEAD"
    elif mode == "commit":
        target["commit"] = "HEAD"
    return {
        "original_ask_or_approved_contract": "Add a safe reviewer.",
        "acceptance_criteria_and_invariants": ["Never edit files."],
        "relevant_repo_rules": ["Use pytest."],
        "review_target": target,
        "verification_evidence": [
            {"command": "pytest -q", "result": "1 passed", "status": "pass"}
        ],
        "known_baseline_failures": [],
        "external_reference_scope": external_reference_scope,
    }


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "reviewer@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Reviewer Fixture"], cwd=root, check=True)
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/a.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)


@pytest.fixture
def review_context(tmp_path):
    _init_repo(tmp_path)
    context = parse_review_capsule(json.dumps(_capsule()), root=tmp_path)
    task_id = "review-task"
    register_review_context(task_id, context)
    try:
        yield task_id, context, tmp_path
    finally:
        clear_review_context(task_id)


def test_parse_review_capsule_binds_trusted_root_and_scope(tmp_path):
    (tmp_path / "src").mkdir()
    context = parse_review_capsule(json.dumps(_capsule()), root=tmp_path)
    assert context.root == tmp_path.resolve()
    assert context.scoped_paths == ("src",)
    assert context.external_reference_scope == "none"
    assert len(context.capsule_digest) == 64


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({}),
        json.dumps(_capsule(paths=["../escape"])),
        json.dumps(_capsule(paths=["/tmp/escape"])),
    ],
)
def test_parse_review_capsule_rejects_malformed_or_escaping_scope(tmp_path, payload):
    with pytest.raises(ReviewCapsuleError):
        parse_review_capsule(payload, root=tmp_path)


@pytest.mark.parametrize("field", ["command", "result"])
def test_parse_review_capsule_rejects_empty_verification_evidence(tmp_path, field):
    capsule = _capsule()
    capsule["verification_evidence"][0][field] = "   "

    with pytest.raises(ReviewCapsuleError, match="command/result"):
        parse_review_capsule(json.dumps(capsule), root=tmp_path)


def test_review_read_file_is_root_scoped_and_bounded(review_context):
    task_id, context, root = review_context
    result = json.loads(review_read_file({"path": "src/a.py"}, task_id=task_id))
    assert result["path"] == "src/a.py"
    assert result["content"] == "1|VALUE = 1"
    assert context.local_paths == {"src/a.py"}

    with pytest.raises(ReviewToolError):
        review_read_file({"path": "../outside"}, task_id=task_id)
    with pytest.raises(ReviewToolError):
        review_read_file({"path": str(root / "src" / "a.py")}, task_id=task_id)


def test_review_read_file_rejects_symlink_escape_before_open(review_context, tmp_path):
    task_id, _context, root = review_context
    outside = tmp_path.parent / f"outside-{os.getpid()}.txt"
    outside.write_text("secret", encoding="utf-8")
    link = root / "src" / "outside-link"
    try:
        link.symlink_to(outside)
        with patch("pathlib.Path.open", wraps=Path.open) as open_mock:
            with pytest.raises(ReviewToolError):
                review_read_file({"path": "src/outside-link"}, task_id=task_id)
            assert open_mock.call_count == 0
    finally:
        outside.unlink(missing_ok=True)


def test_review_search_files_records_only_returned_in_root_paths(review_context):
    task_id, context, _root = review_context
    result = json.loads(
        review_search_files(
            {"pattern": "VALUE", "path": "src", "file_glob": "*.py", "limit": 10},
            task_id=task_id,
        )
    )
    assert result["matches"] == [{"path": "src/a.py", "line": 1, "text": "VALUE = 1"}]
    assert context.local_paths == {"src/a.py"}


def test_review_search_files_excludes_git_metadata_and_oversized_files(review_context):
    task_id, context, root = review_context
    git_secret = root / ".git" / "review-secret.txt"
    git_secret.write_text("HIDDEN_REVIEW_SENTINEL\n", encoding="utf-8")
    oversized = root / "src" / "oversized.txt"
    oversized.write_text(
        "HIDDEN_REVIEW_SENTINEL\n" + ("x" * 2_100_000),
        encoding="utf-8",
    )

    result = json.loads(
        review_search_files(
            {
                "pattern": "HIDDEN_REVIEW_SENTINEL",
                "path": ".",
                "file_glob": "*.txt",
            },
            task_id=task_id,
        )
    )

    assert result["matches"] == []
    assert ".git/review-secret.txt" not in context.local_paths
    assert "src/oversized.txt" not in context.local_paths


@pytest.mark.parametrize(
    "operation,args",
    [
        ("status", {}),
        ("rev_parse", {}),
        ("merge_base", {}),
        ("diff", {}),
        ("show", {}),
        ("log", {"max_count": 5}),
        ("blame", {"path": "src/a.py", "start_line": 1, "end_line": 1}),
        ("grep", {"pattern": "VALUE", "path": "src"}),
        ("ls_files", {"path": "src"}),
    ],
)
def test_review_git_supports_only_structured_read_operations(review_context, operation, args):
    task_id, context, root = review_context
    if operation == "diff":
        (root / "src" / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
    result = json.loads(review_git({"operation": operation, **args}, task_id=task_id))
    assert result["operation"] == operation
    assert result["returncode"] == 0
    assert operation in context.git_operations


def test_review_git_grep_no_match_is_an_empty_success(review_context):
    task_id, context, _root = review_context

    result = json.loads(
        review_git(
            {"operation": "grep", "pattern": "SYMBOL_THAT_DOES_NOT_EXIST", "path": "src"},
            task_id=task_id,
        )
    )

    assert result["operation"] == "grep"
    assert result["returncode"] == 1
    assert result["stdout"] == ""
    assert "grep" in context.git_operations


@pytest.mark.parametrize(
    "args",
    [
        {"operation": "checkout"},
        {"operation": "diff", "path": "../escape"},
        {"operation": "grep", "pattern": "$(touch /tmp/pwned)", "path": "src"},
        {"operation": "ls_files", "path": "src;git status"},
        {"operation": "status", "command": "git status | cat"},
    ],
)
def test_review_git_rejects_mutation_escape_and_shell_input_before_subprocess(review_context, args):
    task_id, _context, _root = review_context
    with patch("tools.review_tools.subprocess.run") as run_mock:
        with pytest.raises(ReviewToolError):
            review_git(args, task_id=task_id)
        assert run_mock.call_count == 0


def _valid_report():
    return {
        "findings": [
            {
                "title": "[P1] Preserve the configured review boundary",
                "body": "The runtime accepts an out-of-scope path and would disclose unrelated source.",
                "priority": 1,
                "confidence": 0.95,
                "code_location": {"path": "src/a.py", "start_line": 1, "end_line": 1},
                "evidence": ["review_git diff", "src/a.py:1"],
            }
        ],
        "overall_correctness": "incorrect",
        "explanation": "The scoped path invariant is violated.",
        "confidence": 0.95,
        "sources_used": {
            "local_paths": ["src/a.py"],
            "git_operations": ["diff"],
            "web_urls": [],
        },
    }


def test_report_review_findings_validates_and_canonicalizes_runtime_sources(review_context):
    task_id, context, root = review_context
    review_read_file({"path": "src/a.py"}, task_id=task_id)
    (root / "src" / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
    review_git({"operation": "diff"}, task_id=task_id)

    result = json.loads(report_review_findings(_valid_report(), task_id=task_id))
    assert result["accepted"] is True
    assert result["report"] == context.report
    assert context.report["overall_correctness"] == "incorrect"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda report: report["findings"][0].update(priority=3),
        lambda report: report["findings"][0].update(confidence=1.1),
        lambda report: report["findings"][0]["code_location"].update(path="../outside.py"),
        lambda report: report["findings"][0]["code_location"].update(start_line=20, end_line=10),
        lambda report: report["findings"][0]["code_location"].update(start_line=1, end_line=30),
        lambda report: report["findings"][0]["code_location"].update(start_line=2, end_line=2),
        lambda report: report["sources_used"]["local_paths"].append("src/invented.py"),
        lambda report: report["sources_used"]["git_operations"].append("show"),
    ],
)
def test_report_review_findings_rejects_invalid_or_invented_evidence(review_context, mutator):
    task_id, _context, root = review_context
    review_read_file({"path": "src/a.py"}, task_id=task_id)
    (root / "src" / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
    review_git({"operation": "diff"}, task_id=task_id)
    report = _valid_report()
    mutator(report)
    with pytest.raises(ReviewToolError):
        report_review_findings(report, task_id=task_id)


@pytest.mark.parametrize(
    "findings,overall",
    [
        (_valid_report()["findings"], "correct"),
        ([], "incorrect"),
    ],
)
def test_report_review_findings_rejects_contradictory_overall_verdict(
    review_context, findings, overall
):
    task_id, _context, _root = review_context
    review_read_file({"path": "src/a.py"}, task_id=task_id)
    report = _valid_report()
    report["findings"] = findings
    report["overall_correctness"] = overall
    report["sources_used"]["git_operations"] = []

    with pytest.raises(ReviewToolError, match="overall_correctness"):
        report_review_findings(report, task_id=task_id)


def test_report_review_findings_rejects_uninspected_target(review_context):
    task_id, _context, _root = review_context
    report = {
        "findings": [],
        "overall_correctness": "unverified",
        "explanation": "No target evidence was inspected.",
        "confidence": 0.1,
        "sources_used": {"local_paths": [], "git_operations": [], "web_urls": []},
    }
    with pytest.raises(ReviewToolError, match="inspect"):
        report_review_findings(report, task_id=task_id)


def test_report_review_findings_rejects_status_only_target_check(review_context):
    task_id, _context, _root = review_context
    review_git({"operation": "status"}, task_id=task_id)
    report = {
        "findings": [],
        "overall_correctness": "unverified",
        "explanation": "Only target status was inspected.",
        "confidence": 0.1,
        "sources_used": {
            "local_paths": [],
            "git_operations": ["status"],
            "web_urls": [],
        },
    }
    with pytest.raises(ReviewToolError, match="inspect"):
        report_review_findings(report, task_id=task_id)


def test_report_review_findings_accepts_valid_no_findings(review_context):
    task_id, context, _root = review_context
    review_read_file({"path": "src/a.py"}, task_id=task_id)
    report = {
        "findings": [],
        "overall_correctness": "correct",
        "explanation": "No evidence-backed blockers were found.",
        "confidence": 0.8,
        "sources_used": {
            "local_paths": ["src/a.py"],
            "git_operations": [],
            "web_urls": [],
        },
    }
    result = json.loads(report_review_findings(report, task_id=task_id))
    assert result["accepted"] is True
    assert context.report["findings"] == []

    with pytest.raises(ReviewToolError):
        report_review_findings(report, task_id=task_id)
