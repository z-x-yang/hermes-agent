import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from tools.review_tools import (
    ReviewToolError,
    clear_review_context,
    parse_review_capsule,
    register_review_context,
)
from tools.web_tools import _web_extract_readonly_handler, _web_search_readonly_handler


def _capsule(scope):
    return {
        "original_ask_or_approved_contract": "Review the change.",
        "acceptance_criteria_and_invariants": ["Use only approved evidence."],
        "relevant_repo_rules": [],
        "review_target": {"mode": "uncommitted", "paths": ["src"]},
        "verification_evidence": [{"command": "pytest -q", "result": "1 passed", "status": "pass"}],
        "known_baseline_failures": [],
        "external_reference_scope": scope,
    }


def _context(tmp_path, scope):
    (tmp_path / "src").mkdir()
    context = parse_review_capsule(json.dumps(_capsule(scope)), root=tmp_path)
    register_review_context("review-web", context)
    return context


def test_reviewer_web_scope_none_rejects_search_before_backend(tmp_path):
    _context(tmp_path, "none")
    try:
        with patch("tools.web_tools.web_search_readonly") as backend:
            with pytest.raises(ReviewToolError):
                _web_search_readonly_handler(
                    {"query": "official Python docs", "limit": 5},
                    task_id="review-web",
                )
            assert backend.call_count == 0
    finally:
        clear_review_context("review-web")


def test_reviewer_web_scope_none_rejects_extract_before_backend(tmp_path):
    _context(tmp_path, "none")
    try:
        with patch("tools.web_tools.web_extract_readonly", new_callable=AsyncMock) as backend:
            with pytest.raises(ReviewToolError):
                asyncio.run(
                    _web_extract_readonly_handler(
                        {"urls": ["https://docs.python.org/3/"]},
                        task_id="review-web",
                    )
                )
            assert backend.await_count == 0
    finally:
        clear_review_context("review-web")


def test_reviewer_authoritative_docs_scope_records_actual_search_urls(tmp_path):
    context = _context(tmp_path, "authoritative_docs_only")
    try:
        payload = json.dumps(
            {"data": {"web": [{"url": "https://docs.python.org/3/library/subprocess.html"}]}}
        )
        with patch("tools.web_tools.web_search_readonly", return_value=payload) as backend:
            result = _web_search_readonly_handler(
                {"query": "official Python subprocess docs", "limit": 5},
                task_id="review-web",
            )
        assert result == payload
        assert backend.call_count == 1
        assert context.web_urls == {"https://docs.python.org/3/library/subprocess.html"}
    finally:
        clear_review_context("review-web")


def test_reviewer_authoritative_docs_scope_records_extract_urls(tmp_path):
    context = _context(tmp_path, "authoritative_docs_only")
    url = "https://docs.python.org/3/library/pathlib.html"
    try:
        with patch(
            "tools.web_tools.web_extract_readonly",
            new=AsyncMock(return_value=json.dumps({"results": [{"url": url, "content": "docs"}]})),
        ) as backend:
            asyncio.run(
                _web_extract_readonly_handler(
                    {"urls": [url], "char_limit": 2000},
                    task_id="review-web",
                )
            )
        assert backend.await_count == 1
        assert context.web_urls == {url}
    finally:
        clear_review_context("review-web")
