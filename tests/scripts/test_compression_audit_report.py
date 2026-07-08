from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/compression_audit_report.py")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_report_summarizes_cache_and_fallbacks(tmp_path):
    logs = tmp_path / "logs"
    _write_jsonl(logs / "compression_audit.jsonl", [
        {
            "event": "context_compression",
            "compression_id": "c1",
            "session_id": "s1",
            "result": "success",
            "tokens": {"before_estimate": 1000, "after_estimate": 300},
            "summary_call": {
                "mode": "append_cached",
                "fallback_reason": None,
                "cache": {"reported": True, "read_tokens": 700, "write_tokens": 100, "hit_rate_estimate": 0.7},
            },
        },
        {
            "event": "context_compression",
            "compression_id": "c2",
            "session_id": "s1",
            "result": "fallback",
            "tokens": {"before_estimate": 1200, "after_estimate": 500},
            "summary_call": {
                "mode": "serialized_prompt",
                "fallback_from": {"fallback_reason": "append_cached_context_overflow"},
                "cache": {"reported": False, "read_tokens": None, "write_tokens": None, "hit_rate_estimate": None},
            },
        },
    ])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--home", str(tmp_path), "--last", "20"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "append_cached: 1" in result.stdout
    assert "serialized_prompt: 1" in result.stdout
    assert "cache reported: 1/2" in result.stdout
    assert "append_cached_context_overflow" in result.stdout


def test_report_show_summary_reads_redacted_sample_only(tmp_path):
    logs = tmp_path / "logs"
    _write_jsonl(logs / "compression_audit.jsonl", [{
        "event": "context_compression",
        "compression_id": "c1",
        "session_id": "s1",
        "result": "success",
        "summary_call": {"mode": "append_cached", "cache": {"reported": True}},
    }])
    _write_jsonl(logs / "compression_summary_samples.jsonl", [{
        "event": "compression_summary_sample",
        "compression_id": "c1",
        "session_id": "s1",
        "summary_excerpt": "## Primary Request and Intent\nredacted sample",
        "section_check": {"has_all_canonical_sections": True, "all_user_messages_count": 1},
        "quality_flags": [],
    }])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--home", str(tmp_path), "--compression-id", "c1", "--show-summary"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "redacted sample" in result.stdout
    assert "raw tool" not in result.stdout.lower()
