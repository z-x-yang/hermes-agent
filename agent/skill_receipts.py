"""Deterministic receipts for skill content loaded before context compaction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

RECEIPT_START = "[LOADED SKILL RECEIPT v1]"
RECEIPT_END = "[/LOADED SKILL RECEIPT]"
RELOAD_INSTRUCTION = (
    "Reload each listed skill/reference with exact skill_view before relying on it "
    "after compaction."
)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _content_sha256(content: Any) -> str:
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _skill_call_arguments(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        for raw_call in message.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function") or {}
            if function.get("name") not in {"skill_view", "skill_view_readonly"}:
                continue
            call_id = str(raw_call.get("id") or "")
            if call_id:
                calls[call_id] = _json_object(function.get("arguments"))
    return calls


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _previous_receipt_entries(
    messages: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("_compressed_summary") is not True:
            continue
        text = _message_text(message.get("content"))
        if not text.lstrip().startswith("[CONTEXT COMPACTION]"):
            continue
        search_start = 0
        while True:
            start = text.find(RECEIPT_START, search_start)
            if start < 0:
                break
            payload_start = start + len(RECEIPT_START)
            end = text.find(RECEIPT_END, payload_start)
            if end < 0:
                break
            payload = _json_object(text[payload_start:end].strip())
            if payload.get("version") == 1:
                for raw in payload.get("skills") or []:
                    if not isinstance(raw, dict):
                        continue
                    name = str(raw.get("name") or "").strip()
                    if not name:
                        continue
                    files = {
                        str(item.get("path") or ""): str(item.get("content_sha256") or "")
                        for item in raw.get("loaded_files") or []
                        if isinstance(item, dict) and item.get("path")
                    }
                    entries[name] = {
                        "name": name,
                        "source": str(raw.get("source") or ""),
                        "content_sha256": raw.get("content_sha256"),
                        "loaded_files": files,
                    }
            search_start = end + len(RECEIPT_END)
    return entries


def build_loaded_skill_receipt_block(
    messages: list[dict[str, Any]],
) -> str | None:
    """Return a compact reload receipt for successful skill-view tool results."""
    call_arguments = _skill_call_arguments(messages)
    skills = _previous_receipt_entries(messages)

    for message in messages:
        if message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        if call_id not in call_arguments:
            continue
        arguments = call_arguments[call_id]
        result = _json_object(message.get("content"))
        if result.get("success") is not True:
            continue

        name = str(result.get("name") or arguments.get("name") or "").strip()
        if not name:
            continue
        source = str(result.get("skill_dir") or result.get("path") or "").strip()
        if not source and ":" in name:
            source = "plugin:" + name.split(":", 1)[0]
        entry = skills.setdefault(
            name,
            {
                "name": name,
                "source": source,
                "content_sha256": None,
                "loaded_files": {},
            },
        )
        if source:
            entry["source"] = source

        content_hash = _content_sha256(result.get("content", ""))
        file_path = str(result.get("file") or arguments.get("file_path") or "").strip()
        if file_path:
            entry["loaded_files"][file_path] = content_hash
        else:
            entry["content_sha256"] = content_hash

    if not skills:
        return None

    normalized = []
    for name in sorted(skills):
        entry = skills[name]
        normalized.append(
            {
                "name": name,
                "source": entry["source"],
                "content_sha256": entry["content_sha256"],
                "loaded_files": [
                    {"path": path, "content_sha256": digest}
                    for path, digest in sorted(entry["loaded_files"].items())
                ],
            }
        )
    payload = {
        "version": 1,
        "reload_required": True,
        "skills": normalized,
    }
    return (
        RECEIPT_START
        + "\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        + RECEIPT_END
        + "\n"
        + RELOAD_INSTRUCTION
    )
