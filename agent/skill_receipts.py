"""Deterministic, bounded receipts for skills loaded around context compaction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

LEGACY_RECEIPT_START = "[LOADED SKILL RECEIPT v1]"
RECEIPT_START = "[LOADED SKILL RECEIPT v2]"
RECEIPT_END = "[/LOADED SKILL RECEIPT]"
RELOAD_INSTRUCTION = (
    "Reload only the listed skills/references still needed by the current task "
    "before relying on them after compaction."
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


def _visible_references(raw: dict[str, Any], version: int) -> list[str]:
    if version == 1:
        return sorted({
            str(item.get("path") or "").strip()
            for item in raw.get("loaded_files") or []
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        })
    return sorted({
        str(path).strip()
        for path in raw.get("references") or []
        if str(path).strip()
    })


def _receipt_entries_from_text(text: str) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for marker, expected_version in (
        (RECEIPT_START, 2),
        (LEGACY_RECEIPT_START, 1),
    ):
        start = text.find(marker)
        if start < 0:
            continue
        payload_start = start + len(marker)
        end = text.find(RECEIPT_END, payload_start)
        if end < 0:
            continue
        payload = _json_object(text[payload_start:end].strip())
        if payload.get("version") != expected_version:
            continue
        for raw in payload.get("skills") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if name:
                entries[name] = {
                    "name": name,
                    "references": _visible_references(raw, expected_version),
                }
        break
    return entries


def _latest_previous_receipt(
    messages: list[dict[str, Any]],
) -> tuple[int | None, dict[str, dict[str, Any]]]:
    """Return the latest trusted compaction boundary and its visible receipt."""
    latest_index: int | None = None
    latest_entries: dict[str, dict[str, Any]] = {}
    for index, message in enumerate(messages):
        if message.get("_compressed_summary") is not True:
            continue
        text = _message_text(message.get("content"))
        if not text.lstrip().startswith("[CONTEXT COMPACTION]"):
            continue
        latest_index = index
        latest_entries = _receipt_entries_from_text(text)
    return latest_index, latest_entries


def build_loaded_skill_receipt(
    messages: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """Build a model-minimal receipt plus a machine-only provenance audit.

    A receipt is a one-compaction lease: only successful skill loads after the
    latest trusted compaction boundary are renewed into the next receipt. Skills
    not reloaded during that interval leave the receipt but remain discoverable
    through the normal skill index.
    """
    previous_index, previous = _latest_previous_receipt(messages)
    current_messages = messages[(previous_index + 1) if previous_index is not None else 0 :]
    call_arguments = _skill_call_arguments(current_messages)
    skills: dict[str, dict[str, Any]] = {}

    for message in current_messages:
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

    normalized_audit = []
    visible_skills = []
    for name in sorted(skills):
        entry = skills[name]
        loaded_files = [
            {"path": path, "content_sha256": digest}
            for path, digest in sorted(entry["loaded_files"].items())
        ]
        normalized_audit.append({
            "name": name,
            "source": entry["source"],
            "content_sha256": entry["content_sha256"],
            "loaded_files": loaded_files,
        })
        visible_entry: dict[str, Any] = {"name": name}
        if loaded_files:
            visible_entry["references"] = [item["path"] for item in loaded_files]
        visible_skills.append(visible_entry)

    audit = {
        "version": 2,
        "previous_skill_count": len(previous),
        "active_skill_count": len(skills),
        "expired_skill_count": len(set(previous) - set(skills)),
        "skills": normalized_audit,
    }
    if not visible_skills:
        return None, audit

    payload = {"version": 2, "skills": visible_skills}
    block = (
        RECEIPT_START
        + "\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        + RECEIPT_END
        + "\n"
        + RELOAD_INSTRUCTION
    )
    return block, audit
