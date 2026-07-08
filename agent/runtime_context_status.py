"""Request-local runtime context status blocks for model calls.

These helpers provide a tiny, auditable channel for Hermes operational state
that must be visible near the end of a single model request without becoming
conversation history.  The block is appended to the API-copy of the final user
message, not to the persisted transcript, so provider adapters preserve temporal
ordering even when they extract ``role=system`` messages into separate fields.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

TAG = "hermes-runtime-context"
AUDIT_FILENAME = "runtime_context_status_audit.jsonl"
_VALID_MODES = {"off", "shadow", "inject"}
_SAFE_METADATA_KEYS = {
    "compression_count",
    "trigger_reason",
    "trigger_tokens",
    "trigger_threshold_tokens",
    "trigger_message_count",
    "rough_tokens",
    "threshold_tokens",
    "near_threshold_ratio",
}


def build_pre_compression_notice() -> str:
    """Return the LLM-visible near-compression runtime status block."""
    return """<hermes-runtime-context>
[System note: The following is temporary Hermes runtime metadata, NOT new user input.]

The visible conversation is close to Hermes context compression.
No compression has happened yet for this model call, but older turns may be summarized before a future model call if the conversation keeps growing.

This block was not saved as user input.
</hermes-runtime-context>"""


def build_post_compression_notice() -> str:
    """Return the LLM-visible just-compressed runtime status block."""
    return """<hermes-runtime-context>
[System note: The following is temporary Hermes runtime metadata, NOT new user input.]

Hermes completed context compression immediately before this model call.
This status appears only on the first model call after that compression event.

Earlier conversation turns are now represented by the visible [CONTEXT COMPACTION] block.
Messages after that block are recent original conversation messages retained verbatim.

This block was not saved as user input.
</hermes-runtime-context>"""


def runtime_context_status_mode(agent: Any) -> str:
    """Return the normalized runtime-context-status mode for an agent."""
    mode = str(getattr(agent, "_runtime_context_status_mode", "off") or "off").strip().lower()
    return mode if mode in _VALID_MODES else "off"


def _audit_enabled(agent: Any) -> bool:
    return bool(getattr(agent, "_runtime_context_status_audit_enabled", True))


def _short_hash(value: Any) -> str:
    text = str(value or "")
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        key: _json_safe_scalar(metadata.get(key))
        for key in _SAFE_METADATA_KEYS
        if key in metadata
    }


def _audit_path() -> Path:
    return get_hermes_home() / "logs" / AUDIT_FILENAME


def audit_runtime_context_status(agent: Any, event: str, payload: dict[str, Any]) -> None:
    """Append a content-free runtime context status audit record.

    This intentionally never records user message text, status-block text, or
    raw dedupe keys. Audit must never affect the live turn.
    """
    if not _audit_enabled(agent):
        return
    try:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "session_id": getattr(agent, "session_id", "") or "",
            "turn_id": payload.get("turn_id") or getattr(agent, "_current_turn_id", "") or "",
            "event": event,
            "mode": runtime_context_status_mode(agent),
        }
        for key, value in payload.items():
            if key in {"content", "status", "text", "user_text", "dedupe_key"}:
                continue
            if key == "metadata" and isinstance(value, dict):
                record.update(_safe_metadata(value))
                continue
            record[key] = _json_safe_scalar(value)
        if payload.get("dedupe_key") is not None:
            record["dedupe_key_hash"] = _short_hash(payload.get("dedupe_key"))

        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:  # pragma: no cover - audit must be fail-open
        logger.debug("runtime context status audit failed: %s", exc)


def queue_runtime_context_status(
    agent: Any,
    status: str,
    *,
    kind: str,
    dedupe_key: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Queue a request-local runtime status block for later API-copy injection."""
    mode = runtime_context_status_mode(agent)
    pending = getattr(agent, "_pending_runtime_context_statuses", None)
    pending_count_before = len(pending) if isinstance(pending, list) else 0
    if mode == "off":
        audit_runtime_context_status(agent, "queue", {
            "status_kind": kind,
            "result": "disabled",
            "dedupe_key": dedupe_key,
            "pending_count_before": pending_count_before,
            "pending_count_after": pending_count_before,
            "metadata": metadata or {},
        })
        return
    if not isinstance(status, str) or not status.strip():
        audit_runtime_context_status(agent, "drop", {
            "status_kind": kind,
            "result": "dropped_empty_status",
            "dedupe_key": dedupe_key,
            "pending_count_before": pending_count_before,
            "pending_count_after": pending_count_before,
            "metadata": metadata or {},
        })
        return

    queued_keys = getattr(agent, "_queued_runtime_context_status_keys", None)
    if not isinstance(queued_keys, set):
        queued_keys = set()
        setattr(agent, "_queued_runtime_context_status_keys", queued_keys)
    if not isinstance(pending, list):
        pending = []
        setattr(agent, "_pending_runtime_context_statuses", pending)
        pending_count_before = 0

    if dedupe_key in queued_keys:
        audit_runtime_context_status(agent, "queue", {
            "status_kind": kind,
            "result": "deduped",
            "dedupe_key": dedupe_key,
            "pending_count_before": pending_count_before,
            "pending_count_after": len(pending),
            "metadata": metadata or {},
        })
        return

    pending.append({
        "kind": kind,
        "content": status.strip(),
        "dedupe_key": dedupe_key,
        "metadata": dict(metadata or {}),
    })
    queued_keys.add(dedupe_key)
    audit_runtime_context_status(agent, "queue", {
        "status_kind": kind,
        "result": "queued",
        "dedupe_key": dedupe_key,
        "pending_count_before": pending_count_before,
        "pending_count_after": len(pending),
        "status_chars": len(status.strip()),
        "metadata": metadata or {},
    })


def peek_runtime_context_statuses(agent: Any) -> list[dict[str, Any]]:
    pending = getattr(agent, "_pending_runtime_context_statuses", None)
    if not isinstance(pending, list):
        return []
    return [item for item in pending if isinstance(item, dict)]


def consume_runtime_context_statuses(agent: Any) -> list[dict[str, Any]]:
    """Return and clear queued runtime status blocks."""
    pending = peek_runtime_context_statuses(agent)
    if not pending:
        return []
    setattr(agent, "_pending_runtime_context_statuses", [])
    setattr(agent, "_queued_runtime_context_status_keys", set())
    audit_runtime_context_status(agent, "consume", {
        "result": "consumed",
        "pending_count_before": len(pending),
        "pending_count_after": 0,
    })
    return pending


def _combined_status_content(pending: Iterable[dict[str, Any]]) -> tuple[str, list[str], int, dict[str, Any]]:
    blocks: list[str] = []
    kinds: list[str] = []
    chars = 0
    merged_metadata: dict[str, Any] = {}
    for item in pending:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        blocks.append(content.strip())
        kind = str(item.get("kind") or "unknown")
        kinds.append(kind)
        chars += len(content.strip())
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            merged_metadata.update(_safe_metadata(metadata))
    return "\n\n".join(blocks), kinds, chars, merged_metadata


def _audit_injection(
    agent: Any,
    *,
    event: str,
    result: str,
    kinds: list[str],
    status_chars: int,
    target_role: str | None = None,
    target_message_index_from_end: int | None = None,
    turn_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    audit_runtime_context_status(agent, event, {
        "status_kind": ",".join(kinds) if kinds else "unknown",
        "result": result,
        "target_role": target_role,
        "target_message_index_from_end": target_message_index_from_end,
        "status_chars": status_chars,
        "turn_id": turn_id,
        "metadata": metadata or {},
    })


def inject_runtime_context_statuses(
    api_messages: list[dict],
    pending: list[dict],
    *,
    agent: Any = None,
    turn_id: str = "",
) -> bool:
    """Append queued runtime status blocks to the final string user message.

    Returns True only when ``api_messages`` was mutated.  In ``shadow`` mode the
    would-be target is audited but the payload is intentionally unchanged.
    """
    if not pending:
        return False
    agent = agent or object()
    mode = runtime_context_status_mode(agent)
    block, kinds, status_chars, metadata = _combined_status_content(pending)
    if not block:
        _audit_injection(
            agent,
            event="drop",
            result="dropped_empty_status",
            kinds=kinds,
            status_chars=0,
            turn_id=turn_id,
            metadata=metadata,
        )
        return False

    user_seen = False
    for index_from_end, msg in enumerate(reversed(api_messages)):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        user_seen = True
        if not isinstance(msg.get("content"), str):
            _audit_injection(
                agent,
                event="drop",
                result="dropped_no_string_user_message",
                kinds=kinds,
                status_chars=status_chars,
                target_role="user",
                target_message_index_from_end=index_from_end,
                turn_id=turn_id,
                metadata=metadata,
            )
            return False
        if mode == "shadow":
            _audit_injection(
                agent,
                event="inject",
                result="shadow_logged",
                kinds=kinds,
                status_chars=status_chars,
                target_role="user",
                target_message_index_from_end=index_from_end,
                turn_id=turn_id,
                metadata=metadata,
            )
            return False
        if mode != "inject":
            _audit_injection(
                agent,
                event="drop",
                result="disabled",
                kinds=kinds,
                status_chars=status_chars,
                target_role="user",
                target_message_index_from_end=index_from_end,
                turn_id=turn_id,
                metadata=metadata,
            )
            return False
        msg["content"] = msg["content"] + "\n\n" + block
        _audit_injection(
            agent,
            event="inject",
            result="injected",
            kinds=kinds,
            status_chars=status_chars,
            target_role="user",
            target_message_index_from_end=index_from_end,
            turn_id=turn_id,
            metadata=metadata,
        )
        return True

    _audit_injection(
        agent,
        event="drop",
        result="dropped_no_user_message" if not user_seen else "dropped_no_string_user_message",
        kinds=kinds,
        status_chars=status_chars,
        turn_id=turn_id,
        metadata=metadata,
    )
    return False
