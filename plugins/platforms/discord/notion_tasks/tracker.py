"""Persistent map: notion_page_id -> Discord message locations + status meta.

JSON file via atomic_json_write, mirrors gateway/platforms/helpers.py
ThreadParticipationTracker.

Each record holds the page's title/status meta plus a ``locations`` map keyed
by Discord message_id, so the same task shown in a channel AND in a thread can
be kept in sync. The tracker is written by the live gateway at CLICK time (it
captures the pre-edit message content as ``orig_content`` for a clean undo);
the send paths do not need to write it, which keeps cross-process delivery
(``hermes send`` standalone HTTP) free of tracker races.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UNSET = object()


class NotionTaskTracker:
    def __init__(self, max_tracked: int = 1000):
        self._max = max_tracked
        self._tasks: dict[str, dict[str, Any]] = self._load()

    def _state_path(self) -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "discord_notion_tasks.json"

    def _load(self) -> dict:
        path = self._state_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            # Corruption must be visible (fail-fast), not silently swallowed.
            # Don't crash the gateway: back the bad file up and start fresh —
            # buttons still work because handle_action re-fetches at click time.
            logger.error("notion task tracker: corrupt state at %s (%s); backing up and resetting",
                         path, exc)
            try:
                path.rename(path.with_suffix(".corrupt"))
            except Exception:
                logger.warning("notion task tracker: could not back up corrupt state", exc_info=True)
            return {}
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        logger.error("notion task tracker: state at %s is not a dict; resetting", path)
        return {}

    def _save(self) -> None:
        from utils import atomic_json_write
        if len(self._tasks) > self._max:
            items = sorted(self._tasks.items(), key=lambda kv: kv[1].get("updated_at", 0))
            self._tasks = dict(items[-self._max:])
        atomic_json_write(self._state_path(), self._tasks, indent=2)

    def get(self, page_id) -> dict | None:
        return self._tasks.get(str(page_id))

    def locations(self, page_id) -> list[dict]:
        rec = self._tasks.get(str(page_id)) or {}
        return list((rec.get("locations") or {}).values())

    def upsert_meta(self, page_id, *, title=None, status_kind=None,
                    original_status=_UNSET, done=_UNSET) -> None:
        rec = self._tasks.setdefault(str(page_id), {})
        if title is not None:
            rec["title"] = title
        if status_kind is not None:
            rec["status_kind"] = status_kind
        if original_status is not _UNSET:
            rec["original_status"] = original_status
        if done is not _UNSET:
            rec["done"] = bool(done)
        rec["updated_at"] = time.time()
        self._save()

    def add_location(self, page_id, *, message_id, channel_id, orig_content=None) -> None:
        """Record a Discord message showing this task.

        ``orig_content`` is the message's pre-edit text, captured once for undo;
        a later call with ``None`` never clobbers an already-stored original.
        """
        rec = self._tasks.setdefault(str(page_id), {})
        locs = rec.setdefault("locations", {})
        mid = str(message_id)
        loc = locs.setdefault(mid, {"message_id": mid})
        loc["channel_id"] = str(channel_id)
        if orig_content is not None and loc.get("orig_content") is None:
            loc["orig_content"] = orig_content
        rec["updated_at"] = time.time()
        self._save()

    def upsert_followthrough(self, page_id, *, message_id=None, channel_id=None,
                             selected_by=None, choice_kind=None, choice_text=None,
                             thread_id=None, thread_url=None, state=None) -> None:
        """Persist the latest Task Clarify follow-through state for recovery.

        This is not the authority for whether a task has a Discord thread —
        Notion's thread binding is. It records the card decision so a stale
        card click can be explained after a gateway restart.
        """
        rec = self._tasks.setdefault(str(page_id), {})
        rec["followthrough"] = {
            "card_message_id": str(message_id or ""),
            "card_channel_id": str(channel_id or ""),
            "selected_by": str(selected_by or ""),
            "selected_at": time.time(),
            "choice_kind": str(choice_kind or ""),
            "choice_text": str(choice_text or ""),
            "target_thread_id": str(thread_id or ""),
            "target_thread_url": str(thread_url or ""),
            "state": str(state or ""),
        }
        rec["updated_at"] = time.time()
        self._save()

    def upsert_task_clarify_snapshot(self, page_id, *, context=None,
                                     primary_choices=None,
                                     other_enabled=None) -> None:
        """Persist the original authored Task Clarify card body.

        Once a card is marked selected/done/dropped, the visible embed no longer
        carries the full 1/2/3 choice list. Store the pre-edit card shape so an
        undo/reopen can restore the exact authored brief instead of rebuilding
        from the already-mutated embed.
        """
        rec = self._tasks.setdefault(str(page_id), {})
        snap = rec.setdefault("task_clarify_card", {})
        if context is not None:
            snap["context"] = str(context)
        if primary_choices is not None:
            choices = []
            for item in list(primary_choices or [])[:3]:
                if not isinstance(item, dict):
                    continue
                choices.append({
                    "label": str(item.get("label") or ""),
                    "description": str(item.get("description") or ""),
                })
            snap["primary_choices"] = choices
        if other_enabled is not None:
            snap["other_enabled"] = bool(other_enabled)
        rec["updated_at"] = time.time()
        self._save()

    def task_clarify_snapshot(self, page_id) -> dict:
        rec = self._tasks.get(str(page_id)) or {}
        snap = rec.get("task_clarify_card") or {}
        return dict(snap) if isinstance(snap, dict) else {}
