"""Persistent snooze reminders for Discord Notion task buttons.

A snooze is notification state, not task state: it deliberately does not mutate
Notion due dates. The gateway polls this small JSON store and re-posts a task
reminder when a pending record becomes due, unless Notion already reports the
page as Done.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SnoozeChoice:
    value: str
    label: str
    description: str
    due_at: datetime


def _now_eastern() -> datetime:
    return datetime.now(EASTERN)


def resolve_due(value: str, *, now: datetime | None = None) -> datetime:
    """Resolve a preset value to an absolute datetime.

    ``now`` may be naive in unit tests; the returned datetime preserves that
    naive/aware style so deterministic equality remains simple.
    """
    base = now or _now_eastern()
    if value == "1h":
        return base + timedelta(hours=1)
    if value == "tonight":
        due = base.replace(hour=20, minute=30, second=0, microsecond=0)
        if due <= base:
            due += timedelta(days=1)
        return due
    if value == "tomorrow_morning":
        return (base + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
    if value == "next_monday":
        days = (7 - base.weekday()) % 7
        if days == 0:
            days = 7
        return (base + timedelta(days=days)).replace(hour=9, minute=30, second=0, microsecond=0)
    raise ValueError(f"unknown snooze preset: {value!r}")


def ceil_to_minute(dt: datetime) -> datetime:
    """Return ``dt`` at Discord/Notion UI precision without moving earlier."""
    if dt.second or dt.microsecond:
        dt = dt + timedelta(minutes=1)
    return dt.replace(second=0, microsecond=0)


def format_notion_datetime(dt: datetime) -> str:
    """Datetime string for Notion date properties.

    Notion stores date-times at minute precision. Sending seconds/microseconds
    can make a verified write look like a read-back mismatch even though the UI
    stored the intended minute, so align before writing.
    """
    return ceil_to_minute(dt).isoformat()


def _fmt_local(dt: datetime) -> str:
    # Short Chinese-friendly label. Keep weekday out of the confirmation so it
    # stays compact in Discord ephemeral responses.
    return dt.strftime("%m/%d %H:%M")


def snooze_choices(*, now: datetime | None = None) -> list[SnoozeChoice]:
    base = now or _now_eastern()
    specs = [
        ("1h", "1小时后", "短暂停一下，稍后再戳你"),
        ("tonight", "今晚 8:30", "今天晚点再提醒"),
        ("tomorrow_morning", "明早 9:30", "明天开工时提醒"),
        ("next_monday", "下周一 9:30", "下周开始时提醒"),
    ]
    return [SnoozeChoice(value, label, desc, resolve_due(value, now=base)) for value, label, desc in specs]


def notion_page_url(page_id: str) -> str:
    return f"https://app.notion.com/p/{str(page_id).replace('-', '').lower()}"


def reminder_content(title: str, page_id: str, *, status_unknown: bool = False) -> str:
    note = "\n\n（未能确认 Notion 当前状态，先提醒你一下。）" if status_unknown else ""
    return f"⏰ 稍后提醒：{title}\n\nNotion: {notion_page_url(page_id)}{note}"


class SnoozeStore:
    """Tiny JSON-backed store for pending Discord task snoozes."""

    def __init__(self, max_records: int = 2000):
        self._max = max_records
        self._records: dict[str, dict[str, Any]] = self._load()

    def _state_path(self) -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "discord_notion_task_snoozes.json"

    def _load(self) -> dict[str, dict[str, Any]]:
        path = self._state_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("notion task snooze: corrupt state at %s (%s); backing up and resetting",
                         path, exc)
            try:
                path.rename(path.with_suffix(".corrupt"))
            except Exception:
                logger.warning("notion task snooze: could not back up corrupt state", exc_info=True)
            return {}
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        logger.error("notion task snooze: state at %s is not a dict; resetting", path)
        return {}

    def _save(self) -> None:
        from utils import atomic_json_write
        if len(self._records) > self._max:
            items = sorted(self._records.items(), key=lambda kv: kv[1].get("updated_at", 0))
            self._records = dict(items[-self._max:])
        atomic_json_write(self._state_path(), self._records, indent=2)

    def schedule(
        self,
        *,
        page_id: str,
        title: str,
        due_at: float,
        channel_id: str,
        message_id: str | None,
        user_id: str | None,
        original_content: str | None,
        preset: str,
    ) -> str:
        now = time.time()
        pid = str(page_id).replace("-", "").lower()
        # One pending snooze per page is the sane default: clicking snooze again
        # updates the reminder time instead of creating duplicate nags.
        rec_id = self._pending_id_for_page(pid) or f"{pid}:{int(now * 1000)}"
        self._records[rec_id] = {
            "id": rec_id,
            "page_id": pid,
            "title": title,
            "due_at": float(due_at),
            "channel_id": str(channel_id),
            "message_id": str(message_id or ""),
            "user_id": str(user_id or ""),
            "original_content": original_content or "",
            "preset": preset,
            "status": "pending",
            "created_at": self._records.get(rec_id, {}).get("created_at", now),
            "updated_at": now,
        }
        self._save()
        return rec_id

    def _pending_id_for_page(self, page_id: str) -> str | None:
        for rec_id, rec in self._records.items():
            if rec.get("page_id") == page_id and rec.get("status") == "pending":
                return rec_id
        return None

    def pending_for(self, page_id: str) -> dict | None:
        """The page's pending snooze record (copy), or None.

        Used by the task-card renderer to mark a row as ⏰ 已延后 with its due
        time. At most one pending record exists per page (see schedule()).
        """
        pid = str(page_id).replace("-", "").lower()
        rec_id = self._pending_id_for_page(pid)
        return self.get(rec_id) if rec_id else None

    def get(self, rec_id: str) -> dict | None:
        rec = self._records.get(str(rec_id))
        return dict(rec) if rec else None

    def due(self, *, now: float | None = None) -> list[dict]:
        ts = time.time() if now is None else float(now)
        due = [dict(r) for r in self._records.values()
               if r.get("status") == "pending" and float(r.get("due_at", 0) or 0) <= ts]
        return sorted(due, key=lambda r: float(r.get("due_at", 0) or 0))

    def mark_sent(self, rec_id: str, *, sent_message_id: str | None = None) -> None:
        rec = self._records.get(str(rec_id))
        if not rec:
            return
        rec["status"] = "sent"
        rec["sent_message_id"] = str(sent_message_id or "")
        rec["sent_at"] = time.time()
        rec["updated_at"] = rec["sent_at"]
        self._save()

    def mark_failed(self, rec_id: str, *, error: str) -> None:
        rec = self._records.get(str(rec_id))
        if not rec:
            return
        rec["last_error"] = str(error)
        rec["attempts"] = int(rec.get("attempts", 0) or 0) + 1
        rec["updated_at"] = time.time()
        self._save()

    def cancel(self, rec_id: str, *, reason: str) -> None:
        rec = self._records.get(str(rec_id))
        if not rec:
            return
        rec["status"] = "cancelled"
        rec["cancel_reason"] = reason
        rec["updated_at"] = time.time()
        self._save()

    def cancel_pending(self, page_id: str, *, reason: str) -> int:
        pid = str(page_id).replace("-", "").lower()
        count = 0
        for rec in self._records.values():
            if rec.get("page_id") == pid and rec.get("status") == "pending":
                rec["status"] = "cancelled"
                rec["cancel_reason"] = reason
                rec["updated_at"] = time.time()
                count += 1
        if count:
            self._save()
        return count
