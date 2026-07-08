"""Attach the task card (numbered buttons + embed) to OUTGOING messages.

Two delivery paths must render it:
  - the live gateway adapter (``DiscordAdapter.send`` -> discord.py View/Embed),
  - the standalone Discord HTTP path (``_standalone_send``), which is what
    ``hermes send`` uses from a cron/CLI process with no live adapter — this is
    how the email-reminder cron actually pushes its task messages.

``detect_task_links`` is the shared async step (parse links, verify each is a
Tasks-DB page via Notion). ``standalone_task_payload`` builds the raw Discord
component + embed JSON for the HTTP path and imports no ``discord`` module, so
it is safe in a CLI process that has no discord.py client.
"""
from __future__ import annotations

import logging

from . import detection
from .components import action_pairs_with_snooze, components_payload, task_card_embed
from .threading import read_thread_binding

logger = logging.getLogger(__name__)


def _thread_url_from_page(page: dict) -> str:
    try:
        return str(read_thread_binding(page).get("thread_url") or "").strip()
    except Exception:
        logger.warning("notion task: failed to read Discord thread binding", exc_info=True)
        return ""


async def detect_task_link_items(message: str, *, notion, tasks_ids=None) -> list[dict]:
    """Return task-card items with send-time thread-link metadata."""
    if not detection.has_notion_link(message):
        return []
    ids = tasks_ids or detection.DEFAULT_TASKS_IDS
    out: list[dict] = []
    seen: set[str] = set()
    for link in detection.extract_notion_links(message):
        if link.page_id in seen:
            continue
        try:
            page = await notion.get_page(link.page_id)
        except Exception as exc:
            logger.warning("notion task: get_page(%s) failed, no button: %s", link.page_id, exc)
            continue
        if not detection.is_task_page(page, ids):
            continue
        seen.add(link.page_id)
        out.append({
            "page_id": link.page_id,
            "title": link.anchor or detection.page_title(page),
            "thread_url": _thread_url_from_page(page),
        })
    return out


async def detect_task_links(message: str, *, notion, tasks_ids=None) -> list[tuple[str, str]]:
    """Return ``[(page_id, title), ...]`` for Tasks-DB links in ``message``.

    Each candidate link is verified against Notion (``get_page`` + parent match);
    non-task pages (docs/projects) and unreadable pages are skipped. A get_page
    failure is logged and the link dropped (no button) — intentional graceful
    degradation, never a silent success.
    """
    return [(str(item["page_id"]), str(item["title"]))
            for item in await detect_task_link_items(message, notion=notion, tasks_ids=tasks_ids)]


async def standalone_task_payload(message: str) -> tuple[list[dict], dict | None]:
    """``(action_rows, card_embed)`` raw JSON for the HTTP send path.

    ``([], None)`` when the message has no task links. Send-time card rows are
    always "open" — this process has no tracker/snooze state; the click path
    rebuilds the card from live state.

    Builds its own NotionClient (reads NOTION_API_KEY from the CLI process env).
    Never raises into the send path: any failure logs and yields no attachments
    so the message still goes out (the card can't appear, but delivery must not
    break).
    """
    if not detection.has_notion_link(message):
        return [], None
    try:
        from .notion_client import NotionClient
        items = await detect_task_link_items(message, notion=NotionClient())
    except Exception:
        logger.warning("notion task: standalone task-card build failed; sending without card",
                       exc_info=True)
        return [], None
    if not items:
        return [], None
    if len(items) > 25:
        logger.warning("notion task: %d task links in one message; only first 25 get buttons",
                       len(items))
    page_ids = [str(item["page_id"]) for item in items]
    pairs = action_pairs_with_snooze(page_ids)
    if len(pairs) < len(items[:25]) * 2:
        logger.warning("notion task: not enough component slots for snooze on every task")
    thread_url_by_page = {
        str(item["page_id"]): str(item.get("thread_url") or "")
        for item in items
        if item.get("thread_url")
    }
    rows = [{"num": i, "title": title, "state": "open", "due_label": None,
             "page_id": pid}
            for i, (pid, title) in enumerate(
                [(str(item["page_id"]), str(item["title"])) for item in items],
                start=1,
            )]
    return components_payload(pairs, link_url_by_page=thread_url_by_page), task_card_embed(rows)
