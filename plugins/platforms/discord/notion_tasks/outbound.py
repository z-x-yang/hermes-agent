"""Attach task buttons to OUTGOING messages.

Two delivery paths must render the ✓ 完成 button:
  - the live gateway adapter (``DiscordAdapter.send`` -> discord.py View), and
  - the standalone Discord HTTP path (``_standalone_send``), which is what
    ``hermes send`` uses from a cron/CLI process with no live adapter — this is
    how the email-reminder cron actually pushes its task messages.

``detect_task_links`` is the shared async step (parse links, verify each is a
Tasks-DB page via Notion). ``standalone_task_components`` builds the raw Discord
component JSON for the HTTP path and imports no ``discord`` module, so it is safe
in a CLI process that has no discord.py client.
"""
from __future__ import annotations

import logging

from . import detection
from .components import components_payload

logger = logging.getLogger(__name__)


async def detect_task_links(message: str, *, notion, tasks_ids=None) -> list[tuple[str, str]]:
    """Return ``[(page_id, title), ...]`` for Tasks-DB links in ``message``.

    Each candidate link is verified against Notion (``get_page`` + parent match);
    non-task pages (docs/projects) and unreadable pages are skipped. A get_page
    failure is logged and the link dropped (no button) — intentional graceful
    degradation, never a silent success.
    """
    if not detection.has_notion_link(message):
        return []
    ids = tasks_ids or detection.DEFAULT_TASKS_IDS
    out: list[tuple[str, str]] = []
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
        title = link.anchor or detection.page_title(page)
        out.append((link.page_id, title))
    return out


async def standalone_task_components(message: str) -> list[dict]:
    """Raw Discord action-row JSON (``[]`` if none) for the HTTP send path.

    Builds its own NotionClient (reads NOTION_API_KEY from the CLI process env).
    Never raises into the send path: any failure logs and yields no buttons so
    the message still goes out (the buttons can't appear, but delivery must not
    break).
    """
    if not detection.has_notion_link(message):
        return []
    try:
        from .notion_client import NotionClient
        tasks = await detect_task_links(message, notion=NotionClient())
    except Exception:
        logger.warning("notion task: standalone component build failed; sending without buttons",
                       exc_info=True)
        return []
    if not tasks:
        return []
    if len(tasks) > 25:
        logger.warning("notion task: %d task links in one message; only first 25 get buttons",
                       len(tasks))
    return components_payload([("done", pid) for pid, _ in tasks])
