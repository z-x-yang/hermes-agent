"""Orchestration for the Discord notion-task buttons.

Wires link detection, the Notion client, the tracker, and the DynamicItem
buttons. The adapter registers an instance as the module-level "active
controller" (see registry) so DynamicItem callbacks — rebuilt from custom_id
after a restart, with no view instance — can reach it.

Design: the click handler is the single authority. At click time it fetches the
page from Notion (authoritative title / Status kind / current status) and reads
the pre-edit message content from the interaction. The send paths only attach
the card; they write no tracker state. This makes the button work regardless of
which process/path delivered the message (incl. ``hermes send`` standalone HTTP)
and survives a lost/empty tracker after a restart.

Rendering invariant: all per-task state (done strikethrough, snoozed marker,
numbering) lives in the task-card EMBED built from ``components.task_card_embed``.
The upstream message body is never edited — no edit call in this module may pass
``content=``. Buttons carry only the row number ("✓ 3"); the full task title
lives in the matching card row.
"""
from __future__ import annotations

import inspect
import logging
import re
from datetime import datetime
from typing import Any

import discord

from . import detection
from .buttons import (
    SlashHoldReasonModal,
    build_button,
    build_snooze_select,
    build_task_bind_picker_view,
)
from .components import numbered_label, pack_group_rows, task_card_embed, task_clarify_embed
from .snooze import (
    create_snooze_cron,
    EASTERN,
    SnoozeStore,
    ceil_to_minute,
    format_notion_datetime,
    reminder_content,
    resolve_due,
)
from .outbound import detect_task_link_items, detect_task_links

logger = logging.getLogger(__name__)

DEFAULT_TASKS_IDS = detection.DEFAULT_TASKS_IDS


def _plain_rich(prop: dict | None) -> str:
    return "".join(x.get("plain_text", "") for x in (prop or {}).get("rich_text", [])).strip()


def _date_start(prop: dict | None) -> str | None:
    return ((prop or {}).get("date") or {}).get("start")


def _hold_due_label(page: dict) -> str | None:
    props = (page or {}).get("properties") or {}
    start = _date_start(props.get("Next Check"))
    if not start:
        return None
    try:
        return datetime.fromisoformat(start).strftime("%m/%d %H:%M")
    except Exception:
        return start[:16]


_CHOICE_ACTION_RE = re.compile(r"^choice(?P<num>[123])$")
_CHOICE_LINE_RE = re.compile(r"(?m)^\s*(?P<num>[123])\.\s+\*\*(?P<label>[^*]+)\*\*\s+—\s+(?P<desc>[^\n]+)")
_TASK_CLARIFY_SECONDARY_ACTIONS = (
    {"action": "open_thread"},
    {"action": "snooze"},
    {"action": "hold"},
    {"action": "drop"},
    {"action": "done"},
)
_TASK_CLARIFY_SHORT_LABELS = {
    "choice1": "1.",
    "choice2": "2.",
    "choice3": "3.",
    "other": "Other",
    "ack": "已接手",
    "open_thread": "🧵",
    "undo": "↩",
    "snooze": "⏰",
    "hold": "⏸",
    "drop": "🗑",
    "done": "✓",
}


def _embed_description(embed) -> str:
    if isinstance(embed, dict):
        return str(embed.get("description") or "")
    return str(getattr(embed, "description", "") or "")


def _embed_title(embed) -> str:
    if isinstance(embed, dict):
        return str(embed.get("title") or "")
    return str(getattr(embed, "title", "") or "")


def _task_clarify_embed_from_interaction(interaction):
    msg = getattr(interaction, "message", None)
    for embed in getattr(msg, "embeds", []) or []:
        title = _embed_title(embed)
        desc = _embed_description(embed)
        if title.startswith("🧭 Task Clarify") or "**可选下一步**" in desc or _CHOICE_LINE_RE.search(desc):
            return embed
    return None


def _task_clarify_context(embed) -> str:
    desc = _embed_description(embed).strip()
    if "\n\n**可选下一步**" in desc:
        return desc.split("\n\n**可选下一步**", 1)[0].strip()
    return desc or "这个任务需要你选一个下一步。"


def _task_clarify_choices(embed) -> list[dict]:
    desc = _embed_description(embed)
    choices = []
    for line in _CHOICE_LINE_RE.finditer(desc):
        choices.append({"label": line.group("label").strip(), "description": line.group("desc").strip()})
    return choices[:3]


def _strip_task_clarify_status_lines(context: str) -> str:
    lines = []
    for line in str(context or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("已选择：") or stripped.startswith("状态："):
            continue
        lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip() or "这个任务需要你选一个下一步。"


def _thread_url_from_binding(interaction, *, thread_id: str = "", thread_url: str = "") -> str:
    if thread_url:
        return thread_url
    if not thread_id:
        return ""
    guild = getattr(interaction, "guild", None) or getattr(getattr(interaction, "message", None), "guild", None)
    guild_id = str(getattr(guild, "id", "") or "")
    return f"https://discord.com/channels/{guild_id}/{thread_id}" if guild_id else ""


def _thread_url_from_page(page: dict, interaction) -> str:
    try:
        from .threading import read_thread_binding
        binding = read_thread_binding(page)
    except Exception:
        return ""
    return _thread_url_from_binding(
        interaction,
        thread_id=str(binding.get("thread_id") or ""),
        thread_url=str(binding.get("thread_url") or ""),
    )


def _thread_url_from_interaction(interaction) -> str:
    msg = getattr(interaction, "message", None)
    for row in getattr(msg, "components", []) or []:
        items = getattr(row, "children", None) or getattr(row, "components", None) or []
        for child in items:
            item = getattr(child, "item", child)
            if str(getattr(item, "label", "") or "") != _TASK_CLARIFY_SHORT_LABELS["open_thread"]:
                continue
            url = str(getattr(item, "url", "") or "").strip()
            if url:
                return url
    return ""


def _current_channel_id(interaction) -> str:
    chan = getattr(interaction, "channel", None)
    return str(getattr(chan, "id", "") or getattr(interaction, "channel_id", "") or "").strip()


def _page_id_from_ref(task_ref: str) -> str:
    text = str(task_ref or "").strip()
    if not text:
        return ""
    links = detection.extract_notion_links(text)
    if links:
        return detection.normalize_id(links[0].page_id) or ""
    return detection.normalize_id(text) or ""


def _selected_choice_from_interaction(interaction, action: str) -> dict:
    match = _CHOICE_ACTION_RE.match(action or "")
    if not match:
        return {"seed": "", "text": "", "kind": action or ""}
    num = match.group("num")
    embed = _task_clarify_embed_from_interaction(interaction)
    if embed is not None:
        for line in _CHOICE_LINE_RE.finditer(_embed_description(embed)):
            if line.group("num") == num:
                label = line.group("label").strip()
                detail = line.group("desc").strip()
                return {
                    "seed": f"Selected option: {num}. {label}\nChoice detail: {detail}",
                    "text": f"{label} — {detail}",
                    "kind": action,
                }
    return {"seed": f"Selected option: {num}.", "text": f"{num}.", "kind": action}


def _selected_choice_seed(interaction, action: str) -> str:
    return _selected_choice_from_interaction(interaction, action).get("seed", "")


def _card_state_from_page(page: dict) -> dict:
    status, _kind = detection.read_status(page)
    if status == "Done":
        return {"state": "done", "due_label": None}
    if status == "Dropped":
        return {"state": "dropped", "due_label": None}
    if status == "Hold":
        return {"state": "snoozed", "due_label": _hold_due_label(page)}
    return {"state": "open", "due_label": None}


def _state_is_undoable(state: dict | None) -> bool:
    return (state or {}).get("state") in {"done", "dropped"}


def _interaction_response_done(interaction) -> bool:
    response = getattr(interaction, "response", None)
    is_done = getattr(response, "is_done", None)
    if callable(is_done):
        try:
            return bool(is_done())
        except Exception:
            logger.warning("notion task: response.is_done() failed", exc_info=True)
    return False


async def _defer_message_update(interaction) -> None:
    """ACK a component interaction before Notion/network work.

    Discord shows "interaction failed" if the first response takes too long.
    A deferred message update is invisible to the user and lets the controller
    edit the original message once the verified Notion write/read-back finishes.
    """
    if _interaction_response_done(interaction):
        return
    response = getattr(interaction, "response", None)
    defer = getattr(response, "defer", None)
    if not callable(defer):
        return
    try:
        result = defer(thinking=False)
    except TypeError:
        result = defer()
    except Exception:
        logger.warning("notion task: failed to defer Discord interaction", exc_info=True)
        return
    try:
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.warning("notion task: failed to defer Discord interaction", exc_info=True)


async def _defer_slash_response(interaction, *, ephemeral: bool = True) -> None:
    """ACK a slash command before Notion/network work.

    Unlike component updates, application commands need a deferred channel
    response (`thinking=True`) so the later notice can be sent through followup
    without hitting Discord's short initial-response timeout.
    """
    if _interaction_response_done(interaction):
        return
    response = getattr(interaction, "response", None)
    defer = getattr(response, "defer", None)
    if not callable(defer):
        return
    try:
        result = defer(thinking=True, ephemeral=ephemeral)
    except TypeError:
        try:
            result = defer(thinking=True)
        except TypeError:
            result = defer()
    except Exception:
        logger.warning("notion task: failed to defer Discord slash interaction", exc_info=True)
        return
    try:
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.warning("notion task: failed to defer Discord slash interaction", exc_info=True)


async def _send_interaction_notice(interaction, content: str, **kwargs) -> None:
    if not _interaction_response_done(interaction):
        result = interaction.response.send_message(content, **kwargs)
        if inspect.isawaitable(result):
            await result
        return
    followup = getattr(interaction, "followup", None)
    send = getattr(followup, "send", None)
    if callable(send):
        result = send(content, **kwargs)
        if inspect.isawaitable(result):
            await result
        return
    logger.warning("notion task: no interaction followup available for notice: %s", content)


async def _send_interaction_modal(interaction, modal) -> None:
    send_modal = getattr(getattr(interaction, "response", None), "send_modal", None)
    if not callable(send_modal):
        await _send_interaction_notice(interaction, "当前客户端不能打开输入框。", ephemeral=True)
        return
    result = send_modal(modal)
    if inspect.isawaitable(result):
        await result


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


def _thread_title_version(page: dict) -> int:
    raw = (((page or {}).get("properties") or {}).get("Thread Title Version") or {}).get("number")
    if isinstance(raw, (int, float)) and raw >= 0:
        return int(raw) + 1
    return 1


def _discord_thread_deliver(interaction, *, fallback_channel_id: str = "") -> str:
    chan = getattr(interaction, "channel", None) or getattr(getattr(interaction, "message", None), "channel", None)
    thread_id = str(getattr(chan, "id", "") or fallback_channel_id or "").strip()
    parent = getattr(chan, "parent", None)
    parent_id = str(
        getattr(chan, "parent_id", "")
        or getattr(parent, "id", "")
        or fallback_channel_id
        or ""
    ).strip()
    if parent_id and thread_id and parent_id != thread_id:
        return f"discord:{parent_id}:{thread_id}"
    if thread_id:
        return f"discord:{thread_id}"
    return "origin"


async def _edit_interaction_message(interaction, **kwargs) -> None:
    if not _interaction_response_done(interaction):
        result = interaction.response.edit_message(**kwargs)
        if inspect.isawaitable(result):
            await result
        return
    edit_original = getattr(interaction, "edit_original_response", None)
    if callable(edit_original):
        result = edit_original(**kwargs)
        if inspect.isawaitable(result):
            await result
        return
    msg = getattr(interaction, "message", None)
    edit = getattr(msg, "edit", None)
    if callable(edit):
        result = edit(**kwargs)
        if inspect.isawaitable(result):
            await result
        return
    raise RuntimeError("Discord interaction was already ACKed and cannot edit original message")


class NotionTaskController:
    def __init__(self, *, notion, tracker, allowed_ids_getter,
                 tasks_ids=None, fetch_channel=None, snoozes=None, now_fn=None):
        self.notion = notion
        self.tracker = tracker
        self.snoozes = snoozes or SnoozeStore()
        # callable -> (user_ids set, role_ids set); read LIVE on each click so
        # it reflects the adapter's allowlist after on_ready resolves usernames
        # (the adapter's sets are empty at __init__ time — snapshotting here
        # would wrongly reject every click).
        self._allowed_ids_getter = allowed_ids_getter
        self.tasks_ids = set(tasks_ids or DEFAULT_TASKS_IDS)
        # async (channel_id) -> channel  (injected by adapter; None in tests)
        self._fetch_channel = fetch_channel
        self._now_fn = now_fn or datetime.now
        self.adapter: Any = None

    def _mark_thread_title_managed(self, thread_id: str) -> None:
        """Prevent session auto-title from overwriting a Task-managed thread name."""
        thread_id = str(thread_id or "").strip()
        if not thread_id:
            return
        marker = getattr(getattr(self, "adapter", None), "mark_auto_titled_thread", None)
        if not callable(marker):
            return
        try:
            marker(thread_id)
        except Exception:
            logger.warning(
                "notion task: failed to persist title lock for Discord thread %s",
                thread_id,
                exc_info=True,
            )

    async def resolve_task_for_discord_interaction(self, interaction, task_ref: str | None = None) -> dict:
        """Resolve a slash-command interaction to one authoritative Notion Task."""
        explicit_id = _page_id_from_ref(task_ref or "")
        if explicit_id:
            if not hasattr(self.notion, "get_page"):
                raise RuntimeError("Notion client must provide get_page for explicit task resolution")
            page = await self.notion.get_page(explicit_id)
            if not detection.is_task_page(page, self.tasks_ids):
                raise RuntimeError("提供的页面不是 Notion Tasks 里的任务。")
            return {
                "page_id": explicit_id,
                "page": page,
                "title": detection.page_title(page),
                "source": "explicit_ref",
            }

        thread_id = _current_channel_id(interaction)
        if not thread_id:
            raise RuntimeError("当前频道没有可用的 Discord thread id，请提供 Notion Task URL 或 page id。")
        finder = getattr(self.notion, "find_task_by_discord_thread_id", None)
        if not callable(finder):
            raise RuntimeError("Notion client must provide find_task_by_discord_thread_id")
        result = finder(thread_id, self.tasks_ids)
        raw_matches = await result if inspect.isawaitable(result) else result
        matches = list(raw_matches) if isinstance(raw_matches, (list, tuple)) else []
        if not matches:
            raise RuntimeError("当前子区未绑定 Notion Task，请先用 /task-bind。")
        if len(matches) > 1:
            raise RuntimeError("当前子区匹配到多个 Notion Task，需要人工修复绑定。")
        page = matches[0]
        page_id = detection.normalize_id(page.get("id")) or ""
        if not page_id:
            raise RuntimeError("Notion 返回的 Task 缺少 page id，未改动。")
        if not detection.is_task_page(page, self.tasks_ids):
            raise RuntimeError("当前子区绑定的页面不是 Notion Tasks 里的任务。")
        return {
            "page_id": page_id,
            "page": page,
            "title": detection.page_title(page),
            "source": "current_thread_binding",
        }

    async def handle_slash_done(self, interaction):
        await _defer_slash_response(interaction, ephemeral=True)
        try:
            resolved = await self.resolve_task_for_discord_interaction(interaction)
            page_id = resolved["page_id"]
            page = resolved["page"]
            title = resolved["title"]
            _status, kind = detection.read_status(page)
            updated_page = await self.notion.set_status_verified(page_id, "Done", kind or "status")
        except Exception as exc:
            await _send_interaction_notice(interaction, f"完成失败，Notion 未确认：{exc}", ephemeral=True)
            return
        self.snoozes.cancel_pending(page_id, reason="completed")
        title = detection.page_title(updated_page) or title
        await _send_interaction_notice(interaction, f"✓ 已完成：{title}", ephemeral=True)

    async def handle_slash_hold(self, interaction):
        await _send_interaction_modal(interaction, SlashHoldReasonModal())

    async def handle_slash_snooze(self, interaction):
        await _defer_slash_response(interaction, ephemeral=True)
        try:
            resolved = await self.resolve_task_for_discord_interaction(interaction)
        except Exception as exc:
            await _send_interaction_notice(interaction, f"稍后提醒失败：{exc}", ephemeral=True)
            return
        page_id = resolved["page_id"]
        page = resolved["page"]
        status, _kind = detection.read_status(page)
        if status in {"Done", "Dropped"}:
            self.snoozes.cancel_pending(page_id, reason="already_terminal")
            await _send_interaction_notice(interaction, "这个任务已经结束，不再提醒。", ephemeral=True)
            return
        msg = getattr(interaction, "message", None)
        chan_id = _current_channel_id(interaction)
        view = discord.ui.View(timeout=300)
        view.add_item(build_snooze_select(
            page_id,
            source_channel_id=chan_id,
            source_message_id=str(getattr(msg, "id", "") or ""),
            source_content=str(getattr(msg, "content", "") or ""),
            now=self._now_fn(),
        ))
        await _send_interaction_notice(interaction, "稍后多久提醒？", view=view, ephemeral=True)

    async def handle_slash_hold_submit(self, reason: str, interaction):
        await _defer_slash_response(interaction, ephemeral=True)
        reason = str(reason or "").strip()
        if len(reason) > 500:
            reason = reason[:499] + "…"
        try:
            resolved = await self.resolve_task_for_discord_interaction(interaction)
            page_id = resolved["page_id"]
            title = resolved["title"]
            page = await self.notion.set_hold_verified(
                page_id,
                next_check=None,
                reason=reason,
                waiting_for=None,
            )
        except Exception as exc:
            await _send_interaction_notice(interaction, f"暂挂失败，Notion 未确认：{exc}", ephemeral=True)
            return
        self.snoozes.cancel_pending(page_id, reason="manual_hold")
        title = detection.page_title(page) or title
        await _send_interaction_notice(interaction, f"⏸ 已暂挂：{title}", ephemeral=True)

    async def handle_slash_reopen(self, interaction):
        await _defer_slash_response(interaction, ephemeral=True)
        try:
            resolved = await self.resolve_task_for_discord_interaction(interaction)
            page_id = resolved["page_id"]
            title = resolved["title"]
            page = await self.notion.reopen_verified(page_id)
        except Exception as exc:
            await _send_interaction_notice(interaction, f"重新打开失败，Notion 未确认：{exc}", ephemeral=True)
            return
        self.snoozes.cancel_pending(page_id, reason="reopened")
        title = detection.page_title(page) or title
        await _send_interaction_notice(interaction, f"↩ 已重新打开：{title}", ephemeral=True)

    async def handle_slash_bind(self, interaction, task_ref: str = ""):
        if str(task_ref or "").strip():
            await self.handle_slash_bind_submit(task_ref, interaction)
            return
        await _defer_slash_response(interaction, ephemeral=True)
        await self._send_slash_bind_picker(interaction, query="")

    async def handle_slash_bind_search_submit(self, query: str, interaction):
        await _defer_slash_response(interaction, ephemeral=True)
        await self._send_slash_bind_picker(interaction, query=str(query or "").strip())

    async def _send_slash_bind_picker(self, interaction, *, query: str):
        finder = getattr(self.notion, "search_tasks_for_bind", None)
        if not callable(finder):
            await _send_interaction_notice(interaction, "Notion Task 搜索暂不可用。", ephemeral=True)
            return
        try:
            pages = await _maybe_await(finder(query, self.tasks_ids, limit=25))
        except Exception as exc:
            await _send_interaction_notice(interaction, f"搜索 Notion Task 失败：{exc}", ephemeral=True)
            return
        pages = list(pages or [])[:25]
        view = build_task_bind_picker_view(pages, query=query)
        if query:
            content = (
                f"选择要绑定的 Notion Task（搜索：{query}）。"
                if pages else f"没找到匹配 “{query}” 的未完成 Task，可以重新搜索。"
            )
        else:
            content = (
                "选择要绑定的 Notion Task，或点“搜索任务”。"
                if pages else "最近没有可绑定的未完成 Task，点“搜索任务”试试。"
            )
        await _send_interaction_notice(interaction, content, view=view, ephemeral=True)

    async def handle_slash_bind_submit(self, task_ref: str, interaction):
        await _defer_slash_response(interaction, ephemeral=True)
        current_thread_id = _current_channel_id(interaction)
        if not current_thread_id:
            await _send_interaction_notice(interaction, "当前频道没有 Discord thread id，未绑定。", ephemeral=True)
            return
        try:
            resolved = await self.resolve_task_for_discord_interaction(interaction, task_ref)
        except Exception as exc:
            await _send_interaction_notice(interaction, f"绑定失败：{exc}", ephemeral=True)
            return
        page_id = resolved["page_id"]
        page = resolved["page"]
        title = resolved["title"]
        from .threading import read_thread_binding
        try:
            current_matches = await _maybe_await(
                self.notion.find_task_by_discord_thread_id(current_thread_id, self.tasks_ids)
            )
        except Exception as exc:
            await _send_interaction_notice(interaction, f"检查当前子区绑定失败：{exc}", ephemeral=True)
            return
        for match in list(current_matches or []):
            match_id = detection.normalize_id(match.get("id")) or ""
            if match_id and match_id != page_id:
                await _send_interaction_notice(interaction, "当前子区已经绑定另一个 Notion Task，未覆盖。", ephemeral=True)
                return
        binding = read_thread_binding(page)
        existing_thread_id = str(binding.get("thread_id") or "")
        if existing_thread_id and existing_thread_id != current_thread_id:
            await _send_interaction_notice(interaction, "目标 Notion Task 已经绑定另一个子区，未抢占。", ephemeral=True)
            return
        if existing_thread_id == current_thread_id:
            self._mark_thread_title_managed(current_thread_id)
            await _send_interaction_notice(interaction, f"已绑定 Notion Task：{title}\nNotion: https://app.notion.com/p/{page_id}", ephemeral=True)
            return
        thread_url = _thread_url_from_binding(interaction, thread_id=current_thread_id)
        try:
            page = await self.notion.set_thread_binding_verified(
                page_id,
                thread_id=current_thread_id,
                thread_url=thread_url,
                title_mode="manual_locked",
                title_version=_thread_title_version(page),
            )
        except Exception as exc:
            await _send_interaction_notice(interaction, f"绑定失败，Notion 未确认：{exc}", ephemeral=True)
            return
        self._mark_thread_title_managed(current_thread_id)
        title = detection.page_title(page) or title
        await _send_interaction_notice(
            interaction,
            f"已绑定 Notion Task：{title}\nNotion: https://app.notion.com/p/{page_id}\n之后可直接用 /task-done /task-hold /task-snooze /task-reopen。",
            ephemeral=True,
        )

    # ---- card + view rendering -------------------------------------------
    def _card_rows(self, tasks, done_of, state_of=None):
        """Card rows for ``tasks`` (``[(page_id, title)]`` in message order).

        State priority per row: live Notion-derived state > done flag > legacy
        pending snooze record > open.
        Row order defines the button numbering, so both always match.
        """
        rows = []
        state_of = state_of or {}
        for idx, (pid, title) in enumerate(tasks, start=1):
            live_state = state_of.get(pid) or {}
            if live_state.get("state") in {"done", "snoozed", "dropped"}:
                rows.append({"num": idx, "title": title, "state": live_state["state"],
                             "due_label": live_state.get("due_label"), "page_id": pid})
                continue
            if done_of.get(pid):
                rows.append({"num": idx, "title": title, "state": "done",
                             "due_label": None, "page_id": pid})
                continue
            pending = self.snoozes.pending_for(pid)
            if pending:
                due_label = datetime.fromtimestamp(
                    float(pending.get("due_at", 0) or 0), EASTERN).strftime("%m/%d %H:%M")
                rows.append({"num": idx, "title": title, "state": "snoozed",
                             "due_label": due_label, "page_id": pid})
            else:
                rows.append({"num": idx, "title": title, "state": "open",
                             "due_label": None, "page_id": pid})
        return rows

    def _view_for_tasks(self, tasks, done_of, *, thread_url_by_page: dict[str, str] | None = None):
        """Numbered button view for ``tasks``.

        For up to five tasks, show the full Workbench action set for each open
        task. For larger multi-task messages, preserve primary Done/Undo
        coverage for up to 25 tasks and spend remaining slots on Snooze only;
        Discord gives us 25 buttons total, and losing every later task's Done
        button is worse UX than omitting secondary controls.
        """
        if len(tasks) > 25:
            logger.warning("notion task: %d task links in one message; only first 25 get buttons",
                           len(tasks))
        tasks = tasks[:25]
        thread_url_by_page = thread_url_by_page or {}
        order = [pid for pid, _title in tasks]
        num_of = {pid: i + 1 for i, pid in enumerate(order)}

        actions: list[tuple[str, str, str]] = []
        if len(tasks) <= 5:
            for pid, title in tasks:
                if done_of.get(pid):
                    actions.append(("undo", pid, title))
                else:
                    for action in ("open_thread", "done", "hold", "drop", "snooze"):
                        actions.append((action, pid, title))
        else:
            actions = [("undo" if done_of.get(pid) else "done", pid, title)
                       for pid, title in tasks]
            spare = 25 - len(actions)
            for pid, title in tasks:
                if spare <= 0:
                    break
                if not done_of.get(pid):
                    actions.append(("snooze", pid, title))
                    spare -= 1

        # Regroup per task (order == numbering) so each task's ✓/⏰ sit next to
        # each other, then bin-pack whole groups into as few rows as fit (≤5
        # buttons/row, groups never split). If the groups can't be kept intact
        # within 5 rows, leave row unset and let discord.py flow the buttons.
        grouped: dict[str, list[tuple[str, str, str]]] = {pid: [] for pid in order}
        for action, pid, title in actions:
            grouped[pid].append((action, pid, title))
        groups = [grouped[pid] for pid in order]
        rowidx = pack_group_rows([len(g) for g in groups])
        view = discord.ui.View(timeout=None)
        for gi, group in enumerate(groups):
            for action, pid, title in group:
                thread_url = thread_url_by_page.get(pid, "") if action == "open_thread" else ""
                if thread_url:
                    link_style: Any = getattr(discord.ButtonStyle, "link", 5)
                    btn = discord.ui.Button(
                        label=numbered_label(action, num_of[pid]),
                        style=link_style,
                        url=thread_url,
                    )
                else:
                    btn = build_button(action, pid, title=title, num=num_of[pid])
                if rowidx is not None:
                    try:
                        btn.row = rowidx[gi]
                    except Exception:
                        pass
                view.add_item(btn)
        return view

    def _card_embed(self, tasks, done_of, state_of=None):
        card = task_card_embed(self._card_rows(tasks, done_of, state_of=state_of))
        return discord.Embed.from_dict(card) if card else None

    def _short_button(self, action: str, page_id: str, *, row: int | None = None):
        btn = build_button(action, page_id)
        label = _TASK_CLARIFY_SHORT_LABELS.get(action)
        if label:
            try:
                btn.item.label = label
            except Exception:
                pass
        if row is not None:
            try:
                btn.row = row
            except Exception:
                pass
            try:
                btn.item.row = row
            except Exception:
                pass
        return btn

    def _task_clarify_card_parts(self, page_id: str, embed) -> tuple[str, list[dict], bool]:
        desc = _embed_description(embed)
        context = _task_clarify_context(embed)
        choices = _task_clarify_choices(embed)
        other_enabled = "Other" in desc
        if choices:
            try:
                self.tracker.upsert_task_clarify_snapshot(
                    page_id,
                    context=context,
                    primary_choices=choices,
                    other_enabled=other_enabled,
                )
            except Exception:
                logger.warning("notion task: failed to persist Task Clarify snapshot for %s", page_id, exc_info=True)
            return context, choices, other_enabled

        snapshot: dict[str, Any] = {}
        get_snapshot = getattr(self.tracker, "task_clarify_snapshot", None)
        if callable(get_snapshot):
            try:
                raw_snapshot = get_snapshot(page_id) or {}
                snapshot = raw_snapshot if isinstance(raw_snapshot, dict) else {}
            except Exception:
                logger.warning("notion task: failed to read Task Clarify snapshot for %s", page_id, exc_info=True)
                snapshot = {}
        snap_choices = list(snapshot.get("primary_choices") or [])[:3]
        if snap_choices:
            return (
                str(snapshot.get("context") or _strip_task_clarify_status_lines(context)),
                snap_choices,
                bool(snapshot.get("other_enabled", other_enabled or True)),
            )
        return _strip_task_clarify_status_lines(context), choices, other_enabled

    def _task_clarify_view(self, page_id: str, *, thread_url: str = "", embed=None,
                           selected: bool = False, task_state: dict | None = None,
                           choices: list[dict] | None = None,
                           other_enabled: bool | None = None):
        view = discord.ui.View(timeout=None)
        state_name = (task_state or {}).get("state")
        terminal = state_name in {"done", "dropped"}
        if not selected and not terminal:
            available_choices = choices if choices is not None else (_task_clarify_choices(embed) if embed is not None else [])
            choice_count = len(available_choices)
            for idx in range(1, min(choice_count, 3) + 1):
                view.add_item(self._short_button(f"choice{idx}", page_id, row=0))
            show_other = other_enabled
            if show_other is None:
                show_other = "Other" in _embed_description(embed) if embed is not None else False
            if show_other:
                view.add_item(self._short_button("other", page_id, row=0))
            if choice_count:
                view.add_item(self._short_button("ack", page_id, row=0))
        if thread_url:
            thread_btn = discord.ui.Button(
                label=_TASK_CLARIFY_SHORT_LABELS["open_thread"],
                style=getattr(discord.ButtonStyle, "link", 5),
                url=thread_url,
                row=1,
            )
            view.add_item(thread_btn)
        else:
            view.add_item(self._short_button("open_thread", page_id, row=1))
        routine = ("undo",) if terminal else ("snooze", "hold", "drop", "done")
        for action in routine:
            view.add_item(self._short_button(action, page_id, row=1))
        return view

    async def _edit_task_clarify_card(self, page_id: str, *, title: str, interaction,
                                      thread_url: str = "", selected_choice_text: str = "",
                                      followthrough_state: str = "continued",
                                      task_state: dict | None = None) -> bool:
        embed = _task_clarify_embed_from_interaction(interaction)
        if embed is None:
            return False
        selected = bool(selected_choice_text)
        state_name = (task_state or {}).get("state")
        if not selected and state_name in {"done", "dropped", "snoozed"}:
            selected_choice_text = {
                "done": "完成",
                "dropped": "弃置",
                "snoozed": "暂挂 / 延后提醒",
            }.get(state_name, "")
            followthrough_state = state_name
            selected = bool(selected_choice_text)
        context, primary_choices, other_enabled = self._task_clarify_card_parts(page_id, embed)
        card = {
            "notionTaskId": page_id,
            "notionTaskTitle": title,
            "body": {"context": context},
            "primaryChoices": primary_choices,
            "otherChoice": {"enabled": other_enabled},
            "secondaryActions": list(_TASK_CLARIFY_SECONDARY_ACTIONS),
            "threadUrl": thread_url,
        }
        if selected:
            card["selectedChoice"] = {"text": selected_choice_text}
            card["followthroughState"] = followthrough_state
        view = self._task_clarify_view(
            page_id,
            thread_url=thread_url,
            embed=embed,
            selected=selected,
            task_state=task_state,
            choices=primary_choices,
            other_enabled=other_enabled,
        )
        await _edit_interaction_message(
            interaction,
            embed=discord.Embed.from_dict(task_clarify_embed(card)),
            view=view,
        )
        return True

    def _persist_followthrough_state(self, page_id: str, *, interaction, choice_kind: str,
                                     choice_text: str, thread_id: str, thread_url: str,
                                     state: str) -> None:
        msg = getattr(interaction, "message", None)
        chan = getattr(msg, "channel", None)
        user = getattr(interaction, "user", None)
        try:
            self.tracker.upsert_followthrough(
                page_id,
                message_id=str(getattr(msg, "id", "") or ""),
                channel_id=str(getattr(chan, "id", "") or ""),
                selected_by=str(getattr(user, "display_name", "") or getattr(user, "name", "") or getattr(user, "id", "") or ""),
                choice_kind=choice_kind,
                choice_text=choice_text,
                thread_id=thread_id,
                thread_url=thread_url,
                state=state,
            )
        except Exception:
            logger.warning("notion task: failed to persist follow-through state for %s", page_id, exc_info=True)

    async def _dispatch_followthrough(self, *, interaction, thread_id: str, thread_name: str,
                                      seed_extra: str) -> bool:
        if not seed_extra:
            return True
        adapter = getattr(self, "adapter", None)
        dispatch = getattr(adapter, "dispatch_task_followthrough", None)
        if not callable(dispatch):
            return True
        text = "\n".join([
            seed_extra,
            "",
            "等价用户消息：按这个方向继续。",
        ]).strip()
        result = dispatch(interaction, thread_id=thread_id, thread_name=thread_name, text=text)
        if inspect.isawaitable(result):
            await result
        return True

    async def render_send_attachments(self, text: str):
        """(view, embed) to attach to an outgoing message, or (None, None).

        Send-time card rows are always "open" — mirroring the standalone HTTP
        path, which has no tracker/snooze state. Click-time rebuilds render the
        true per-task state.
        """
        items = await detect_task_link_items(text or "", notion=self.notion, tasks_ids=self.tasks_ids)
        tasks = [(str(item["page_id"]), str(item["title"])) for item in items]
        if not tasks:
            return None, None
        thread_url_by_page = {
            str(item["page_id"]): str(item.get("thread_url") or "")
            for item in items
            if item.get("thread_url")
        }
        view = self._view_for_tasks(
            tasks,
            {pid: False for pid, _title in tasks},
            thread_url_by_page=thread_url_by_page,
        )
        rows = [{"num": i, "title": title, "state": "open", "due_label": None,
                 "page_id": pid}
                for i, (pid, title) in enumerate(tasks, start=1)]
        return view, discord.Embed.from_dict(task_card_embed(rows))

    async def _detect_task_links_for_refresh(self, content):
        """Detect Tasks links for re-rendering, tracking Notion read failures.

        Unlike :func:`outbound.detect_task_links` (which silently drops any link
        whose ``get_page`` fails — fine for the SEND path), rebuilding a card
        AFTER a status change must not lose an unreadable sibling task's row and
        buttons; the caller has to fail visibly instead.

        Returns ``(tasks, had_read_failure)`` where ``tasks`` is the verified
        ``[(page_id, title, notion_state), ...]`` Tasks-DB pages —
        ``notion_state`` is read from Notion just now (the tracker's local flags
        go stale whenever a status changes outside a button click) — and
        ``had_read_failure`` is True if at least one candidate Notion link could
        not be read.
        """
        if not detection.has_notion_link(content):
            return [], False
        out: list[tuple[str, str, dict]] = []
        seen: set[str] = set()
        had_failure = False
        for link in detection.extract_notion_links(content or ""):
            if link.page_id in seen:
                continue
            try:
                page = await self.notion.get_page(link.page_id)
            except Exception as exc:
                logger.warning("notion task: refresh get_page(%s) failed: %s", link.page_id, exc)
                had_failure = True
                continue
            if not detection.is_task_page(page, self.tasks_ids):
                continue
            seen.add(link.page_id)
            title = link.anchor or detection.page_title(page)
            out.append((link.page_id, title, _card_state_from_page(page)))
        return out, had_failure

    async def _rebuild_card(self, content, *, page_id, title, done, state=None):
        """Rebuild a message's (view, embed) from live state after a change.

        Returns ``("ok", view, embed)`` or ``("failed", None, None)``. Any
        unreadable candidate link fails the whole rebuild — a partial card would
        silently drop that task's row and buttons (never silent, per fail-fast).

        The acted-on page is appended when the body carries no readable link to
        it (thread 📌 mirrors, bare reminder text): the click itself proves the
        message belongs to that task.
        """
        found, had_failure = await self._detect_task_links_for_refresh(content)
        if had_failure:
            return "failed", None, None
        if page_id not in {pid for pid, _t, _s in found}:
            found = found + [(page_id, title, state or {"state": "done" if done else "open", "due_label": None})]
        # The clicked page reflects THIS action's outcome; siblings use the
        # authoritative Notion status read moments ago. The tracker's local
        # ``done`` flag is deliberately NOT consulted here — it only updates on
        # button clicks, so it goes stale (and once struck every sibling) when
        # a status changes on the Notion side.
        state_of = {pid: state if (pid == page_id and state is not None) else ns
                    for pid, _t, ns in found}
        done_of = {pid: _state_is_undoable(st) for pid, st in state_of.items()}
        tasks = [(pid, t) for pid, t, _ns in found]
        return "ok", self._view_for_tasks(tasks, done_of), self._card_embed(tasks, done_of, state_of)

    @staticmethod
    def _item_row(item) -> int | None:
        row = getattr(item, "row", None)
        if row is None and hasattr(item, "item"):
            row = getattr(item.item, "row", None)
        return row

    def _try_add_select_to_spare_row(self, view, select) -> bool:
        used: set[int] = set()
        for child in getattr(view, "children", []) or []:
            row = self._item_row(child)
            if row is None:
                return False
            used.add(int(row))
        if len(used) >= 5:
            return False
        for row in range(5):
            if row not in used:
                select.row = row
                view.add_item(select)
                return True
        return False

    def _build_snooze_menu(
        self,
        page_id,
        *,
        source_channel_id,
        source_message_id,
        source_content,
        title,
        base_view=None,
    ):
        select = build_snooze_select(
            page_id,
            source_channel_id=source_channel_id,
            source_message_id=source_message_id,
            source_content=source_content,
            now=self._now_fn(),
        )
        view = base_view if base_view is not None else discord.ui.View(timeout=300)
        try:
            view.timeout = 300
        except Exception:
            pass
        preserves_source_controls = base_view is not None and self._try_add_select_to_spare_row(view, select)
        if not preserves_source_controls:
            view = discord.ui.View(timeout=300)
            view.add_item(select)

        async def _restore_on_timeout():
            await self._refresh_message_card(
                page_id,
                title=title,
                channel_id=source_channel_id,
                message_id=source_message_id,
            )

        view.on_timeout = _restore_on_timeout
        return view, preserves_source_controls

    # ---- handling clicks -------------------------------------------------
    def _authorized(self, interaction) -> bool:
        from plugins.platforms.discord.adapter import _component_check_auth
        user_ids, role_ids = self._allowed_ids_getter()
        return _component_check_auth(interaction, user_ids, role_ids)

    def _stored_orig(self, page_id, message_id):
        rec = self.tracker.get(page_id) or {}
        loc = (rec.get("locations") or {}).get(str(message_id)) or {}
        return loc.get("orig_content")

    async def _set_status_confirmed(self, page_id: str, target: str, kind: str) -> dict | None:
        """Set Status through a client method that verifies Notion read-back."""
        if not hasattr(self.notion, "set_status_verified"):
            raise RuntimeError("Notion client must provide set_status_verified for Discord task actions")
        return await self.notion.set_status_verified(page_id, target, kind)

    async def _set_dropped_confirmed(self, page_id: str, reason: str) -> dict | None:
        if not hasattr(self.notion, "set_dropped_verified"):
            raise RuntimeError("Notion client must provide set_dropped_verified for Discord task actions")
        return await self.notion.set_dropped_verified(
            page_id,
            reason=reason,
            source_fingerprint=None,
        )

    async def handle_action(self, action: str, page_id: str, interaction):
        if not self._authorized(interaction):
            await interaction.response.send_message("你没有权限操作这个任务。", ephemeral=True)
            return

        if action == "ack":
            await _defer_message_update(interaction)
            embed = _task_clarify_embed_from_interaction(interaction)
            if embed is None:
                await _send_interaction_notice(interaction, "这张消息不是可接手的任务卡。", ephemeral=True)
                return
            raw_title = _embed_title(embed).strip()
            title = raw_title.split("·", 1)[1].strip() if "·" in raw_title else raw_title
            title = title or "任务"
            thread_url = _thread_url_from_interaction(interaction)
            self._persist_followthrough_state(
                page_id,
                interaction=interaction,
                choice_kind="ack",
                choice_text="已接手",
                thread_id="",
                thread_url=thread_url,
                state="acknowledged",
            )
            await self._edit_task_clarify_card(
                page_id,
                title=title,
                interaction=interaction,
                thread_url=thread_url,
                selected_choice_text="已接手",
                followthrough_state="acknowledged",
                task_state={"state": "open", "due_label": None},
            )
            return
        if action == "snooze":
            await self.handle_snooze_menu(page_id, interaction)
            return
        if action == "hold":
            await self.handle_hold_menu(page_id, interaction)
            return
        if action == "open_thread":
            await self.handle_open_thread(page_id, interaction)
            return
        if _CHOICE_ACTION_RE.match(action or ""):
            choice = _selected_choice_from_interaction(interaction, action)
            await self.handle_open_thread(
                page_id,
                interaction,
                seed_extra=choice.get("seed", ""),
                selected_choice_text=choice.get("text", ""),
                choice_kind=choice.get("kind", action),
            )
            return
        if action == "other":
            await _send_interaction_notice(
                interaction,
                "你可以直接回复你的自定义方向；我会按你的话继续，而不是执行 1/2/3。",
                ephemeral=True,
            )
            return
        if action == "rename_thread":
            await self.handle_rename_thread(page_id, interaction)
            return
        if action not in {"done", "undo", "drop"}:
            await interaction.response.send_message(f"不认识这个任务操作：{action}", ephemeral=True)
            return

        await _defer_message_update(interaction)

        # Authoritative read at click time — works even if the tracker has no
        # record for this page (standalone-sent button / post-restart).
        try:
            page = await self.notion.get_page(page_id)
        except Exception as exc:
            logger.warning("notion task: get_page(%s) at click failed: %s", page_id, exc)
            await _send_interaction_notice(interaction,
                f"读取任务失败（Notion 暂时不可用），未改动：{exc}", ephemeral=True)
            return
        title = detection.page_title(page)
        cur_status, kind = detection.read_status(page)
        kind = kind or "select"

        rec = self.tracker.get(page_id) or {}
        terminal = action in {"done", "drop"}

        msg = getattr(interaction, "message", None)
        mid = str(getattr(msg, "id", "") or "")
        chan = getattr(msg, "channel", None)
        chan_id = str(getattr(chan, "id", "") or "")
        content_now = getattr(msg, "content", "") or ""

        if action == "done":
            target = "Done"
            # capture the pre-done status as the undo target (keep an earlier one)
            original = rec.get("original_status") or cur_status
        elif action == "drop":
            target = "Dropped"
            original = rec.get("original_status") or cur_status
        else:
            # undo: never guess. Without a recorded original status we cannot
            # safely restore — fail fast instead of defaulting (per fail-fast).
            original = rec.get("original_status")
            if not original:
                await _send_interaction_notice(interaction,
                    "无法撤销：已丢失该任务的原始状态记录，请在 Notion 中手动调整。", ephemeral=True)
                return
            target = original

        # Persist undo metadata + this location's pre-edit content BEFORE the
        # irreversible Notion write, so a tracker failure aborts before mutating
        # Notion and undo always has what it needs. `done` is written only AFTER
        # a successful write, so the flag never claims a state Notion lacks.
        try:
            self.tracker.upsert_meta(page_id, title=title, status_kind=kind,
                                     original_status=original)
            if mid:
                self.tracker.add_location(page_id, message_id=mid, channel_id=chan_id,
                                          orig_content=content_now)
        except Exception as exc:
            logger.error("notion task: tracker persist failed pre-write for %s: %s",
                         page_id, exc, exc_info=True)
            await _send_interaction_notice(interaction, "记录任务状态失败，未改动，请重试。", ephemeral=True)
            return

        try:
            if action == "drop":
                updated_page = await self._set_dropped_confirmed(page_id, "user_dropped")
            else:
                updated_page = await self._set_status_confirmed(page_id, target, kind)
        except Exception as exc:
            logger.warning("notion task: set_status(%s,%s) failed: %s", page_id, target, exc)
            await _send_interaction_notice(interaction,
                f"标记失败（Notion 暂时不可用），任务未改动：{exc}", ephemeral=True)
            return

        state = _card_state_from_page(updated_page) if updated_page else {
            "state": target.lower() if terminal else "open", "due_label": None}
        self.tracker.upsert_meta(page_id, done=terminal)  # reflect undoable terminal state
        if terminal:
            self.snoozes.cancel_pending(page_id, reason="dropped" if action == "drop" else "completed")
        orig = self._stored_orig(page_id, mid) or content_now
        if _task_clarify_embed_from_interaction(interaction) is not None:
            edited = await self._edit_task_clarify_card(
                page_id,
                title=title,
                interaction=interaction,
                thread_url=_thread_url_from_page(updated_page or page, interaction),
                task_state=state,
            )
            if edited:
                await self._sync_other(page_id, exclude_mid=mid, done=terminal, title=title, state=state)
                return
        mode, view, embed = await self._rebuild_card(
            orig, page_id=page_id, title=title, done=terminal, state=state)
        if mode == "failed":
            # Notion IS updated, but a sibling task link in THIS message couldn't
            # be read, so the card can't be rebuilt without dropping that task's
            # row/buttons. Preserve the existing message/card and tell the user
            # explicitly (fail-fast, never silent).
            logger.warning("notion task: card rebuild failed for %s; "
                           "preserving message as-is", page_id)
            await _send_interaction_notice(interaction,
                "任务已在 Notion 更新，但这条消息里另一个任务暂时读不出来，"
                "卡片没刷新（已保留原样）。请稍后再点一次或手动查看。", ephemeral=True)
            await self._sync_other(page_id, exclude_mid=mid, done=terminal, title=title, state=state)
            return
        try:
            await _edit_interaction_message(interaction, embed=embed, view=view)
        except Exception as exc:
            # Notion IS updated; only the message visual failed — surface it.
            logger.error("notion task: edit_message after write failed for %s: %s",
                         page_id, exc, exc_info=True)
            followup = getattr(interaction, "followup", None)
            if followup is not None:
                try:
                    await followup.send("任务状态已更新，但消息刷新失败，请手动查看。", ephemeral=True)
                except Exception:
                    logger.exception("notion task: failed to send post-write failure notice")
        await self._sync_other(page_id, exclude_mid=mid, done=terminal, title=title, state=state)

    async def handle_hold_menu(self, page_id: str, interaction):
        """V0 lightweight Hold: explicit button click parks the task in Notion."""
        await self.handle_hold_confirm(page_id, "manual_hold", None, interaction)

    async def handle_other_direction_submit(self, page_id: str, direction: str, interaction):
        direction = str(direction or "").strip()
        if not direction:
            await _send_interaction_notice(interaction, "没有收到自定义方向，未创建/更新子区。", ephemeral=True)
            return
        if len(direction) > 1000:
            direction = direction[:999] + "…"
        seed_extra = f"Selected option: Other\nCustom direction: {direction}"
        await self.handle_open_thread(
            page_id,
            interaction,
            seed_extra=seed_extra,
            selected_choice_text=f"Other — {direction}",
            choice_kind="other",
        )

    async def _post_seed_to_existing_thread(self, *, thread_id: str, page_id: str, page: dict, seed_extra: str) -> bool:
        if not seed_extra or not self._fetch_channel or not thread_id:
            return False
        seed = "\n".join([
            f"Task: {detection.page_title(page)}",
            f"Notion: https://app.notion.com/p/{page_id}",
            seed_extra,
            "Next action: 从这里继续处理这个任务。",
        ])
        try:
            thread = await self._fetch_channel(thread_id)
            send = getattr(thread, "send", None)
            if not callable(send):
                return False
            result = send(seed)
            if inspect.isawaitable(result):
                await result
            return True
        except Exception as exc:
            logger.warning("notion task: failed to post choice seed to existing thread %s: %s", thread_id, exc)
            return False

    async def handle_open_thread(self, page_id: str, interaction, *, seed_extra: str = "",
                                 selected_choice_text: str = "", choice_kind: str = "open_thread"):
        if not self._authorized(interaction):
            await interaction.response.send_message("你没有权限操作这个任务。", ephemeral=True)
            return
        await _defer_message_update(interaction)
        try:
            page = await self.notion.get_page(page_id)
        except Exception as exc:
            await _send_interaction_notice(interaction, f"读取任务失败，未创建子区：{exc}", ephemeral=True)
            return

        from .threading import generate_thread_title, read_thread_binding
        binding = read_thread_binding(page)
        if binding.get("thread_id"):
            thread_id = str(binding["thread_id"])
            self._mark_thread_title_managed(thread_id)
            thread_url = _thread_url_from_binding(
                interaction,
                thread_id=thread_id,
                thread_url=str(binding.get("thread_url") or ""),
            )
            seeded = await self._post_seed_to_existing_thread(
                thread_id=thread_id,
                page_id=page_id,
                page=page,
                seed_extra=seed_extra,
            )
            thread_name = detection.page_title(page)
            if self._fetch_channel:
                try:
                    thread = await self._fetch_channel(thread_id)
                    thread_name = str(getattr(thread, "name", "") or thread_name)
                except Exception:
                    logger.debug("notion task: could not fetch thread name for %s", thread_id, exc_info=True)
            state = "continued"
            if seed_extra:
                try:
                    await self._dispatch_followthrough(
                        interaction=interaction,
                        thread_id=thread_id,
                        thread_name=thread_name,
                        seed_extra=seed_extra,
                    )
                except Exception as exc:
                    logger.warning("notion task: follow-through dispatch failed for %s: %s", page_id, exc, exc_info=True)
                    state = "failed"
            if selected_choice_text or choice_kind != "open_thread":
                self._persist_followthrough_state(
                    page_id,
                    interaction=interaction,
                    choice_kind=choice_kind,
                    choice_text=selected_choice_text,
                    thread_id=thread_id,
                    thread_url=thread_url,
                    state=state,
                )
            edited = await self._edit_task_clarify_card(
                page_id,
                title=detection.page_title(page),
                interaction=interaction,
                thread_url=thread_url,
                selected_choice_text=selected_choice_text,
                followthrough_state=state,
            )
            if state == "failed":
                await _send_interaction_notice(interaction, "已记录选择，但接到子区继续失败。请进子区手动发一句继续。", ephemeral=True)
            elif not edited:
                suffix = "，已把选中策略贴到子区里。" if seeded else "。"
                await _send_interaction_notice(interaction, f"已有子区：<#{thread_id}>{suffix}", ephemeral=True)
            return

        msg = getattr(interaction, "message", None)
        adapter = getattr(self, "adapter", None)
        if msg is None or adapter is None:
            await _send_interaction_notice(interaction, "当前消息不能创建 Discord 子区。", ephemeral=True)
            return

        title = generate_thread_title(page)
        seed_lines = [
            f"Task: {detection.page_title(page)}",
            f"Notion: https://app.notion.com/p/{page_id}",
        ]
        if seed_extra:
            seed_lines.append(seed_extra)
        seed_lines.append("Next action: 从这里继续处理这个任务。")
        seed = "\n".join(seed_lines)
        result = await adapter.create_task_thread_from_message(msg, name=title, seed=seed)
        if not result.get("success"):
            await _send_interaction_notice(interaction, f"创建子区失败：{result.get('error')}", ephemeral=True)
            return

        thread_id = str(result["thread_id"])
        guild_id = getattr(getattr(msg, "guild", None), "id", "@me")
        thread_url = f"https://discord.com/channels/{guild_id}/{thread_id}"
        try:
            await self.notion.set_thread_binding_verified(
                page_id,
                thread_id=thread_id,
                thread_url=thread_url,
                title_mode="auto",
                title_version=1,
            )
        except Exception as exc:
            await _send_interaction_notice(interaction,
                f"子区已创建 <#{thread_id}>，但 Notion 绑定失败：{exc}", ephemeral=True)
            return
        state = "continued"
        if seed_extra:
            try:
                await self._dispatch_followthrough(
                    interaction=interaction,
                    thread_id=thread_id,
                    thread_name=str(result.get("thread_name") or title),
                    seed_extra=seed_extra,
                )
            except Exception as exc:
                logger.warning("notion task: follow-through dispatch failed for %s: %s", page_id, exc, exc_info=True)
                state = "failed"
        if selected_choice_text or choice_kind != "open_thread":
            self._persist_followthrough_state(
                page_id,
                interaction=interaction,
                choice_kind=choice_kind,
                choice_text=selected_choice_text,
                thread_id=thread_id,
                thread_url=thread_url,
                state=state,
            )
        edited = await self._edit_task_clarify_card(
            page_id,
            title=detection.page_title(page),
            interaction=interaction,
            thread_url=thread_url,
            selected_choice_text=selected_choice_text,
            followthrough_state=state,
        )
        if state == "failed":
            await _send_interaction_notice(interaction, "已创建子区，但接到子区继续失败。请进子区手动发一句继续。", ephemeral=True)
        elif not edited:
            await _send_interaction_notice(interaction, f"已打开任务子区：<#{thread_id}>", ephemeral=True)

    async def handle_rename_thread(self, page_id: str, interaction):
        await interaction.response.send_message("子区改名候选的实现还在下一步接线。", ephemeral=True)

    async def _refresh_message_from_interaction(self, page_id: str, title: str, interaction) -> None:
        msg = getattr(interaction, "message", None)
        chan = getattr(msg, "channel", None)
        if msg is None or chan is None:
            return
        await self._refresh_message_card(
            page_id,
            title=title,
            channel_id=str(getattr(chan, "id", "") or ""),
            message_id=str(getattr(msg, "id", "") or ""),
        )

    async def handle_hold_reason_submit(self, page_id: str, reason: str, interaction):
        reason = str(reason or "").strip()
        if len(reason) > 500:
            reason = reason[:499] + "…"
        await self.handle_hold_confirm(page_id, reason or "manual_hold", None, interaction)

    async def handle_hold_confirm(self, page_id: str, reason: str, next_check: str | None, interaction):
        if not self._authorized(interaction):
            await interaction.response.send_message("你没有权限操作这个任务。", ephemeral=True)
            return
        await _defer_message_update(interaction)
        try:
            page = await self.notion.set_hold_verified(
                page_id,
                next_check=next_check,
                reason=reason,
                waiting_for=None,
            )
        except Exception as exc:
            await _send_interaction_notice(interaction, f"暂挂失败，Notion 未确认：{exc}", ephemeral=True)
            return
        title = detection.page_title(page)
        self.snoozes.cancel_pending(page_id, reason="moved_to_notion_hold")
        state = _card_state_from_page(page)
        if _task_clarify_embed_from_interaction(interaction) is not None:
            edited = await self._edit_task_clarify_card(
                page_id,
                title=title,
                interaction=interaction,
                thread_url=_thread_url_from_page(page, interaction),
                task_state=state,
            )
            if edited:
                await _send_interaction_notice(interaction, f"已暂挂：{title}", ephemeral=True)
                return
        await _send_interaction_notice(interaction, f"已暂挂：{title}", ephemeral=True)
        await self._refresh_message_from_interaction(page_id, title, interaction)

    async def handle_snooze_menu(self, page_id: str, interaction):
        """Replace the clicked message controls with an in-place snooze picker."""
        await _defer_message_update(interaction)
        try:
            page = await self.notion.get_page(page_id)
        except Exception as exc:
            logger.warning("notion task: get_page(%s) for snooze failed: %s", page_id, exc)
            await _send_interaction_notice(interaction,
                f"读取任务失败（Notion 暂时不可用），没安排提醒：{exc}", ephemeral=True)
            return
        status, _kind = detection.read_status(page)
        if status == "Done":
            self.snoozes.cancel_pending(page_id, reason="already_done")
            await _send_interaction_notice(interaction, "这个任务已经是 Done，不再提醒你。", ephemeral=True)
            return

        msg = getattr(interaction, "message", None)
        chan = getattr(msg, "channel", None)
        source_channel_id = str(getattr(chan, "id", "") or "")
        source_message_id = str(getattr(msg, "id", "") or "")
        source_content = getattr(msg, "content", "") or ""
        title = detection.page_title(page)
        state = _card_state_from_page(page)
        clarify_embed = _task_clarify_embed_from_interaction(interaction)
        if clarify_embed is not None:
            base_view = self._task_clarify_view(
                page_id,
                thread_url=_thread_url_from_page(page, interaction),
                embed=clarify_embed,
                task_state=state,
            )
            view, preserves_source_controls = self._build_snooze_menu(
                page_id,
                source_channel_id=source_channel_id,
                source_message_id=source_message_id,
                source_content=source_content,
                title=title,
                base_view=base_view,
            )
            if not preserves_source_controls:
                await _send_interaction_notice(interaction, "稍后多久提醒？", view=view, ephemeral=True)
                return
            card = {
                "notionTaskId": page_id,
                "notionTaskTitle": title,
                "body": {"context": _task_clarify_context(clarify_embed)},
                "primaryChoices": _task_clarify_choices(clarify_embed),
                "otherChoice": {"enabled": "Other" in _embed_description(clarify_embed)},
                "secondaryActions": list(_TASK_CLARIFY_SECONDARY_ACTIONS),
                "threadUrl": _thread_url_from_page(page, interaction),
            }
            await _edit_interaction_message(
                interaction,
                embed=discord.Embed.from_dict(task_clarify_embed(card)),
                view=view,
            )
            return
        mode, base_view, embed = await self._rebuild_card(
            source_content,
            page_id=page_id,
            title=title,
            done=False,
            state=state,
        )
        view, preserves_source_controls = self._build_snooze_menu(
            page_id,
            source_channel_id=source_channel_id,
            source_message_id=source_message_id,
            source_content=source_content,
            title=title,
            base_view=base_view if mode == "ok" else None,
        )
        if not preserves_source_controls:
            await _send_interaction_notice(interaction, "稍后多久提醒？", view=view, ephemeral=True)
            return
        edit_kwargs = {"view": view}
        if mode == "ok" and embed is not None:
            edit_kwargs["embed"] = embed
        await _edit_interaction_message(interaction, **edit_kwargs)

    async def handle_snooze_choice(
        self,
        page_id: str,
        choice: str,
        interaction,
        *,
        source_channel_id: str,
        source_message_id: str,
        source_content: str,
    ):
        if not self._authorized(interaction):
            await interaction.response.send_message("你没有权限操作这个任务。", ephemeral=True)
            return
        await _defer_message_update(interaction)
        try:
            page = await self.notion.get_page(page_id)
        except Exception as exc:
            logger.warning("notion task: get_page(%s) for snooze choice failed: %s", page_id, exc)
            await _send_interaction_notice(interaction,
                f"读取任务失败（Notion 暂时不可用），没安排提醒：{exc}", ephemeral=True)
            return
        status, _kind = detection.read_status(page)
        if status == "Done":
            self.snoozes.cancel_pending(page_id, reason="already_done")
            await _send_interaction_notice(interaction, "这个任务已经是 Done，不再提醒你。", ephemeral=True)
            return

        try:
            due = resolve_due(choice, now=self._now_fn())
        except ValueError:
            await _send_interaction_notice(interaction, "这个提醒选项我不认识，没安排。", ephemeral=True)
            return
        due = ceil_to_minute(due)

        next_check = format_notion_datetime(due)
        try:
            page = await self.notion.set_hold_verified(
                page_id,
                next_check=next_check,
                reason="snoozed",
                waiting_for=None,
            )
        except Exception as exc:
            await _send_interaction_notice(interaction,
                f"稍后提醒设置失败，Notion 未确认：{exc}", ephemeral=True)
            return
        title = detection.page_title(page)
        self.snoozes.cancel_pending(page_id, reason="rescheduled")
        deliver = _discord_thread_deliver(interaction, fallback_channel_id=source_channel_id)
        user = getattr(interaction, "user", None)
        rec_id = self.snoozes.schedule_cron(
            page_id=page_id,
            title=title,
            due_at=due.timestamp(),
            channel_id=source_channel_id,
            message_id=source_message_id,
            user_id=str(getattr(user, "id", "") or ""),
            original_content=source_content,
            preset=choice,
            next_check=next_check,
            deliver=deliver,
        )
        try:
            record = self.snoozes.get(rec_id) or {}
            cron_job = create_snooze_cron(record, deliver=deliver, due_at=due)
            if not cron_job.get("id"):
                raise RuntimeError("cron create returned no job id")
            self.snoozes.attach_cron(rec_id, job=cron_job)
        except Exception as exc:
            self.snoozes.cancel(rec_id, reason="cron_create_failed")
            await _send_interaction_notice(
                interaction,
                f"提醒时间已写入 Notion，但创建 Cron 提醒失败：{exc}",
                ephemeral=True,
            )
            return
        if source_message_id:
            self.tracker.add_location(page_id, message_id=source_message_id,
                                      channel_id=source_channel_id,
                                      orig_content=source_content)
        state = _card_state_from_page(page)
        current_msg = getattr(interaction, "message", None)
        current_mid = str(getattr(current_msg, "id", "") or "")
        current_content = getattr(current_msg, "content", "") or ""
        if _task_clarify_embed_from_interaction(interaction) is not None:
            edited = await self._edit_task_clarify_card(
                page_id,
                title=title,
                interaction=interaction,
                thread_url=_thread_url_from_page(page, interaction),
                task_state=state,
            )
            if edited:
                if source_message_id and source_message_id != current_mid:
                    await self._refresh_message_card(page_id, title=title,
                                                     channel_id=source_channel_id,
                                                     message_id=source_message_id)
                await self._sync_other(page_id, exclude_mid=source_message_id or current_mid,
                                       done=False, title=title, state=state)
                return
        mode, view, embed = await self._rebuild_card(
            source_content or current_content,
            page_id=page_id,
            title=title,
            done=False,
            state=state,
        )
        if mode == "failed":
            await _send_interaction_notice(
                interaction,
                "提醒时间已写入 Notion，但这条消息里另一个任务暂时读不出来，卡片没刷新。",
                ephemeral=True,
            )
        else:
            try:
                await _edit_interaction_message(interaction, embed=embed, view=view)
            except Exception:
                logger.warning("notion task: failed to edit source after snooze choice", exc_info=True)
                await _send_interaction_notice(
                    interaction,
                    "提醒时间已写入 Notion，但消息刷新失败，请手动查看。",
                    ephemeral=True,
                )
        if source_message_id and source_message_id != current_mid:
            await self._refresh_message_card(page_id, title=title,
                                             channel_id=source_channel_id,
                                             message_id=source_message_id)
        await self._sync_other(page_id, exclude_mid=source_message_id or current_mid,
                               done=False, title=title, state=state)

    async def _refresh_message_card(self, page_id, *, title, channel_id, message_id):
        """Visual-only card refresh (e.g. a row flipping to ⏰ 已延后).

        The state change is already persisted and confirmed to the user before
        this runs; a failed refresh only leaves the card stale, so it logs and
        moves on rather than failing the interaction.
        """
        if not self._fetch_channel or not channel_id or not message_id:
            return
        try:
            channel = await self._fetch_channel(channel_id)
            msg = await channel.fetch_message(int(message_id))
            content_now = getattr(msg, "content", "") or ""
            orig = self._stored_orig(page_id, message_id) or content_now
            done = bool((self.tracker.get(page_id) or {}).get("done"))
            mode, view, embed = await self._rebuild_card(
                orig, page_id=page_id, title=title, done=done)
            if mode == "failed":
                logger.warning("notion task: card refresh for %s skipped "
                               "(sibling task unreadable)", message_id)
                return
            await msg.edit(embed=embed, view=view)
        except Exception as exc:
            logger.warning("notion task: card refresh for %s failed: %s", message_id, exc)

    async def dispatch_due_snoozes(self) -> int:
        """Send all due snooze reminders. Returns number of reminders sent."""
        if not self._fetch_channel:
            return 0
        now_ts = self._now_fn().timestamp()
        sent_count = 0
        for rec in self.snoozes.due(now=now_ts):
            rec_id = rec.get("id")
            page_id = rec.get("page_id")
            if not rec_id or not page_id:
                continue
            title = rec.get("title") or "(untitled task)"
            status_unknown = False
            try:
                page = await self.notion.get_page(page_id)
                title = detection.page_title(page) or title
                status, _kind = detection.read_status(page)
                if status is None:
                    self.snoozes.mark_failed(rec_id, error="Notion Status property unreadable")
                    continue
                if status == "Done":
                    self.snoozes.cancel(rec_id, reason="already_done")
                    continue
            except Exception as exc:
                logger.warning("notion task: get_page(%s) before snooze send failed: %s", page_id, exc)
                # Do not send on an unknown Notion state. Sending anyway can
                # violate the Done-skip contract if the task was completed while
                # Notion/API was temporarily unavailable. Keep it pending so the
                # next poll retries.
                self.snoozes.mark_failed(rec_id, error=str(exc))
                continue

            try:
                channel = await self._fetch_channel(rec.get("channel_id"))
                content = reminder_content(title, page_id, status_unknown=status_unknown)
                # The reminder card is an explicit fresh open row — NOT
                # _card_rows: this record is still "pending" until mark_sent
                # below, and must not paint its own row as ⏰ 已延后.
                rows = [{"num": 1, "title": title, "state": "open",
                         "due_label": None, "page_id": page_id}]
                msg = await channel.send(
                    content=content,
                    view=self._view_for_tasks([(page_id, title)], {page_id: False}),
                    embed=discord.Embed.from_dict(task_card_embed(rows)),
                )
                mid = str(getattr(msg, "id", "") or "")
                self.snoozes.mark_sent(rec_id, sent_message_id=mid)
                if mid:
                    self.tracker.add_location(page_id, message_id=mid,
                                              channel_id=rec.get("channel_id"),
                                              orig_content=content)
                sent_count += 1
            except Exception as exc:
                logger.warning("notion task: sending due snooze %s failed: %s", rec_id, exc)
                self.snoozes.mark_failed(rec_id, error=str(exc))
        return sent_count

    async def _sync_other(self, page_id, *, exclude_mid, done, title, state=None):
        if not self._fetch_channel:
            return
        for loc in self.tracker.locations(page_id):
            mid = str(loc.get("message_id") or "")
            if not mid or mid == exclude_mid:
                continue
            chan_id = loc.get("channel_id")
            if not chan_id:
                continue
            try:
                channel = await self._fetch_channel(chan_id)
                msg = await channel.fetch_message(int(mid))
                content_now = getattr(msg, "content", "") or ""
                self.tracker.add_location(page_id, message_id=mid, channel_id=chan_id,
                                          orig_content=content_now)
                orig = self._stored_orig(page_id, mid) or content_now
                # Same rebuild as the clicked message: a sibling location may
                # itself be a multi-task reminder, so completing one task must
                # keep the other tasks' rows and buttons intact.
                mode, view, embed = await self._rebuild_card(
                    orig, page_id=page_id, title=title, done=done, state=state)
                if mode == "failed":
                    # A sibling task link can't be read — never rebuild a partial
                    # card. Skip this edit; a later sync/click can retry.
                    logger.warning("notion task: sibling %s card rebuild failed; "
                                   "skipping edit (message preserved)", mid)
                    continue
                await msg.edit(embed=embed, view=view)
            except Exception as exc:
                logger.warning("notion task: sync %s failed: %s", mid, exc)

    # ---- thread opened on a task message --------------------------------
    async def on_thread_opened(self, thread):
        try:
            starter = thread.starter_message or await thread.fetch_message(thread.id)
        except Exception:
            starter = getattr(thread, "starter_message", None)
        if starter is None:
            return
        content = getattr(starter, "content", "") or ""
        for link in detection.extract_notion_links(content):
            try:
                page = await self.notion.get_page(link.page_id)
            except Exception as exc:
                logger.warning("notion task: thread get_page(%s) failed: %s", link.page_id, exc)
                continue
            if not detection.is_task_page(page, self.tasks_ids):
                continue
            title = link.anchor or detection.page_title(page)
            status, kind = detection.read_status(page)
            state = _card_state_from_page(page)
            done = _state_is_undoable(state)  # live Notion status is the truth, not stale tracker
            base = f"📌 {title}"
            tasks = [(link.page_id, title)]
            done_of = {link.page_id: done}
            state_of = {link.page_id: state}
            # Register the STARTER (main channel) message as a location too, so a
            # click in the thread syncs the original message (and vice versa) —
            # the main message may have been sent standalone with no tracker entry.
            self.tracker.upsert_meta(link.page_id, title=title, status_kind=kind or "select")
            starter_id = str(getattr(starter, "id", "") or "")
            if starter_id:
                starter_chan_id = str(getattr(getattr(starter, "channel", None), "id", "") or "")
                self.tracker.add_location(link.page_id, message_id=starter_id,
                                          channel_id=starter_chan_id, orig_content=content)
            try:
                sent = await thread.send(
                    content=base,
                    view=self._view_for_tasks(tasks, done_of),
                    embed=self._card_embed(tasks, done_of, state_of=state_of),
                )
                self.tracker.add_location(link.page_id, message_id=sent.id,
                                          channel_id=thread.id, orig_content=base)
            except Exception as exc:
                logger.warning("notion task: thread button send failed: %s", exc)
            return  # one task per starter message
