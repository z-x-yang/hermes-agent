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
from datetime import datetime
from typing import Any

import discord

from . import detection
from .buttons import build_button, build_snooze_select
from .components import pack_group_rows, task_card_embed
from .snooze import (
    EASTERN,
    SnoozeStore,
    ceil_to_minute,
    format_notion_datetime,
    reminder_content,
    resolve_due,
)
from .outbound import detect_task_links

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

    def _view_for_tasks(self, tasks, done_of):
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

    async def render_send_attachments(self, text: str):
        """(view, embed) to attach to an outgoing message, or (None, None).

        Send-time card rows are always "open" — mirroring the standalone HTTP
        path, which has no tracker/snooze state. Click-time rebuilds render the
        true per-task state.
        """
        tasks = await detect_task_links(text or "", notion=self.notion, tasks_ids=self.tasks_ids)
        if not tasks:
            return None, None
        view = self._view_for_tasks(tasks, {pid: False for pid, _title in tasks})
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

        if action == "snooze":
            await self.handle_snooze_menu(page_id, interaction)
            return
        if action == "hold":
            await self.handle_hold_menu(page_id, interaction)
            return
        if action == "open_thread":
            await self.handle_open_thread(page_id, interaction)
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

    async def handle_open_thread(self, page_id: str, interaction):
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
            await _send_interaction_notice(interaction, f"已有子区：<#{binding['thread_id']}>", ephemeral=True)
            return

        msg = getattr(interaction, "message", None)
        adapter = getattr(self, "adapter", None)
        if msg is None or adapter is None:
            await _send_interaction_notice(interaction, "当前消息不能创建 Discord 子区。", ephemeral=True)
            return

        title = generate_thread_title(page)
        seed = "\n".join([
            f"Task: {detection.page_title(page)}",
            f"Notion: https://app.notion.com/p/{page_id}",
            "Next action: 从这里继续处理这个任务。",
        ])
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

        try:
            page = await self.notion.set_hold_verified(
                page_id,
                next_check=format_notion_datetime(due),
                reason="snoozed",
                waiting_for=None,
            )
        except Exception as exc:
            await _send_interaction_notice(interaction,
                f"稍后提醒设置失败，Notion 未确认：{exc}", ephemeral=True)
            return
        title = detection.page_title(page)
        self.snoozes.cancel_pending(page_id, reason="moved_to_notion_hold")
        if source_message_id:
            self.tracker.add_location(page_id, message_id=source_message_id,
                                      channel_id=source_channel_id,
                                      orig_content=source_content)
        state = _card_state_from_page(page)
        current_msg = getattr(interaction, "message", None)
        current_mid = str(getattr(current_msg, "id", "") or "")
        current_content = getattr(current_msg, "content", "") or ""
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
