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

import logging
from datetime import datetime

import discord

from . import detection
from .buttons import build_button, build_snooze_select
from .components import pack_group_rows, task_card_embed
from .snooze import EASTERN, SnoozeStore, reminder_content, resolve_due
from .outbound import detect_task_links

logger = logging.getLogger(__name__)

DEFAULT_TASKS_IDS = detection.DEFAULT_TASKS_IDS


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

    # ---- card + view rendering -------------------------------------------
    def _card_rows(self, tasks, done_of):
        """Card rows for ``tasks`` (``[(page_id, title)]`` in message order).

        State priority per row: done > snoozed (live pending record) > open.
        Row order defines the button numbering, so both always match.
        """
        rows = []
        for idx, (pid, title) in enumerate(tasks, start=1):
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
        """Numbered button view for ``tasks``: one primary action per task
        (undo if done, done otherwise), then any spare component slots spent on
        snooze buttons for the earliest not-done tasks.

        Discord caps a view at 25 components (5 rows × 5). Tasks beyond 25 keep
        their card row but get no buttons — losing a task's ✓ button is worse
        than losing its ⏰, so primaries are reserved first.
        """
        if len(tasks) > 25:
            logger.warning("notion task: %d task links in one message; only first 25 get buttons",
                           len(tasks))
        tasks = tasks[:25]
        order = [pid for pid, _title in tasks]
        num_of = {pid: i + 1 for i, pid in enumerate(order)}

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

    def _card_embed(self, tasks, done_of):
        card = task_card_embed(self._card_rows(tasks, done_of))
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
        ``[(page_id, title, notion_done), ...]`` Tasks-DB pages —
        ``notion_done`` is the page's AUTHORITATIVE state read from Notion just
        now (the tracker's local ``done`` flag goes stale whenever a status
        changes outside a button click, e.g. the triage agent correcting a task
        back to To Do) — and ``had_read_failure`` is True if at least one
        candidate Notion link could not be read.
        """
        if not detection.has_notion_link(content):
            return [], False
        out: list[tuple[str, str, bool]] = []
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
            status, _kind = detection.read_status(page)
            out.append((link.page_id, title, status == "Done"))
        return out, had_failure

    async def _rebuild_card(self, content, *, page_id, title, done):
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
        if page_id not in {pid for pid, _t, _d in found}:
            found = found + [(page_id, title, done)]
        # The clicked page reflects THIS action's outcome; siblings use the
        # authoritative Notion status read moments ago. The tracker's local
        # ``done`` flag is deliberately NOT consulted here — it only updates on
        # button clicks, so it goes stale (and once struck every sibling) when
        # a status changes on the Notion side.
        done_of = {pid: (done if pid == page_id else nd) for pid, _t, nd in found}
        tasks = [(pid, t) for pid, t, _nd in found]
        return "ok", self._view_for_tasks(tasks, done_of), self._card_embed(tasks, done_of)

    def _build_snooze_menu(self, page_id, *, source_channel_id, source_message_id, source_content):
        view = discord.ui.View(timeout=300)
        view.add_item(build_snooze_select(
            page_id,
            source_channel_id=source_channel_id,
            source_message_id=source_message_id,
            source_content=source_content,
            now=self._now_fn(),
        ))
        return view

    # ---- handling clicks -------------------------------------------------
    def _authorized(self, interaction) -> bool:
        from plugins.platforms.discord.adapter import _component_check_auth
        user_ids, role_ids = self._allowed_ids_getter()
        return _component_check_auth(interaction, user_ids, role_ids)

    def _stored_orig(self, page_id, message_id):
        rec = self.tracker.get(page_id) or {}
        loc = (rec.get("locations") or {}).get(str(message_id)) or {}
        return loc.get("orig_content")

    async def handle_action(self, action: str, page_id: str, interaction):
        if not self._authorized(interaction):
            await interaction.response.send_message("你没有权限操作这个任务。", ephemeral=True)
            return

        if action == "snooze":
            await self.handle_snooze_menu(page_id, interaction)
            return

        # Authoritative read at click time — works even if the tracker has no
        # record for this page (standalone-sent button / post-restart).
        try:
            page = await self.notion.get_page(page_id)
        except Exception as exc:
            logger.warning("notion task: get_page(%s) at click failed: %s", page_id, exc)
            await interaction.response.send_message(
                f"读取任务失败（Notion 暂时不可用），未改动：{exc}", ephemeral=True)
            return
        title = detection.page_title(page)
        cur_status, kind = detection.read_status(page)
        kind = kind or "select"

        rec = self.tracker.get(page_id) or {}
        done = (action == "done")

        msg = getattr(interaction, "message", None)
        mid = str(getattr(msg, "id", "") or "")
        chan = getattr(msg, "channel", None)
        chan_id = str(getattr(chan, "id", "") or "")
        content_now = getattr(msg, "content", "") or ""

        if done:
            target = "Done"
            # capture the pre-done status as the undo target (keep an earlier one)
            original = rec.get("original_status") or cur_status
        else:
            # undo: never guess. Without a recorded original status we cannot
            # safely restore — fail fast instead of defaulting (per fail-fast).
            original = rec.get("original_status")
            if not original:
                await interaction.response.send_message(
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
            await interaction.response.send_message("记录任务状态失败，未改动，请重试。", ephemeral=True)
            return

        try:
            await self.notion.set_status(page_id, target, kind)
        except Exception as exc:
            logger.warning("notion task: set_status(%s,%s) failed: %s", page_id, target, exc)
            await interaction.response.send_message(
                f"标记失败（Notion 暂时不可用），任务未改动：{exc}", ephemeral=True)
            return

        self.tracker.upsert_meta(page_id, done=done)  # reflect Notion reality
        if done:
            self.snoozes.cancel_pending(page_id, reason="completed")
        orig = self._stored_orig(page_id, mid) or content_now
        mode, view, embed = await self._rebuild_card(
            orig, page_id=page_id, title=title, done=done)
        if mode == "failed":
            # Notion IS updated, but a sibling task link in THIS message couldn't
            # be read, so the card can't be rebuilt without dropping that task's
            # row/buttons. Preserve the existing message/card and tell the user
            # explicitly (fail-fast, never silent).
            logger.warning("notion task: card rebuild failed for %s; "
                           "preserving message as-is", page_id)
            await interaction.response.send_message(
                "任务已在 Notion 标记完成，但这条消息里另一个任务暂时读不出来，"
                "卡片没刷新（已保留原样）。请稍后再点一次或手动查看。", ephemeral=True)
            await self._sync_other(page_id, exclude_mid=mid, done=done, title=title)
            return
        try:
            await interaction.response.edit_message(embed=embed, view=view)
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
        await self._sync_other(page_id, exclude_mid=mid, done=done, title=title)

    async def handle_snooze_menu(self, page_id: str, interaction):
        """Open an ephemeral select menu for choosing a snooze preset."""
        try:
            page = await self.notion.get_page(page_id)
        except Exception as exc:
            logger.warning("notion task: get_page(%s) for snooze failed: %s", page_id, exc)
            await interaction.response.send_message(
                f"读取任务失败（Notion 暂时不可用），没安排提醒：{exc}", ephemeral=True)
            return
        status, _kind = detection.read_status(page)
        if status == "Done":
            self.snoozes.cancel_pending(page_id, reason="already_done")
            await interaction.response.send_message("这个任务已经是 Done，不再提醒你。", ephemeral=True)
            return

        msg = getattr(interaction, "message", None)
        chan = getattr(msg, "channel", None)
        view = self._build_snooze_menu(
            page_id,
            source_channel_id=str(getattr(chan, "id", "") or ""),
            source_message_id=str(getattr(msg, "id", "") or ""),
            source_content=getattr(msg, "content", "") or "",
        )
        await interaction.response.send_message("稍后多久提醒？", view=view, ephemeral=True)

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
        try:
            page = await self.notion.get_page(page_id)
        except Exception as exc:
            logger.warning("notion task: get_page(%s) for snooze choice failed: %s", page_id, exc)
            await interaction.response.send_message(
                f"读取任务失败（Notion 暂时不可用），没安排提醒：{exc}", ephemeral=True)
            return
        status, _kind = detection.read_status(page)
        if status == "Done":
            self.snoozes.cancel_pending(page_id, reason="already_done")
            await interaction.response.send_message("这个任务已经是 Done，不再提醒你。", ephemeral=True)
            return

        try:
            due = resolve_due(choice, now=self._now_fn())
        except ValueError:
            await interaction.response.send_message("这个提醒选项我不认识，没安排。", ephemeral=True)
            return

        title = detection.page_title(page)
        user = getattr(interaction, "user", None)
        self.snoozes.schedule(
            page_id=page_id,
            title=title,
            due_at=due.timestamp(),
            channel_id=source_channel_id,
            message_id=source_message_id,
            user_id=str(getattr(user, "id", "") or ""),
            original_content=source_content,
            preset=choice,
        )
        if source_message_id:
            self.tracker.add_location(page_id, message_id=source_message_id,
                                      channel_id=source_channel_id,
                                      orig_content=source_content)
        await interaction.response.send_message(
            f"好，{due.strftime('%m/%d %H:%M')} 再提醒你：{title}", ephemeral=True)
        await self._refresh_message_card(page_id, title=title,
                                         channel_id=source_channel_id,
                                         message_id=source_message_id)

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

    async def _sync_other(self, page_id, *, exclude_mid, done, title):
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
                    orig, page_id=page_id, title=title, done=done)
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
            done = (status == "Done")  # live Notion status is the truth, not stale tracker
            base = f"📌 {title}"
            tasks = [(link.page_id, title)]
            done_of = {link.page_id: done}
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
                    embed=self._card_embed(tasks, done_of),
                )
                self.tracker.add_location(link.page_id, message_id=sent.id,
                                          channel_id=thread.id, orig_content=base)
            except Exception as exc:
                logger.warning("notion task: thread button send failed: %s", exc)
            return  # one task per starter message
