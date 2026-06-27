"""Orchestration for the Discord notion-task buttons.

Wires link detection, the Notion client, the tracker, and the DynamicItem
buttons. The adapter registers an instance as the module-level "active
controller" (see registry) so DynamicItem callbacks — rebuilt from custom_id
after a restart, with no view instance — can reach it.

Design: the click handler is the single authority. At click time it fetches the
page from Notion (authoritative title / Status kind / current status) and reads
the pre-edit message content from the interaction. The send paths only attach a
button; they write no tracker state. This makes the button work regardless of
which process/path delivered the message (incl. ``hermes send`` standalone HTTP)
and survives a lost/empty tracker after a restart.
"""
from __future__ import annotations

import logging

import discord

from . import detection
from .buttons import build_button
from .components import strike_done
from .outbound import detect_task_links

logger = logging.getLogger(__name__)

DEFAULT_TASKS_IDS = detection.DEFAULT_TASKS_IDS


class NotionTaskController:
    def __init__(self, *, notion, tracker, allowed_ids_getter,
                 tasks_ids=None, fetch_channel=None):
        self.notion = notion
        self.tracker = tracker
        # callable -> (user_ids set, role_ids set); read LIVE on each click so
        # it reflects the adapter's allowlist after on_ready resolves usernames
        # (the adapter's sets are empty at __init__ time — snapshotting here
        # would wrongly reject every click).
        self._allowed_ids_getter = allowed_ids_getter
        self.tasks_ids = set(tasks_ids or DEFAULT_TASKS_IDS)
        # async (channel_id) -> channel  (injected by adapter; None in tests)
        self._fetch_channel = fetch_channel

    # ---- rendering buttons on outgoing messages (live adapter path) ------
    async def render_view_for_text(self, text: str):
        """Build a discord.py View of ✓ 完成 buttons, or None if no Tasks links."""
        tasks = await detect_task_links(text or "", notion=self.notion, tasks_ids=self.tasks_ids)
        if not tasks:
            return None
        view = discord.ui.View(timeout=None)
        for page_id, title in tasks:
            view.add_item(build_button("done", page_id, title=title))
            if len(view.children) >= 25:
                logger.warning("notion task: >25 task links in one message; extras dropped")
                break
        return view

    def _build_view(self, page_id, title, done):
        view = discord.ui.View(timeout=None)
        view.add_item(build_button("undo" if done else "done", page_id, title=title))
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
        orig = self._stored_orig(page_id, mid) or content_now
        new_content = strike_done(orig) if done else orig
        try:
            await interaction.response.edit_message(
                content=new_content, view=self._build_view(page_id, title, done))
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
                new = strike_done(orig) if done else orig
                await msg.edit(content=new, view=self._build_view(page_id, title, done))
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
            view = self._build_view(link.page_id, title, done)
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
                    content=(strike_done(base) if done else base), view=view)
                self.tracker.add_location(link.page_id, message_id=sent.id,
                                          channel_id=thread.id, orig_content=base)
            except Exception as exc:
                logger.warning("notion task: thread button send failed: %s", exc)
            return  # one task per starter message
