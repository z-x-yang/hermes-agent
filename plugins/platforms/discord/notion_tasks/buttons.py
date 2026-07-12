"""discord.py DynamicItem button for completing/undoing a Notion task.

DynamicItem (discord.py >= 2.4) lets the page_id live in custom_id and be
re-routed after a bot restart via the template regex — exactly what a
persistent, per-task button needs (plain add_view can't, since custom_id
must be static there).
"""
from __future__ import annotations

import inspect
import logging

import discord

from . import detection
from .components import CUSTOM_ID_RE, make_custom_id, numbered_label  # noqa: F401 (re-exported)
from .registry import get_active_controller
from .snooze import snooze_choices

logger = logging.getLogger(__name__)


class OtherDirectionModal(discord.ui.Modal):
    def __init__(self, page_id: str):
        super().__init__(title="自定义执行方向", timeout=None)
        self.page_id = page_id
        self.direction_input = discord.ui.TextInput(
            label="你想怎么处理？",
            placeholder="比如：先查旧邮件，再起草回复；或者先整理父任务背景。",
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=1000,
        )
        self.add_item(self.direction_input)

    async def on_submit(self, interaction: discord.Interaction):
        ctrl = get_active_controller()
        if ctrl is None:
            logger.warning("notion task Other modal submitted but no active controller")
            await interaction.response.send_message("功能暂不可用，请稍后再试。", ephemeral=True)
            return
        direction = str(getattr(self.direction_input, "value", "") or "").strip()
        await ctrl.handle_other_direction_submit(self.page_id, direction, interaction)


class HoldReasonModal(discord.ui.Modal):
    def __init__(self, page_id: str):
        super().__init__(title="暂挂原因", timeout=None)
        self.page_id = page_id
        self.reason_input = discord.ui.TextInput(
            label="为什么先暂挂？（可留空）",
            placeholder="比如：等老板反馈；等对方回信；现在不是优先级。",
            style=discord.TextStyle.paragraph,
            required=False,
            max_length=500,
        )
        self.add_item(self.reason_input)

    async def on_submit(self, interaction: discord.Interaction):
        ctrl = get_active_controller()
        if ctrl is None:
            logger.warning("notion task Hold modal submitted but no active controller")
            await interaction.response.send_message("功能暂不可用，请稍后再试。", ephemeral=True)
            return
        reason = str(getattr(self.reason_input, "value", "") or "").strip()
        await ctrl.handle_hold_reason_submit(self.page_id, reason, interaction)


class SlashHoldReasonModal(discord.ui.Modal):
    def __init__(self):
        super().__init__(title="暂挂原因", timeout=None)
        self.reason_input = discord.ui.TextInput(
            label="为什么先暂挂？（可留空）",
            placeholder="比如：等老板反馈；等对方回信；现在不是优先级。",
            style=discord.TextStyle.paragraph,
            required=False,
            max_length=500,
        )
        self.add_item(self.reason_input)

    async def on_submit(self, interaction: discord.Interaction):
        ctrl = get_active_controller()
        if ctrl is None:
            logger.warning("notion task slash Hold modal submitted but no active controller")
            await interaction.response.send_message("功能暂不可用，请稍后再试。", ephemeral=True)
            return
        reason = str(getattr(self.reason_input, "value", "") or "").strip()
        await ctrl.handle_slash_hold_submit(reason, interaction)


class TaskBindModal(discord.ui.Modal):
    def __init__(self):
        super().__init__(title="绑定 Notion Task", timeout=None)
        self.task_input = discord.ui.TextInput(
            label="Notion Task URL 或 page id",
            placeholder="https://www.notion.so/... 或 32 位 page id",
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=500,
        )
        self.add_item(self.task_input)

    async def on_submit(self, interaction: discord.Interaction):
        ctrl = get_active_controller()
        if ctrl is None:
            logger.warning("notion task bind modal submitted but no active controller")
            await interaction.response.send_message("功能暂不可用，请稍后再试。", ephemeral=True)
            return
        task_ref = str(getattr(self.task_input, "value", "") or "").strip()
        await ctrl.handle_slash_bind_submit(task_ref, interaction)


class TaskBindSearchModal(discord.ui.Modal):
    def __init__(self):
        super().__init__(title="搜索 Notion Task", timeout=None)
        self.query_input = discord.ui.TextInput(
            label="任务标题关键词",
            placeholder="比如：Alice / rebuttal / camp slides",
            style=discord.TextStyle.short,
            required=True,
            max_length=100,
        )
        self.add_item(self.query_input)

    async def on_submit(self, interaction: discord.Interaction):
        ctrl = get_active_controller()
        if ctrl is None:
            logger.warning("notion task bind search modal submitted but no active controller")
            await interaction.response.send_message("功能暂不可用，请稍后再试。", ephemeral=True)
            return
        query = str(getattr(self.query_input, "value", "") or "").strip()
        await ctrl.handle_slash_bind_search_submit(query, interaction)


class TaskActionButton(discord.ui.DynamicItem[discord.ui.Button], template=CUSTOM_ID_RE):
    def __init__(self, action: str, page_id: str, *, title: str | None = None,
                 num: int | None = None):
        self.action = action
        self.page_id = page_id
        if action in ("done", "ack"):
            style = discord.ButtonStyle.green
        elif action in ("choice1", "resume", "open_thread"):
            style = discord.ButtonStyle.primary
        elif action == "drop":
            style = discord.ButtonStyle.danger
        elif action in ("choice2", "choice3", "other", "undo", "snooze", "hold", "rename_thread"):
            style = discord.ButtonStyle.secondary
        else:
            raise ValueError(f"unknown task action: {action!r}")
        # Label is the row number only ("✓ 3") — the full task text lives in the
        # card embed. num=None (from_custom_id rebuilds) gets the legacy label,
        # which is never rendered anywhere.
        super().__init__(
            discord.ui.Button(label=numbered_label(action, num), style=style,
                              custom_id=make_custom_id(action, page_id))
        )

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], match["page_id"])

    async def callback(self, interaction: discord.Interaction):
        ctrl = get_active_controller()
        if ctrl is None:
            logger.warning("notion task button clicked but no active controller")
            try:
                await interaction.response.send_message("功能暂不可用，请稍后再试。", ephemeral=True)
            except Exception:
                logger.exception("notion task: failed to send no-controller fallback response")
            return
        if self.action == "other":
            send_modal = getattr(getattr(interaction, "response", None), "send_modal", None)
            if not callable(send_modal):
                await interaction.response.send_message("当前客户端不能打开自定义输入框。", ephemeral=True)
                return
            result = send_modal(OtherDirectionModal(self.page_id))
            if inspect.isawaitable(result):
                await result
            return
        if self.action == "hold":
            send_modal = getattr(getattr(interaction, "response", None), "send_modal", None)
            if not callable(send_modal):
                await interaction.response.send_message("当前客户端不能打开暂挂原因输入框。", ephemeral=True)
                return
            result = send_modal(HoldReasonModal(self.page_id))
            if inspect.isawaitable(result):
                await result
            return
        await ctrl.handle_action(self.action, self.page_id, interaction)


def build_button(action: str, page_id: str, *, title: str | None = None,
                 num: int | None = None) -> TaskActionButton:
    return TaskActionButton(action, page_id, title=title, num=num)


def build_snooze_select(
    page_id: str,
    *,
    source_channel_id: str,
    source_message_id: str,
    source_content: str,
    now=None,
):
    """Build the ephemeral select menu shown after clicking ⏰ 稍后提醒."""
    options = [
        discord.SelectOption(label=choice.label, value=choice.value, description=choice.description)
        for choice in snooze_choices(now=now)
    ]
    select = discord.ui.Select(
        placeholder="什么时候再提醒？",
        options=options,
        custom_id=f"ntask:snooze-select:{page_id}",
    )

    async def _callback(interaction: discord.Interaction):
        ctrl = get_active_controller()
        if ctrl is None:
            logger.warning("notion task snooze selected but no active controller")
            await interaction.response.send_message("功能暂不可用，请稍后再试。", ephemeral=True)
            return
        values = getattr(select, "values", None) or []
        if not values:
            await interaction.response.send_message("没有选中提醒时间。", ephemeral=True)
            return
        await ctrl.handle_snooze_choice(
            page_id,
            values[0],
            interaction,
            source_channel_id=source_channel_id,
            source_message_id=source_message_id,
            source_content=source_content,
        )

    select.callback = _callback
    return select


def _task_bind_option(page: dict):
    page_id = detection.normalize_id((page or {}).get("id")) or str((page or {}).get("id") or "")
    title = detection.page_title(page) or "(untitled)"
    status, _kind = detection.read_status(page)
    label = title[:100]
    desc = " · ".join(x for x in (status, page_id[:8]) if x)[:100]
    return discord.SelectOption(label=label, value=page_id[:100], description=desc or None)


def build_task_bind_picker_view(pages: list[dict], *, query: str = ""):
    """Build the /task-bind picker: recent/search results plus a search button."""
    view = discord.ui.View(timeout=300)
    options = [_task_bind_option(p) for p in (pages or []) if detection.normalize_id((p or {}).get("id"))]
    if options:
        select = discord.ui.Select(
            placeholder="选择要绑定的 Notion Task",
            options=options[:25],
            custom_id="ntask:bind-select",
        )

        async def _select_callback(interaction: discord.Interaction):
            ctrl = get_active_controller()
            if ctrl is None:
                logger.warning("notion task bind selected but no active controller")
                await interaction.response.send_message("功能暂不可用，请稍后再试。", ephemeral=True)
                return
            values = getattr(select, "values", None) or []
            if not values:
                await interaction.response.send_message("没有选中 Notion Task。", ephemeral=True)
                return
            await ctrl.handle_slash_bind_submit(values[0], interaction)

        select.callback = _select_callback
        view.add_item(select)

    button = discord.ui.Button(
        label="重新搜索" if str(query or "").strip() else "搜索任务",
        style=discord.ButtonStyle.secondary,
        custom_id="ntask:bind-search",
    )

    async def _button_callback(interaction: discord.Interaction):
        send_modal = getattr(getattr(interaction, "response", None), "send_modal", None)
        if not callable(send_modal):
            await interaction.response.send_message("当前客户端不能打开搜索框。", ephemeral=True)
            return
        result = send_modal(TaskBindSearchModal())
        if inspect.isawaitable(result):
            await result

    button.callback = _button_callback
    view.add_item(button)
    return view
