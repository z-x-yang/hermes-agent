"""discord.py DynamicItem button for completing/undoing a Notion task.

DynamicItem (discord.py >= 2.4) lets the page_id live in custom_id and be
re-routed after a bot restart via the template regex — exactly what a
persistent, per-task button needs (plain add_view can't, since custom_id
must be static there).
"""
from __future__ import annotations

import logging

import discord

from .components import CUSTOM_ID_RE, make_custom_id, numbered_label  # noqa: F401 (re-exported)
from .registry import get_active_controller
from .snooze import snooze_choices

logger = logging.getLogger(__name__)


class TaskActionButton(discord.ui.DynamicItem[discord.ui.Button], template=CUSTOM_ID_RE):
    def __init__(self, action: str, page_id: str, *, title: str | None = None,
                 num: int | None = None):
        self.action = action
        self.page_id = page_id
        if action == "done":
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
