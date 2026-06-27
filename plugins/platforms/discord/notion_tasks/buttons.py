"""discord.py DynamicItem button for completing/undoing a Notion task.

DynamicItem (discord.py >= 2.4) lets the page_id live in custom_id and be
re-routed after a bot restart via the template regex — exactly what a
persistent, per-task button needs (plain add_view can't, since custom_id
must be static there).
"""
from __future__ import annotations

import logging

import discord

from .components import CUSTOM_ID_RE, LABEL_DONE, LABEL_UNDO, make_custom_id  # noqa: F401 (re-exported)
from .registry import get_active_controller

logger = logging.getLogger(__name__)


class TaskActionButton(discord.ui.DynamicItem[discord.ui.Button], template=CUSTOM_ID_RE):
    def __init__(self, action: str, page_id: str, *, title: str | None = None):
        self.action = action
        self.page_id = page_id
        if action == "done":
            label, style = (LABEL_DONE, discord.ButtonStyle.green)
        else:
            label, style = (LABEL_UNDO, discord.ButtonStyle.secondary)
        super().__init__(
            discord.ui.Button(label=label, style=style, custom_id=make_custom_id(action, page_id))
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


def build_button(action: str, page_id: str, *, title: str | None = None) -> TaskActionButton:
    return TaskActionButton(action, page_id, title=title)
