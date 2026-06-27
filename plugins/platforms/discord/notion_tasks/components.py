"""Pure, discord-free helpers for the task-button custom_id, the raw Discord
component JSON, and the struck-through "done" message rendering.

Kept free of any ``discord`` import so the standalone HTTP send path (which runs
in a CLI process that may not have a live discord.py client) can build button
payloads without pulling in the gateway's discord dependency.
"""
from __future__ import annotations

CUSTOM_ID_RE = r"ntask:(?P<action>done|undo):(?P<page_id>[0-9a-f]{32})"

LABEL_DONE = "✓ 完成"
LABEL_UNDO = "↩︎ 撤销"

# Discord button styles: 3 = green (success), 2 = grey (secondary)
_STYLE_DONE = 3
_STYLE_UNDO = 2

# Discord limits: max 5 action rows per message, max 5 buttons per row.
_MAX_ROWS = 5
_MAX_PER_ROW = 5


def make_custom_id(action: str, page_id: str) -> str:
    return f"ntask:{action}:{page_id}"


def button_component(action: str, page_id: str) -> dict:
    """One Discord button component (raw JSON) for the HTTP send path."""
    if action == "done":
        return {"type": 2, "style": _STYLE_DONE, "label": LABEL_DONE,
                "custom_id": make_custom_id("done", page_id)}
    return {"type": 2, "style": _STYLE_UNDO, "label": LABEL_UNDO,
            "custom_id": make_custom_id("undo", page_id)}


def components_payload(tasks: list[tuple[str, str]]) -> list[dict]:
    """Pack ``(action, page_id)`` pairs into Discord action-row JSON.

    Returns ``[]`` for no tasks. Silently caps at 25 buttons (5 rows × 5) —
    callers that could exceed it must log the drop themselves.
    """
    rows: list[dict] = []
    row: list[dict] = []
    for action, page_id in tasks:
        row.append(button_component(action, page_id))
        if len(row) == _MAX_PER_ROW:
            rows.append({"type": 1, "components": row})
            row = []
            if len(rows) == _MAX_ROWS:
                return rows
    if row and len(rows) < _MAX_ROWS:
        rows.append({"type": 1, "components": row})
    return rows


def strike_done(original: str) -> str:
    """Render a completed task message: prepend ✅ and strike each line.

    Discord strikethrough does not span newlines, so wrap each non-blank line
    individually. Markdown links inside ``~~[t](u)~~`` stay clickable, so the
    original reminder text and Notion URL are preserved (just struck), not
    destroyed. Undo restores the stored original verbatim — never by parsing
    this back out.
    """
    body = (original or "").rstrip()
    if not body.strip():
        return "✅ 已完成"
    lines = body.split("\n")
    struck = "\n".join(f"~~{ln}~~" if ln.strip() else ln for ln in lines)
    return f"✅ {struck}"
