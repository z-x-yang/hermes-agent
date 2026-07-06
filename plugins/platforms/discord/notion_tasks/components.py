"""Pure, discord-free helpers for the task-button custom_id, the raw Discord
component JSON, and the task-card embed rendering.

Kept free of any ``discord`` import so the standalone HTTP send path (which runs
in a CLI process that may not have a live discord.py client) can build button
payloads without pulling in the gateway's discord dependency.

Invariant: the adapter only creates and edits structures built HERE (the task
card embed + button rows). The upstream message body is never edited — buttons
carry only a number, and per-task state renders inside the card.
"""
from __future__ import annotations

_ACTIONS = "done|undo|snooze|hold|drop|resume|open_thread|rename_thread"
CUSTOM_ID_RE = rf"ntask:(?:v1:)?(?P<action>{_ACTIONS}):(?P<page_id>[0-9a-f]{{32}})"

# Legacy full-text labels. Only used when no number is available — i.e. the
# DynamicItem rebuilt from custom_id after a restart, whose label is never
# rendered (it exists purely to route the click).
LABEL_DONE = "✓ 完成"
LABEL_UNDO = "↩︎ 撤销"
LABEL_SNOOZE = "⏰ 稍后提醒"

_ACTION_MARK = {
    "done": "✓",
    "undo": "↩",
    "snooze": "⏰",
    "hold": "暂挂",
    "drop": "弃置",
    "resume": "继续",
    "open_thread": "🧵",
    "rename_thread": "改名",
}
_LEGACY_LABEL = {
    "done": LABEL_DONE,
    "undo": LABEL_UNDO,
    "snooze": LABEL_SNOOZE,
    "hold": "暂挂",
    "drop": "弃置",
    "resume": "继续",
    "open_thread": "打开子区",
    "rename_thread": "改名",
}

_NUM_KEYCAP = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣",
               6: "6️⃣", 7: "7️⃣", 8: "8️⃣", 9: "9️⃣", 10: "🔟"}
# Discord embed hard limits: description 4096. Row titles are capped below;
# masked-link URLs add ~58 chars/row, so a degenerate many-task card (>~20
# rows) can still hit the cap — the final description truncation then clips
# trailing rows (a clipped link renders as plain text, nothing breaks).
_ROW_TITLE_MAX = 150
_DESC_MAX = 4096

# Discord button styles: 1 = blurple (primary), 2 = grey (secondary),
# 3 = green (success), 4 = red (danger).
_STYLE_BY_ACTION = {
    "done": 3,
    "undo": 2,
    "snooze": 2,
    "hold": 2,
    "drop": 4,
    "resume": 1,
    "open_thread": 1,
    "rename_thread": 2,
}

# Discord limits: max 5 action rows per message, max 5 buttons per row.
_MAX_ROWS = 5
_MAX_PER_ROW = 5


def make_custom_id(action: str, page_id: str) -> str:
    if action not in _ACTION_MARK:
        raise ValueError(f"unknown task button action: {action!r}")
    return f"ntask:v1:{action}:{page_id}"


def numbered_label(action: str, num: int | None) -> str:
    """Button label: choice number only (``✓ 3``) — full text lives in the card.

    ``num=None`` falls back to the legacy full-text label (restart-rebuilt
    DynamicItems have no number and their label is never displayed).
    """
    if action not in _ACTION_MARK:
        raise ValueError(f"unknown task button action: {action!r}")
    if num is None:
        return _LEGACY_LABEL[action]
    return f"{_ACTION_MARK[action]} {num}"


def _num_emoji(n: int) -> str:
    return _NUM_KEYCAP.get(n, f"{n}.")


_NOTION_URL_PREFIX = "https://www.notion.so/"


def _row_title(r: dict) -> str:
    """Row title, rendered as a masked link to the Notion page when the row
    carries a ``page_id`` (clicking the title opens the task). ASCII brackets in
    the title are swapped for fullwidth ``［］``: a raw ``]`` would end the
    masked-link text early, and Discord renders backslash-escapes literally
    inside link text (``\\[`` shows as ``\\[``), so escaping is not an option.
    Truncation happens BEFORE linking so the visible length stays bounded and the
    URL is never cut."""
    title = str(r.get("title") or "").strip() or "(untitled)"
    if len(title) > _ROW_TITLE_MAX:
        title = title[: _ROW_TITLE_MAX - 1] + "…"
    page_id = str(r.get("page_id") or "")
    if not page_id:
        return title
    safe = title.replace("[", "［").replace("]", "］")
    return f"[{safe}]({_NOTION_URL_PREFIX}{page_id})"


def task_card_embed(rows: list[dict]) -> dict | None:
    """Task-card embed as a raw dict — the ONE renderer both send paths share.

    ``rows``: ``{"num": int, "title": str, "state": "open"|"done"|"snoozed",
    "due_label": str | None, "page_id": str | None}``. Row order == button
    numbering. State renders only inside this card (done rows strike THEIR OWN
    line, never the message body). Rows with a ``page_id`` render the title as
    a masked link to the Notion page. Returns None for no rows.
    """
    if not rows:
        return None
    lines = []
    for r in rows:
        title = _row_title(r)
        state = r.get("state")
        if state == "done":
            lines.append(f"✅ ~~{title}~~")
        elif state == "dropped":
            lines.append(f"🛑 已弃置 · ~~{title}~~")
        elif state == "snoozed":
            lines.append(f"⏰ 已延后·{r.get('due_label') or '?'} · {title}")
        else:
            lines.append(f"{_num_emoji(int(r['num']))} {title}")
    desc = "\n".join(lines)
    if len(desc) > _DESC_MAX:
        desc = desc[: _DESC_MAX - 1] + "…"
    done_n = sum(1 for r in rows if r.get("state") == "done")
    card_title = "📋 任务" if not done_n else f"📋 任务 · {done_n}/{len(rows)} 已完成"
    return {"title": card_title, "description": desc, "color": 0x4E8CD8,
            "footer": {"text": "点下方对应编号的按钮操作"}}


def button_component(action: str, page_id: str, num: int | None = None) -> dict:
    """One Discord button component (raw JSON) for the HTTP send path."""
    style = _STYLE_BY_ACTION.get(action)
    if style is None:
        raise ValueError(f"unknown task button action: {action!r}")
    return {"type": 2, "style": style, "label": numbered_label(action, num),
            "custom_id": make_custom_id(action, page_id)}


def action_pairs_for_task_card(page_ids: list[str]) -> list[tuple[str, str]]:
    """Default compact V0 Workbench controls for task-card links."""
    capped = list(page_ids)[:_MAX_ROWS * _MAX_PER_ROW]
    pairs: list[tuple[str, str]] = []
    for page_id in capped:
        pairs.append(("open_thread", page_id))
        pairs.append(("done", page_id))
        pairs.append(("hold", page_id))
        pairs.append(("drop", page_id))
        pairs.append(("snooze", page_id))
        if len(pairs) >= _MAX_ROWS * _MAX_PER_ROW:
            break
    return pairs[:_MAX_ROWS * _MAX_PER_ROW]


def action_pairs_with_snooze(page_ids: list[str]) -> list[tuple[str, str]]:
    """Compatibility alias for callers that still use the old helper name."""
    return action_pairs_for_task_card(page_ids)


def pack_group_rows(group_sizes: list[int]) -> list[int] | None:
    """First-fit row index for each button group, packing whole groups into as
    few rows as fit (``≤5`` buttons/row, groups never split across rows).

    Returns one row index per group, or ``None`` if a single group is larger
    than a row or the groups can't fit within the 5-row limit (caller then falls
    back to flat packing). This keeps each task's ✓/⏰ controls together while
    letting several small tasks share a row instead of one row each.
    """
    if any(sz > _MAX_PER_ROW for sz in group_sizes):
        return None
    out: list[int] = []
    row = 0
    used = 0
    for sz in group_sizes:
        if out and used + sz > _MAX_PER_ROW:
            row += 1
            used = 0
        out.append(row)
        used += sz
    if row + 1 > _MAX_ROWS:
        return None
    return out


def components_payload(tasks: list[tuple[str, str]]) -> list[dict]:
    """Pack ``(action, page_id)`` pairs into Discord action-row JSON.

    Consecutive actions for the same page are grouped together, then whole
    per-page groups are bin-packed into as few action rows as fit (``≤5``
    buttons/row) so a task's ✓ 完成 / ⏰ buttons stay together AND a few tasks
    share one row instead of each getting its own. When the groups can't be kept
    intact within the 5-row limit (e.g. 26+ single-button pages), fall back to
    flat packing (5 buttons/row), still capping at 25 buttons (5 rows × 5).

    Returns ``[]`` for no tasks. Callers that could exceed 25 must log the drop
    themselves.
    """
    if not tasks:
        return []
    # numbering: page_id first-occurrence order == task-card row order
    order: list[str] = []
    for _action, page_id in tasks:
        if page_id not in order:
            order.append(page_id)
    num_of = {pid: i + 1 for i, pid in enumerate(order)}
    # group consecutive actions belonging to the same page
    groups: list[tuple[str, list[dict]]] = []
    for action, page_id in tasks:
        btn = button_component(action, page_id, num_of[page_id])
        if groups and groups[-1][0] == page_id:
            groups[-1][1].append(btn)
        else:
            groups.append((page_id, [btn]))

    # bin-pack whole groups into rows when they fit — keeps controls together
    rowidx = pack_group_rows([len(btns) for _pid, btns in groups])
    if rowidx is not None:
        rows: list[dict] = []
        for gi, (_pid, btns) in enumerate(groups):
            while len(rows) <= rowidx[gi]:
                rows.append({"type": 1, "components": []})
            rows[rowidx[gi]]["components"].extend(btns)
        return rows

    # too many distinct pages to keep groups intact: flat-pack, cap at 25
    rows = []
    row: list[dict] = []
    for _page_id, btns in groups:
        for btn in btns:
            row.append(btn)
            if len(row) == _MAX_PER_ROW:
                rows.append({"type": 1, "components": row})
                row = []
                if len(rows) == _MAX_ROWS:
                    return rows
    if row and len(rows) < _MAX_ROWS:
        rows.append({"type": 1, "components": row})
    return rows
