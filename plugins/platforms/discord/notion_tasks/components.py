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

_ACTIONS = "done|undo|snooze|hold|drop|resume|open_thread|rename_thread|choice1|choice2|choice3|other"
CUSTOM_ID_RE = rf"ntask:(?:v1:)?(?P<action>{_ACTIONS}):(?P<page_id>[0-9a-f]{{32}})"

# Legacy full-text labels. Only used when no number is available — i.e. the
# DynamicItem rebuilt from custom_id after a restart, whose label is never
# rendered (it exists purely to route the click).
LABEL_DONE = "✓ 完成"
LABEL_UNDO = "↩︎ 撤销"
LABEL_SNOOZE = "⏰ 稍后提醒"

_ACTION_MARK = {
    "choice1": "1.",
    "choice2": "2.",
    "choice3": "3.",
    "other": "Other",
    "done": "✓",
    "undo": "↩",
    "snooze": "⏰",
    "hold": "⏸",
    "drop": "🗑",
    "resume": "继续",
    "open_thread": "🧵",
    "rename_thread": "改名",
}
_LEGACY_LABEL = {
    "choice1": "1.",
    "choice2": "2.",
    "choice3": "3.",
    "other": "Other",
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
    "choice1": 1,
    "choice2": 2,
    "choice3": 2,
    "other": 2,
    "done": 3,
    "undo": 2,
    "snooze": 2,
    "hold": 2,
    "drop": 4,
    "resume": 1,
    "open_thread": 1,
    "rename_thread": 2,
}

_PRIMARY_CHOICE_ACTIONS = ("choice1", "choice2", "choice3")
_ROUTINE_ACTIONS = {"open_thread", "snooze", "hold", "drop", "done"}
_SHORT_LABEL_BY_ROUTINE_ACTION = {
    "open_thread": "🧵",
    "snooze": "⏰",
    "hold": "⏸",
    "drop": "🗑",
    "done": "✓",
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
    if action in _ROUTINE_ACTIONS:
        return f"{_ACTION_MARK[action]}{num}"
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


def _clean_text(value, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _first_text(*values) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _compact_page_id(value: str | None) -> str:
    chars = "".join(ch for ch in str(value or "") if ch.lower() in "0123456789abcdef")
    return chars[-32:] if len(chars) >= 32 else ""


def _task_clarify_page_id(card: dict) -> str:
    return _compact_page_id(card.get("notionTaskId") or card.get("page_id") or card.get("notionTaskUrl"))


def _task_clarify_title(card: dict) -> str:
    title = _clean_text(card.get("notionTaskTitle") or card.get("title"), "Task")
    if len(title) > 90:
        title = title[:89] + "…"
    return title


def _choice_lines(choices: list[dict]) -> list[str]:
    lines: list[str] = []
    for idx, choice in enumerate((choices or [])[:3], start=1):
        label = _clean_text(choice.get("label"), f"选项 {idx}")
        description = _clean_text(choice.get("description"), "选择后我会在任务子区里按这个方向继续。")
        lines.append(f"{idx}. **{label}** — {description}")
    return lines


def _card_thread_url(card: dict) -> str:
    thread = card.get("thread") if isinstance(card.get("thread"), dict) else {}
    return _first_text(
        card.get("threadUrl"),
        card.get("discordThreadUrl"),
        card.get("targetThreadUrl"),
        thread.get("url") if thread else "",
    )


def _selected_choice_text(card: dict) -> str:
    selected = card.get("selectedChoice") or card.get("selected_choice") or {}
    if not isinstance(selected, dict):
        return _clean_text(selected)
    return _first_text(selected.get("text"), selected.get("label"), selected.get("value"))


def _followthrough_status_text(state: str) -> str:
    if state == "continued":
        return "已在子区继续"
    if state == "following_through":
        return "正在接到子区"
    if state == "failed":
        return "接到子区失败"
    if state == "done":
        return "已完成"
    if state == "dropped":
        return "已弃置"
    if state == "snoozed":
        return "已暂挂 / 延后提醒"
    if state == "selected":
        return "已记录选择"
    return "已选择"


def task_clarify_embed(card: dict) -> dict:
    """Render a Discord-native task decision card.

    The long explanation and the actual 1/2/3 choice text live in the embed
    body. Buttons stay short (`1.`, `2.`, `3.`, `Other`) for mobile Discord.
    Routine controls are intentionally not listed as main choices.
    """
    title = _task_clarify_title(card)
    body = card.get("body") or {}
    context = _clean_text(body.get("context") or card.get("context"), "**这是什么**：任务需要你选择下一步。")
    lines = [context]
    selected_text = _selected_choice_text(card)
    if selected_text:
        lines.extend(["", f"已选择：{selected_text}"])
        state = _clean_text(card.get("followthroughState") or card.get("followthrough_state"), "selected")
        lines.append(f"状态：{_followthrough_status_text(state)}")
    choice_lines = [] if selected_text else _choice_lines(list(card.get("primaryChoices") or []))
    if choice_lines:
        lines.extend(["", "**可选下一步**", *choice_lines, "", "Other：你也可以直接写自己的方向。"])
    desc = "\n".join(lines)
    if len(desc) > _DESC_MAX:
        desc = desc[: _DESC_MAX - 1] + "…"
    return {
        "title": f"🧭 Task Clarify · {title}",
        "description": desc,
        "color": 0xE0A15D,
        "footer": {"text": "1/2/3 是智能建议；Snooze/Hold/Dropped/Done 在二级操作"},
    }


def _raw_button(action: str, page_id: str, label: str, *, style: int | None = None) -> dict:
    if action not in _ACTION_MARK:
        raise ValueError(f"unknown task button action: {action!r}")
    return {
        "type": 2,
        "style": style if style is not None else _STYLE_BY_ACTION[action],
        "label": label,
        "custom_id": make_custom_id(action, page_id),
    }


def _raw_link_button(label: str, url: str) -> dict:
    return {"type": 2, "style": 5, "label": label, "url": url}


def task_clarify_components(card: dict) -> list[dict]:
    """Raw Discord component rows for a Task Clarify card.

    Row 1 contains only main strategy choices plus Other. Routine controls live
    in the following row so they never consume the 1/2/3 choice slots.
    """
    page_id = _task_clarify_page_id(card)
    if not page_id:
        return []
    selected_text = _selected_choice_text(card)
    thread_url = _card_thread_url(card)
    primary: list[dict] = []
    if not selected_text:
        for idx, action in enumerate(_PRIMARY_CHOICE_ACTIONS, start=1):
            if len(card.get("primaryChoices") or []) >= idx:
                primary.append(_raw_button(action, page_id, f"{idx}."))
        if (card.get("otherChoice") or {}).get("enabled", True):
            primary.append(_raw_button("other", page_id, "Other"))
    rows = []
    if primary:
        rows.append({"type": 1, "components": primary[:_MAX_PER_ROW]})
    secondary = []
    for item in list(card.get("secondaryActions") or []):
        action = _clean_text(item.get("action"))
        if action in _ROUTINE_ACTIONS:
            label = _SHORT_LABEL_BY_ROUTINE_ACTION[action]
            if action == "open_thread" and thread_url:
                secondary.append(_raw_link_button(label, thread_url))
            else:
                secondary.append(_raw_button(action, page_id, label))
    if secondary:
        rows.append({"type": 1, "components": secondary[:_MAX_PER_ROW]})
    return rows


def button_component(action: str, page_id: str, num: int | None = None,
                     *, link_url: str = "") -> dict:
    """One Discord button component (raw JSON) for the HTTP send path."""
    style = _STYLE_BY_ACTION.get(action)
    if style is None:
        raise ValueError(f"unknown task button action: {action!r}")
    if action == "open_thread" and link_url:
        return {"type": 2, "style": 5, "label": numbered_label(action, num),
                "url": str(link_url)}
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


def components_payload(tasks: list[tuple[str, str]], *,
                       link_url_by_page: dict[str, str] | None = None) -> list[dict]:
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
    link_url_by_page = link_url_by_page or {}
    for action, page_id in tasks:
        btn = button_component(
            action,
            page_id,
            num_of[page_id],
            link_url=link_url_by_page.get(page_id, "") if action == "open_thread" else "",
        )
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
