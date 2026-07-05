from __future__ import annotations

import re

from . import detection


def _plain_rich(prop: dict | None) -> str:
    return "".join(x.get("plain_text", "") for x in (prop or {}).get("rich_text", [])).strip()


def _select_name(prop: dict | None) -> str:
    return ((prop or {}).get("select") or {}).get("name", "") or ""


def _url(prop: dict | None) -> str:
    return str((prop or {}).get("url") or "").strip()


def _project_name(page: dict) -> str:
    # Tests can inject this; production can later hydrate relation names from Workbench scanner.
    return str(page.get("project_name_for_test") or "").strip()


def _clean_title(text: str) -> str:
    text = re.sub(r"<[@#&]!?(\d+)>", "", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip(" -—:：|｜")
    return text or "Task"


def _clip(text: str, n: int = 80) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def generate_thread_title(
    page: dict,
    *,
    parent_title: str | None = None,
    source_hint: str | None = None,
) -> str:
    task_title = _clean_title(detection.page_title(page))
    project = _clean_title(_project_name(page)) if _project_name(page) else ""
    source = _clean_title(source_hint or "") if source_hint else ""
    if parent_title:
        core = f"{_clean_title(parent_title)} › {task_title}"
    else:
        core = task_title
    prefix = project or source
    return _clip(f"{prefix} · {core}" if prefix else core)


def read_thread_binding(page: dict) -> dict[str, str]:
    props = (page or {}).get("properties") or {}
    return {
        "thread_id": _plain_rich(props.get("Discord Thread ID")),
        "thread_url": _url(props.get("Discord Thread URL")),
        "title_mode": _select_name(props.get("Thread Title Mode")),
    }
