"""Pure helpers for detecting Notion task links and reading/writing Status.

No IO. Used by the Discord notion-task button feature.
"""
from __future__ import annotations

import re
from collections import namedtuple
from urllib.parse import urlparse

NotionLink = namedtuple("NotionLink", ["page_id", "anchor"])

# A Tasks page's parent reports BOTH ids under Notion API v2025-09-03; match either.
#   database_id    : 1f17a58d-229e-816f-839b-ef72f6f2ec72
#   data_source_id : 1f17a58d-229e-8144-96f3-000b99bdcf95
DEFAULT_TASKS_IDS = {"1f17a58d229e816f839bef72f6f2ec72",
                     "1f17a58d229e814496f3000b99bdcf95"}

# Only these hosts are real Notion links. A crafted URL like
# https://evil.example/notion.so/<id> must NOT match — host is the authority,
# verified via urlparse in _id_from_url, not the substring in these regexes.
_NOTION_HOSTS = {"notion.so", "www.notion.so"}

_HEXISH = r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}"
_MD = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_URL = re.compile(r"https?://[^\s)]+")
_ID_IN = re.compile(_HEXISH)


def normalize_id(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.replace("-", "").lower()
    return s if re.fullmatch(r"[0-9a-f]{32}", s) else None


def _id_from_url(url: str) -> str | None:
    # Reject anything whose HOST is not Notion (defends against
    # https://evil.example/notion.so/<id> style spoofs).
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    host = (parsed.netloc or "").lower().rsplit("@", 1)[-1].split(":")[0]
    if host not in _NOTION_HOSTS:
        return None
    # Notion page id is the last 32-hex chunk in the URL PATH (never query/fragment).
    for m in reversed(_ID_IN.findall(parsed.path)):
        nid = normalize_id(m)
        if nid:
            return nid
    return None


def extract_notion_links(text: str) -> list[NotionLink]:
    if not text:
        return []
    seen: dict[str, str | None] = {}
    for anchor, url in _MD.findall(text):
        pid = _id_from_url(url)
        if pid and pid not in seen:
            seen[pid] = anchor.strip() or None
    # bare URLs not already captured via markdown
    md_urls = {u for _, u in _MD.findall(text)}
    for url in _URL.findall(text):
        if url in md_urls:
            continue
        pid = _id_from_url(url)
        if pid and pid not in seen:
            seen[pid] = None
    return [NotionLink(pid, anchor) for pid, anchor in seen.items()]


def parent_ids(page: dict) -> set[str]:
    parent = (page or {}).get("parent") or {}
    out = set()
    for key in ("database_id", "data_source_id"):
        nid = normalize_id(parent.get(key))
        if nid:
            out.add(nid)
    return out


def is_task_page(page: dict, tasks_ids: set[str]) -> bool:
    norm = {normalize_id(t) for t in tasks_ids}
    return bool(parent_ids(page) & {n for n in norm if n})


def read_status(page: dict) -> tuple[str | None, str | None]:
    props = (page or {}).get("properties") or {}
    prop = props.get("Status") or {}
    for kind in ("select", "status"):
        if kind in prop:
            inner = prop.get(kind) or {}
            return (inner.get("name"), kind)
    return (None, None)


def status_patch(value: str, kind: str) -> dict:
    if kind not in ("select", "status"):
        raise ValueError(f"unknown Status kind: {kind!r}")
    return {"Status": {kind: {"name": value}}}


def page_title(page: dict) -> str:
    props = (page or {}).get("properties") or {}
    for prop in props.values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            parts = prop.get("title") or []
            text = "".join(p.get("plain_text", "") for p in parts).strip()
            if text:
                return text
    return "(untitled task)"
