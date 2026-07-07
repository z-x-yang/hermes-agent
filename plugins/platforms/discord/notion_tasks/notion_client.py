"""Minimal async Notion REST client for the Discord task-button feature.

Auth mirrors the proven ``teams_pipeline`` NotionWriter: the Notion internal
integration token in the ``NOTION_API_KEY`` env var (loaded from ~/.hermes/.env
by the gateway at startup). This is the integration that owns the Tasks data
source — NOT the ``mcp-tokens/notion.json`` OAuth token, which is the MCP
connector's credential and is rejected (401) by the public REST API.

The token is read lazily on first request, so construction does no IO and
never crashes gateway startup; a missing key surfaces as an explicit
NotionError when a button is actually clicked.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import httpx

from . import detection

API_BASE = "https://api.notion.com/v1"
API_VERSION = "2025-09-03"
_RETRYABLE = {429, 500, 502, 503, 504}


class NotionError(Exception):
    pass


def _rt(value: str) -> dict:
    text = str(value or "")[:2000]
    return {"rich_text": [{"type": "text", "text": {"content": text}}] if text else []}


def _date(value: str | None) -> dict:
    return {"date": {"start": value}} if value else {"date": None}


def _url(value: str | None) -> dict:
    return {"url": value or None}


def _number(value: int | float | None) -> dict:
    return {"number": value if value is not None else None}


def _select(value: str) -> dict:
    return {"select": {"name": str(value)}}


def _read_rich(page: dict, name: str) -> str:
    chunks = page.get("properties", {}).get(name, {}).get("rich_text", []) or []
    return "".join(x.get("plain_text", "") for x in chunks)


def _read_date(page: dict, name: str) -> str | None:
    return (page.get("properties", {}).get(name, {}).get("date") or {}).get("start")


def _read_url(page: dict, name: str) -> str | None:
    return page.get("properties", {}).get(name, {}).get("url")


def _read_select(page: dict, name: str) -> str:
    return (page.get("properties", {}).get(name, {}).get("select") or {}).get("name", "")


def _read_number(page: dict, name: str) -> int | float | None:
    return page.get("properties", {}).get(name, {}).get("number")


def _parse_date_time(value: str | None) -> datetime | None:
    if not value or "T" not in str(value):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _same_notion_minute(expected: str | None, got: str | None) -> bool:
    """Compare Notion date-time read-back at the precision Notion stores.

    Notion normalizes date-times to minute precision and may add ``.000`` plus
    an offset even when the PATCH value omitted both. Exact string comparison
    is still required for date-only values, but date-times need semantic minute
    comparison to avoid false verified-write failures.
    """
    if expected == got:
        return True
    if not expected or not got:
        return False
    exp_dt = _parse_date_time(expected)
    got_dt = _parse_date_time(got)
    if exp_dt is None or got_dt is None:
        return False
    exp_min = exp_dt.replace(second=0, microsecond=0)
    got_min = got_dt.replace(second=0, microsecond=0)
    if exp_min.tzinfo is not None and got_min.tzinfo is not None:
        return exp_min.astimezone(timezone.utc) == got_min.astimezone(timezone.utc)
    if exp_min.tzinfo is None and got_min.tzinfo is None:
        return exp_min == got_min
    # A naive timestamp PATCH is read back by Notion with an explicit offset.
    # In that case, compare the visible wall-clock minute rather than treating
    # the missing timezone as a different instant.
    return exp_min.replace(tzinfo=None) == got_min.replace(tzinfo=None)


class NotionClient:
    def __init__(self, *, api_key: str | None = None, transport=None,
                 max_attempts: int = 3, backoff: float = 0.5):
        # api_key None -> read NOTION_API_KEY from env lazily (mirrors teams_pipeline)
        self._api_key = api_key
        self._transport = transport
        self._max_attempts = max_attempts
        self._backoff = backoff

    def _token(self) -> str:
        key = (self._api_key if self._api_key is not None
               else os.getenv("NOTION_API_KEY", "")).strip()
        if not key:
            raise NotionError("NOTION_API_KEY is not configured")
        return key

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token()}",
            "Notion-Version": API_VERSION,
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, path: str, *, json_body=None) -> dict:
        last = None
        for attempt in range(self._max_attempts):
            try:
                async with httpx.AsyncClient(timeout=30.0, transport=self._transport) as client:
                    resp = await client.request(method, f"{API_BASE}{path}",
                                                headers=self._headers, json=json_body)
                if resp.status_code in _RETRYABLE:
                    last = NotionError(f"{resp.status_code}: {resp.text[:200]}")
                elif resp.status_code >= 400:
                    raise NotionError(f"{resp.status_code}: {resp.text[:200]}")
                else:
                    return resp.json()
            except httpx.HTTPError as exc:
                last = NotionError(str(exc))
            if attempt + 1 < self._max_attempts:
                await asyncio.sleep(self._backoff * (2 ** attempt))
        raise last or NotionError("request failed")

    async def get_page(self, page_id: str) -> dict:
        return await self._request("GET", f"/pages/{page_id}")

    async def get_data_source(self, data_source_id: str) -> dict:
        return await self._request("GET", f"/data_sources/{data_source_id}")

    async def query_data_source(self, data_source_id: str, body: dict) -> dict:
        return await self._request("POST", f"/data_sources/{data_source_id}/query", json_body=body or {})

    async def find_task_by_discord_thread_id(self, thread_id: str, tasks_ids: set[str]) -> list[dict]:
        body = {
            "filter": {
                "property": "Discord Thread ID",
                "rich_text": {"equals": str(thread_id)},
            },
            "page_size": 2,
        }
        results: list[dict] = []
        first_error: Exception | None = None
        attempted = 0
        normalized_ids = [nid for nid in (detection.normalize_id(t) for t in (tasks_ids or set())) if nid]
        for task_source_id in sorted(normalized_ids):
            attempted += 1
            try:
                resp = await self.query_data_source(task_source_id, body)
            except NotionError as exc:
                # DEFAULT_TASKS_IDS includes the legacy database_id alongside the
                # 2025 data_source_id. Querying the legacy id via /data_sources
                # may 400/404; skip that id but keep real API failures visible if
                # every configured id fails.
                if first_error is None:
                    first_error = exc
                if str(exc).startswith(("400:", "404:")):
                    continue
                raise
            for page in resp.get("results") or []:
                results.append(page)
                if len(results) >= 2:
                    return results[:2]
        if attempted and not results and first_error is not None:
            raise first_error
        return results

    async def set_status(self, page_id: str, value: str, kind: str) -> dict:
        body = {"properties": detection.status_patch(value, kind)}
        return await self._request("PATCH", f"/pages/{page_id}", json_body=body)

    async def set_properties(self, page_id: str, properties: dict) -> dict:
        await self._request("PATCH", f"/pages/{page_id}", json_body={"properties": properties})
        return await self.get_page(page_id)

    async def set_status_verified(self, page_id: str, value: str, kind: str) -> dict:
        await self.set_status(page_id, value, kind)
        page = await self.get_page(page_id)
        status, _kind = detection.read_status(page)
        if status != value:
            raise NotionError(f"Notion status read-back mismatch: expected {value!r}, got {status!r}")
        return page

    async def set_hold_verified(
        self,
        page_id: str,
        *,
        next_check: str | None,
        reason: str,
        waiting_for: str | None,
    ) -> dict:
        props = detection.status_patch("Hold", "status")
        props["Next Check"] = _date(next_check)
        props["Hold Reason"] = _rt(reason)
        if waiting_for is not None:
            props["Waiting For"] = _rt(waiting_for)
        page = await self.set_properties(page_id, props)
        status, _kind = detection.read_status(page)
        if status != "Hold":
            raise NotionError(f"Notion Hold read-back mismatch: got {status!r}")
        got_next = _read_date(page, "Next Check")
        if not _same_notion_minute(next_check, got_next):
            raise NotionError(
                f"Next Check read-back mismatch: expected {next_check!r}, got {got_next!r}"
            )
        got_reason = _read_rich(page, "Hold Reason")
        if got_reason != str(reason or ""):
            raise NotionError(
                f"Hold Reason read-back mismatch: expected {reason!r}, got {got_reason!r}"
            )
        if waiting_for is not None:
            got_waiting = _read_rich(page, "Waiting For")
            if got_waiting != str(waiting_for or ""):
                raise NotionError(
                    f"Waiting For read-back mismatch: expected {waiting_for!r}, got {got_waiting!r}"
                )
        return page

    async def set_dropped_verified(
        self,
        page_id: str,
        *,
        reason: str,
        source_fingerprint: str | None,
    ) -> dict:
        props = detection.status_patch("Dropped", "status")
        props["Dropped Reason"] = _rt(reason)
        if source_fingerprint:
            props["Source Fingerprint"] = _rt(source_fingerprint)
        page = await self.set_properties(page_id, props)
        status, _kind = detection.read_status(page)
        if status != "Dropped":
            raise NotionError(f"Notion Dropped read-back mismatch: got {status!r}")
        got_reason = _read_rich(page, "Dropped Reason")
        if got_reason != str(reason or ""):
            raise NotionError(
                f"Dropped Reason read-back mismatch: expected {reason!r}, got {got_reason!r}"
            )
        if source_fingerprint:
            got_fp = _read_rich(page, "Source Fingerprint")
            if got_fp != str(source_fingerprint):
                raise NotionError(
                    f"Source Fingerprint read-back mismatch: expected {source_fingerprint!r}, got {got_fp!r}"
                )
        return page

    async def set_thread_binding_verified(
        self,
        page_id: str,
        *,
        thread_id: str,
        thread_url: str,
        title_mode: str,
        title_version: int,
    ) -> dict:
        props = {
            "Discord Thread ID": _rt(thread_id),
            "Discord Thread URL": _url(thread_url),
            "Thread Title Mode": _select(title_mode),
            "Thread Title Version": _number(title_version),
        }
        page = await self.set_properties(page_id, props)
        got_text = _read_rich(page, "Discord Thread ID")
        if got_text != str(thread_id):
            raise NotionError(
                f"Discord Thread ID read-back mismatch: expected {thread_id!r}, got {got_text!r}"
            )
        got_url = _read_url(page, "Discord Thread URL")
        if got_url != str(thread_url):
            raise NotionError(
                f"Discord Thread URL read-back mismatch: expected {thread_url!r}, got {got_url!r}"
            )
        got_mode = _read_select(page, "Thread Title Mode")
        if got_mode != str(title_mode):
            raise NotionError(
                f"Thread Title Mode read-back mismatch: expected {title_mode!r}, got {got_mode!r}"
            )
        got_version = _read_number(page, "Thread Title Version")
        if got_version != title_version:
            raise NotionError(
                f"Thread Title Version read-back mismatch: expected {title_version!r}, got {got_version!r}"
            )
        return page
