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

import httpx

from . import detection

API_BASE = "https://api.notion.com/v1"
API_VERSION = "2025-09-03"
_RETRYABLE = {429, 500, 502, 503, 504}


class NotionError(Exception):
    pass


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

    async def set_status(self, page_id: str, value: str, kind: str) -> dict:
        body = {"properties": detection.status_patch(value, kind)}
        return await self._request("PATCH", f"/pages/{page_id}", json_body=body)
