"""Tests for ordered fallback chains in web_search_tool.

These focus on the new opt-in `web.search_backends` config key.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _make_provider(name: str, *, success: bool, error: str | None = None, results=None):
    provider = MagicMock()
    provider.name = name
    provider.display_name = name
    provider.supports_search.return_value = True
    if success:
        provider.search.return_value = {
            "success": True,
            "data": {
                "web": results or [
                    {
                        "title": f"{name} result",
                        "url": f"https://{name}.example.com",
                        "description": f"{name} description",
                        "position": 1,
                    }
                ]
            },
        }
    else:
        provider.search.return_value = {"success": False, "error": error or f"{name} failed"}
    return provider


class TestSearchBackendFallbackChain:
    def test_tries_backends_in_order_until_success(self):
        from tools import web_tools

        brave = _make_provider("brave-free", success=False, error="HTTP 429")
        tavily = _make_provider("tavily", success=False, error="HTTP 503")
        exa = _make_provider(
            "exa",
            success=True,
            results=[
                {
                    "title": "exa result",
                    "url": "https://exa.example.com",
                    "description": "exa description",
                    "position": 1,
                }
            ],
        )
        providers = {"brave-free": brave, "tavily": tavily, "exa": exa}

        with patch("tools.web_tools._load_web_config", return_value={"search_backends": ["brave-free", "tavily", "exa"]}), \
            patch("tools.web_tools._ensure_web_plugins_loaded"), \
            patch("agent.web_search_registry.get_provider", side_effect=lambda name: providers.get(name)), \
            patch("tools.interrupt.is_interrupted", return_value=False), \
            patch.object(web_tools._debug, "log_call"), \
            patch.object(web_tools._debug, "save"):
            result = json.loads(web_tools.web_search_tool("fallback chain", limit=3))

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "exa result"
        brave.search.assert_called_once_with("fallback chain", 3)
        tavily.search.assert_called_once_with("fallback chain", 3)
        exa.search.assert_called_once_with("fallback chain", 3)

    def test_stops_after_first_success(self):
        from tools import web_tools

        brave = _make_provider("brave-free", success=True)
        tavily = _make_provider("tavily", success=True)
        exa = _make_provider("exa", success=True)
        providers = {"brave-free": brave, "tavily": tavily, "exa": exa}

        with patch("tools.web_tools._load_web_config", return_value={"search_backends": ["brave-free", "tavily", "exa"]}), \
            patch("tools.web_tools._ensure_web_plugins_loaded"), \
            patch("agent.web_search_registry.get_provider", side_effect=lambda name: providers.get(name)), \
            patch("tools.interrupt.is_interrupted", return_value=False), \
            patch.object(web_tools._debug, "log_call"), \
            patch.object(web_tools._debug, "save"):
            result = json.loads(web_tools.web_search_tool("first hit", limit=2))

        assert result["success"] is True
        brave.search.assert_called_once_with("first hit", 2)
        tavily.search.assert_not_called()
        exa.search.assert_not_called()
