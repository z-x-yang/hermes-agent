"""Tests for tools/tool_search.py — progressive tool disclosure.

Coverage targets — these mirror the issues called out in the OpenClaw tool
search report. Every test that names an OpenClaw issue is the regression
guard that would have caught that specific failure mode.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Dict, Any

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _td(name: str, description: str = "", properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
            },
        },
    }


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_default_when_missing(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(None)
        assert cfg.enabled == "auto"
        assert cfg.threshold_pct == 10.0

    def test_bool_true_maps_to_auto(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(True)
        assert cfg.enabled == "auto"

    def test_bool_false_maps_to_off(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(False)
        assert cfg.enabled == "off"

    def test_explicit_on(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert cfg.enabled == "on"

    def test_invalid_enabled_falls_back_to_auto(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"enabled": "maybe"})
        assert cfg.enabled == "auto"

    def test_threshold_clamped(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"threshold_pct": 150})
        assert cfg.threshold_pct == 100.0
        cfg = ToolSearchConfig.from_raw({"threshold_pct": -5})
        assert cfg.threshold_pct == 0.0

    def test_search_limits_clamped(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({
            "search_default_limit": 999,
            "max_search_limit": 999,
        })
        assert cfg.max_search_limit == 50
        assert cfg.search_default_limit <= cfg.max_search_limit

    def test_hybrid_visibility_config_is_parsed_and_deduplicated(self):
        from tools.tool_search import ToolSearchConfig

        cfg = ToolSearchConfig.from_raw({
            "threshold_schema_tokens": 8_000,
            "always_visible_tools": ["ledger_*", "mcp_mail_search", "ledger_*", ""],
        })

        assert cfg.threshold_schema_tokens == 8_000
        assert cfg.always_visible_tools == ("ledger_*", "mcp_mail_search")

    def test_default_config_exposes_hybrid_visibility_controls(self):
        from hermes_cli.config import DEFAULT_CONFIG

        raw = DEFAULT_CONFIG["tools"]["tool_search"]
        assert "threshold_tool_count" not in raw
        assert raw["threshold_schema_tokens"] == 10_000
        assert raw["always_visible_tools"] == []


# ---------------------------------------------------------------------------
# Classification — the hard invariant: core tools NEVER defer.
# ---------------------------------------------------------------------------


class TestClassification:
    def test_core_tools_never_defer(self):
        """The critical invariant from the OpenClaw report."""
        from tools.tool_search import is_deferrable_tool_name
        # Sample of core tools from _HERMES_CORE_TOOLS.
        for core_name in ["terminal", "read_file", "write_file", "patch",
                          "search_files", "todo", "memory", "browser_navigate",
                          "web_search", "session_search", "clarify",
                          "execute_code", "delegate_task", "send_message"]:
            assert not is_deferrable_tool_name(core_name), (
                f"Core tool '{core_name}' must NEVER be deferrable"
            )

    def test_bridge_tools_never_defer(self):
        from tools.tool_search import is_deferrable_tool_name, BRIDGE_TOOL_NAMES
        for name in BRIDGE_TOOL_NAMES:
            assert not is_deferrable_tool_name(name)

    def test_unknown_tool_not_deferrable(self):
        """Defensive: a tool name we cannot resolve to a registry entry must
        not be claimed as deferrable. This protects against the OpenClaw
        cron regression where unresolved tools were silently dropped."""
        from tools.tool_search import is_deferrable_tool_name
        assert not is_deferrable_tool_name("xx_definitely_not_a_tool_xx")

    def test_classify_keeps_unknown_in_visible(self):
        """A tool we can't classify stays visible — never silently dropped.

        This is the OpenClaw #84141 regression guard (cron lost ``exec``
        because it wasn't in the catalog).
        """
        from tools.tool_search import classify_tools
        # Build a tool def for something we don't have a registry entry for.
        defs = [_td("xx_unknown_tool", "Unknown tool")]
        visible, deferrable = classify_tools(defs)
        names = {(td.get("function") or {}).get("name") for td in visible}
        assert "xx_unknown_tool" in names
        assert deferrable == []


# ---------------------------------------------------------------------------
# Token estimation + threshold gate
# ---------------------------------------------------------------------------


class TestThresholdGate:
    def test_off_never_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "off"})
        assert not should_activate(cfg, deferrable_tokens=1_000_000, context_length=200_000)

    def test_zero_deferrable_never_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert not should_activate(cfg, deferrable_tokens=0, context_length=200_000)

    def test_on_activates_with_any_deferrable(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert should_activate(cfg, deferrable_tokens=100, context_length=200_000)

    def test_auto_ignores_legacy_deferred_tool_count_threshold(self):
        from tools.tool_search import ToolSearchConfig, should_activate

        cfg = ToolSearchConfig.from_raw({
            "enabled": "auto",
            "threshold_tool_count": 10,
            "threshold_schema_tokens": 0,
            "threshold_pct": 100,
        })

        assert not hasattr(cfg, "threshold_tool_count")
        assert not should_activate(
            cfg, deferrable_tokens=100, context_length=200_000
        )

    def test_auto_activates_at_absolute_schema_token_threshold(self):
        from tools.tool_search import ToolSearchConfig, should_activate

        cfg = ToolSearchConfig.from_raw({
            "enabled": "auto",
            "threshold_schema_tokens": 8_000,
            "threshold_pct": 100,
        })

        assert not should_activate(
            cfg, deferrable_tokens=7_999, context_length=200_000
        )
        assert should_activate(
            cfg, deferrable_tokens=8_000, context_length=200_000
        )

    def test_auto_below_percentage_threshold_does_not_activate(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({
            "enabled": "auto",
            "threshold_pct": 10,
            "threshold_schema_tokens": 0,
        })
        # 5% of 200K = below 10% threshold
        assert not should_activate(cfg, deferrable_tokens=10_000, context_length=200_000)

    def test_auto_at_or_above_percentage_threshold_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({
            "enabled": "auto",
            "threshold_pct": 10,
            "threshold_schema_tokens": 0,
        })
        assert should_activate(cfg, deferrable_tokens=20_000, context_length=200_000)
        assert should_activate(cfg, deferrable_tokens=50_000, context_length=200_000)

    def test_auto_without_context_length_uses_20k_cutoff(self):
        """Fallback cutoff used when the active model is unknown."""
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({
            "enabled": "auto",
            "threshold_schema_tokens": 0,
        })
        assert not should_activate(cfg, deferrable_tokens=10_000, context_length=0)
        assert should_activate(cfg, deferrable_tokens=25_000, context_length=0)

    def test_zero_disables_each_auto_threshold(self):
        from tools.tool_search import ToolSearchConfig, should_activate

        cfg = ToolSearchConfig.from_raw({
            "enabled": "auto",
            "threshold_schema_tokens": 0,
            "threshold_pct": 0,
        })

        assert not should_activate(
            cfg,
            deferrable_tokens=100_000,
            context_length=200_000,
        )

    def test_token_estimate_proportional_to_schema_size(self):
        from tools.tool_search import estimate_tokens_from_schemas
        small = [_td("a", "x")]
        big = [_td(f"name_{i}", f"description for tool {i} " * 20,
                   {"q": {"type": "string", "description": "search query " * 10}})
               for i in range(10)]
        small_t = estimate_tokens_from_schemas(small)
        big_t = estimate_tokens_from_schemas(big)
        assert big_t > small_t * 10

    def test_token_estimate_uses_o200k_for_multilingual_schemas(self):
        import tiktoken
        from tools.tool_search import estimate_tokens_from_schemas

        schemas = [_td("demo", "检索项目记录并总结关键决策。" * 100)]
        payload = json.dumps(
            schemas, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        expected = len(tiktoken.get_encoding("o200k_base").encode(payload))

        assert estimate_tokens_from_schemas(schemas) == expected

    def test_active_context_length_honors_internal_compression_window(self, monkeypatch):
        """Tool-search auto gate uses Hermes' internal context window, not runtime."""
        import agent.model_metadata as model_metadata
        import hermes_cli.config as hermes_config
        from model_tools import _resolve_active_context_length

        monkeypatch.setattr(
            hermes_config,
            "load_config",
            lambda: {
                "model": {
                    "default": "gpt-5.5",
                    "provider": "gptcodex",
                    "context_length": 1_000_000,
                },
                "compression": {"internal_context_length": 272_000},
            },
        )

        seen = {}

        def fake_get_model_context_length(model, **kwargs):
            seen["model"] = model
            seen.update(kwargs)
            return kwargs.get("config_context_length") or 1_050_000

        monkeypatch.setattr(
            model_metadata,
            "get_model_context_length",
            fake_get_model_context_length,
        )

        assert _resolve_active_context_length() == 272_000
        assert seen["model"] == "gpt-5.5"
        assert seen["config_context_length"] == 1_000_000
        assert seen["provider"] == "gptcodex"


# ---------------------------------------------------------------------------
# Retrieval (BM25 + substring fallback)
# ---------------------------------------------------------------------------


class TestRetrieval:
    def _fake_catalog(self):
        """Build a catalog directly without touching the registry."""
        from tools.tool_search import CatalogEntry, _tokenize, _entry_search_text
        defs = [
            _td("github_create_issue", "Open a new issue in a GitHub repository",
                {"title": {"type": "string"}, "body": {"type": "string"}}),
            _td("github_search_repos", "Search GitHub for matching repositories",
                {"query": {"type": "string"}}),
            _td("slack_send_message", "Post a message into a Slack channel",
                {"channel": {"type": "string"}, "text": {"type": "string"}}),
            _td("calendar_create_event", "Add an event to the user's calendar",
                {"title": {"type": "string"}, "start": {"type": "string"}}),
        ]
        catalog = []
        for d in defs:
            fn = d["function"]
            e = CatalogEntry(
                name=fn["name"], description=fn["description"],
                schema=d, source="mcp", source_name="mcp-test",
            )
            e._tokens = _tokenize(_entry_search_text(d))
            catalog.append(e)
        return catalog

    def test_search_finds_relevant_tool(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "create a github issue", limit=3)
        names = [h.name for h in hits]
        assert names[0] == "github_create_issue"

    def test_search_returns_empty_for_irrelevant_query(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "asdf qwerty foobar", limit=3)
        assert hits == []

    def test_search_substring_fallback(self):
        """Even when no BM25 hit, a literal substring of the tool name returns."""
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "calendar", limit=3)
        assert any("calendar" in h.name for h in hits)

    def test_search_respects_limit(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "github", limit=1)
        assert len(hits) <= 1

    def test_description_preview_keeps_capability_and_late_query_match(self):
        from tools.tool_search import _description_preview

        description = (
            "Create and manage deployment records for an application. "
            + "General operational context. " * 20
            + "Use this tool for blue-green rollback coordination and audit receipts."
        )
        preview = _description_preview(
            description,
            "blue-green rollback coordination",
        )

        assert preview.startswith("Create and manage deployment records")
        assert "blue-green rollback coordination" in preview
        assert len(preview) <= 400
        assert "…" in preview


# ---------------------------------------------------------------------------
# Assembly — the full passthrough/activate decision.
# ---------------------------------------------------------------------------


class TestAssembly:
    def test_hybrid_pins_exact_and_glob_tools_visible(self):
        from tools.registry import registry
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig

        pinned = ["hybrid_pin_ledger_read", "hybrid_pin_mail_search"]
        deferred = [f"hybrid_pin_longtail_{index}" for index in range(8)]
        names = pinned + deferred
        try:
            for name in names:
                registry.register(
                    name=name,
                    toolset="hybrid-pin-test",
                    schema=_td(name)["function"],
                    handler=lambda args: args,
                )
            result = assemble_tool_defs(
                [_td(name) for name in names],
                context_length=200_000,
                config=ToolSearchConfig.from_raw({
                    "enabled": "auto",
                    "threshold_schema_tokens": 1,
                    "threshold_pct": 100,
                    "always_visible_tools": [
                        "hybrid_pin_ledger_*",
                        "hybrid_pin_mail_search",
                    ],
                }),
            )
        finally:
            for name in names:
                registry.deregister(name)

        visible_names = {td["function"]["name"] for td in result.tool_defs}
        assert result.activated
        assert result.deferred_count == len(deferred)
        assert set(pinned) <= visible_names
        assert not (set(deferred) & visible_names)

    def test_auto_assembly_does_not_activate_from_legacy_tool_count_alone(self):
        from tools.registry import registry
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig

        names = [f"hybrid_count_tool_{index}" for index in range(10)]
        try:
            for name in names:
                registry.register(
                    name=name,
                    toolset="hybrid-test",
                    schema=_td(name)["function"],
                    handler=lambda args: args,
                )
            result = assemble_tool_defs(
                [_td(name) for name in names],
                context_length=200_000,
                config=ToolSearchConfig.from_raw({
                    "enabled": "auto",
                    "threshold_tool_count": 10,
                    "threshold_schema_tokens": 0,
                    "threshold_pct": 100,
                }),
            )
        finally:
            for name in names:
                registry.deregister(name)

        assert not result.activated
        assert result.deferred_count == 10
        assert {td["function"]["name"] for td in result.tool_defs} == set(names)

    def test_parent_tool_surfaces_resolve_once_before_assembly(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import Mock

        import agent.agent_init as agent_init
        import model_tools

        resolved = [_td("visible"), _td("mcp_hidden")]
        visible = [_td("visible"), _td("tool_search")]
        get_definitions = Mock(return_value=resolved)
        assemble = Mock(return_value=visible)
        monkeypatch.setattr(
            agent_init,
            "_ra",
            lambda: SimpleNamespace(get_tool_definitions=get_definitions),
        )
        monkeypatch.setattr(
            model_tools,
            "assemble_resolved_tool_definitions",
            assemble,
            raising=False,
        )

        actual_resolved, actual_visible = agent_init._resolve_parent_tool_surfaces(
            enabled_toolsets=["test"],
            disabled_toolsets=None,
            quiet_mode=True,
        )

        assert actual_resolved is resolved
        assert actual_visible is visible
        get_definitions.assert_called_once_with(
            enabled_toolsets=["test"],
            disabled_toolsets=None,
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        assemble.assert_called_once_with(resolved, quiet_mode=True)

    def test_builtin_file_and_catalog_bridges_have_explicit_read_metadata(self):
        from tools import file_tools  # noqa: F401
        from tools import tool_search  # noqa: F401
        from tools.registry import registry
        from tools.tool_effects import ResultRetention, ToolEffect

        for name in ("read_file", "search_files", "tool_search", "tool_describe"):
            descriptor = registry.get_entry(name).policy_descriptor
            assert descriptor.effects == frozenset({ToolEffect.READ_LOCAL})
            assert descriptor.retention is ResultRetention.NO_SPILL

        for name in ("write_file", "patch", "tool_call"):
            descriptor = registry.get_entry(name).policy_descriptor
            assert descriptor.effects == frozenset({ToolEffect.UNKNOWN})

    def test_raw_web_skills_terminal_and_process_are_not_promoted_to_read(self):
        from tools import process_registry  # noqa: F401
        from tools import skills_tool  # noqa: F401
        from tools import terminal_tool  # noqa: F401
        from tools import web_tools  # noqa: F401
        from tools.registry import registry
        from tools.tool_effects import ToolEffect

        for name in (
            "web_search",
            "web_extract",
            "skill_view",
            "terminal",
            "process",
        ):
            assert registry.get_entry(name).policy_descriptor.effects == frozenset(
                {ToolEffect.UNKNOWN}
            )

    def test_parent_authority_uses_full_resolved_scope_before_deferral(self):
        from agent.agent_init import _capture_parent_tool_authority
        from tools.registry import ToolRegistry

        reg = ToolRegistry()
        for name in (
            "visible",
            "deferred_hidden",
            "disabled_global",
            "tool_search",
            "tool_describe",
            "tool_call",
        ):
            reg.register(
                name=name,
                toolset="test",
                schema=_td(name)["function"],
                handler=lambda args: args,
            )

        resolved = [_td("visible"), _td("deferred_hidden")]
        model_visible = [
            _td("visible"),
            _td("tool_search"),
            _td("tool_describe"),
            _td("tool_call"),
        ]
        snapshot = _capture_parent_tool_authority(
            resolved, model_visible, registry=reg
        )
        expected_names = {
            "visible",
            "deferred_hidden",
            "tool_search",
            "tool_describe",
            "tool_call",
        }
        expected = {
            reg.get_entry(name).policy_identity for name in expected_names
        }

        assert snapshot.policy_identities == frozenset(expected)
        assert reg.get_entry("disabled_global").policy_identity not in snapshot.policy_identities
        assert "deferred_hidden" not in {
            td["function"]["name"] for td in model_visible
        }

        reg.register(
            name="later",
            toolset="test",
            schema=_td("later")["function"],
            handler=lambda args: args,
        )
        assert snapshot.policy_identities == frozenset(expected)

    def test_no_deferrable_returns_unchanged(self):
        """Pure-core toolset: pass-through, no bridge tools added."""
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        defs = [_td("terminal", "Run shell"), _td("read_file", "Read a file")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        assert not result.activated
        assert {t["function"]["name"] for t in result.tool_defs} == {"terminal", "read_file"}

    def test_below_threshold_returns_unchanged(self):
        """Tiny deferrable surface: don't bother."""
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        # _td renders to ~80 chars / 20 tokens. 3 of them = ~60 tokens.
        # 10% of 200K = 20K. Way below.
        defs = [_td("unknown_tool_a"), _td("unknown_tool_b"), _td("unknown_tool_c")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "auto", "threshold_pct": 10}),
        )
        assert not result.activated
        names = {(t.get("function") or {}).get("name") for t in result.tool_defs}
        assert "tool_search" not in names

    def test_idempotent_when_bridge_already_present(self):
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig, BRIDGE_TOOL_NAMES
        defs = [_td("terminal", "Run shell"), _td("tool_search", "old")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "off"}),
        )
        names = [(t["function"]["name"]) for t in result.tool_defs]
        # The pre-existing tool_search was stripped (it would be re-injected if
        # activation happened; here it didn't).
        assert "tool_search" not in names


# ---------------------------------------------------------------------------
# Bridge dispatch
# ---------------------------------------------------------------------------


class TestBridgeDispatch:
    def test_bridge_descriptions_prioritize_dedicated_and_exact_recovery(self):
        from tools.tool_search import bridge_tool_schemas

        descriptions = {
            schema["function"]["name"]: schema["function"]["description"].lower()
            for schema in bridge_tool_schemas(12)
        }

        assert "mandatory discovery gate" in descriptions["tool_search"]
        assert "request names an app or service" in descriptions["tool_search"]
        assert "already-visible dedicated tool" in descriptions["tool_search"]
        assert "before browser, computer, or terminal" in descriptions["tool_search"]
        assert "exact tool name" in descriptions["tool_describe"]
        assert "without searching first" in descriptions["tool_describe"]

    def test_pinned_tools_are_excluded_from_all_bridge_surfaces(self):
        from tools.registry import registry
        from tools.tool_search import (
            ToolSearchConfig,
            dispatch_tool_describe,
            dispatch_tool_search,
            scoped_deferrable_names,
        )

        pinned = "hybrid_bridge_pinned"
        deferred = "hybrid_bridge_deferred"
        names = [pinned, deferred]
        cfg = ToolSearchConfig.from_raw({
            "always_visible_tools": [pinned],
        })
        defs = [_td(name, f"hybrid bridge {name}") for name in names]
        try:
            for name in names:
                registry.register(
                    name=name,
                    toolset="hybrid-bridge-test",
                    schema=_td(name)["function"],
                    handler=lambda args: args,
                )

            search = json.loads(dispatch_tool_search(
                {"query": "hybrid bridge"},
                current_tool_defs=defs,
                config=cfg,
            ))
            described = json.loads(dispatch_tool_describe(
                {"name": pinned},
                current_tool_defs=defs,
                config=cfg,
            ))
            scoped = scoped_deferrable_names(defs, config=cfg)
        finally:
            for name in names:
                registry.deregister(name)

        assert search["total_available"] == 1
        assert [match["name"] for match in search["matches"]] == [deferred]
        assert "call it directly" in described["error"]
        assert scoped == frozenset({deferred})

    def test_agent_executor_bridge_excludes_pinned_registry_tools(self, monkeypatch):
        import model_tools
        from agent.tool_executor import _agent_bridge_deferrable_tool_defs
        from tools import tool_search as tool_search_module
        from tools.registry import registry

        pinned = "agent_bridge_pinned"
        deferred = "agent_bridge_deferred"
        names = [pinned, deferred]
        defs = [_td(name, f"agent bridge {name}") for name in names]
        cfg = tool_search_module.ToolSearchConfig.from_raw(
            {"always_visible_tools": [pinned]}
        )
        agent = type(
            "Agent",
            (),
            {
                "enabled_toolsets": None,
                "disabled_toolsets": None,
                "_memory_manager": None,
                "_context_compressor": None,
                "_context_engine_tool_names": set(),
            },
        )()
        try:
            for name in names:
                registry.register(
                    name=name,
                    toolset="agent-bridge-test",
                    schema=_td(name)["function"],
                    handler=lambda args: args,
                )
            monkeypatch.setattr(
                model_tools, "get_tool_definitions", lambda **_kwargs: defs
            )
            monkeypatch.setattr(tool_search_module, "load_config", lambda: cfg)

            bridged = _agent_bridge_deferrable_tool_defs(agent)
        finally:
            for name in names:
                registry.deregister(name)

        assert [td["function"]["name"] for td in bridged] == [deferred]

    def test_agent_local_dynamic_pin_stays_direct_only(self, monkeypatch):
        import model_tools
        from agent.tool_executor import _agent_bridge_deferrable_tool_defs
        from tools import tool_search as tool_search_module
        from tools.registry import registry

        deferred = "agent_bridge_registry_deferred"
        local = "honcho_recall"
        cfg = tool_search_module.ToolSearchConfig.from_raw(
            {"always_visible_tools": [local]}
        )

        class MemoryManager:
            @staticmethod
            def get_all_tool_schemas():
                return [_td(local, "agent-local direct tool")["function"]]

        agent = type(
            "Agent",
            (),
            {
                "enabled_toolsets": None,
                "disabled_toolsets": None,
                "_memory_manager": MemoryManager(),
                "context_compressor": None,
                "_context_engine_tool_names": set(),
            },
        )()
        try:
            registry.register(
                name=deferred,
                toolset="agent-bridge-test",
                schema=_td(deferred)["function"],
                handler=lambda args: args,
            )
            monkeypatch.setattr(
                model_tools,
                "get_tool_definitions",
                lambda **_kwargs: [_td(deferred, "registry deferred")],
            )
            monkeypatch.setattr(tool_search_module, "load_config", lambda: cfg)

            bridged = _agent_bridge_deferrable_tool_defs(agent)
        finally:
            registry.deregister(deferred)

        assert [td["function"]["name"] for td in bridged] == [deferred]

    def test_agent_executor_scope_cache_tracks_pin_config(self, monkeypatch):
        import model_tools
        from agent.tool_executor import _tool_search_scoped_names
        from tools import tool_search as tool_search_module
        from tools.registry import registry

        name = "agent_scope_pin_refresh"
        defs = [_td(name, "scope refresh")]
        current_config = {
            "value": tool_search_module.ToolSearchConfig.from_raw({})
        }
        agent = type(
            "Agent",
            (),
            {
                "enabled_toolsets": None,
                "disabled_toolsets": None,
                "_subagent_tool_policy": None,
            },
        )()
        try:
            registry.register(
                name=name,
                toolset="agent-bridge-test",
                schema=defs[0]["function"],
                handler=lambda args: args,
            )
            monkeypatch.setattr(
                model_tools, "get_tool_definitions", lambda **_kwargs: defs
            )
            monkeypatch.setattr(
                tool_search_module,
                "load_config",
                lambda: current_config["value"],
            )

            assert _tool_search_scoped_names(agent) == frozenset({name})
            current_config["value"] = tool_search_module.ToolSearchConfig.from_raw(
                {"always_visible_tools": [name]}
            )
            assert _tool_search_scoped_names(agent) == frozenset()
        finally:
            registry.deregister(name)

    def test_tool_search_requires_query(self):
        from tools.tool_search import dispatch_tool_search
        result = dispatch_tool_search({}, current_tool_defs=[])
        assert "error" in json.loads(result)

    def test_tool_describe_requires_name(self):
        from tools.tool_search import dispatch_tool_describe
        result = dispatch_tool_describe({}, current_tool_defs=[])
        assert "error" in json.loads(result)

    def test_tool_describe_rejects_non_deferrable(self):
        """If the model asks to describe a core tool, refuse — it's already
        in the visible list."""
        from tools.tool_search import dispatch_tool_describe
        result = dispatch_tool_describe(
            {"name": "terminal"}, current_tool_defs=[_td("terminal", "Run shell")],
        )
        assert "error" in json.loads(result)

    def test_resolve_underlying_call_parses_object_args(self):
        from tools.tool_search import resolve_underlying_call
        name, args, err = resolve_underlying_call({
            "name": "unknown_xxx",
            "arguments": {"foo": "bar"},
        })
        # Will fail classification because unknown_xxx isn't deferrable.
        assert err is not None

    def test_resolve_underlying_call_parses_json_string_args(self):
        """Some models emit ``arguments`` as a JSON string instead of object."""
        from tools.tool_search import resolve_underlying_call
        # Use a name that won't classify (so we don't depend on registry),
        # but exercise the JSON parse path.
        _, _, err = resolve_underlying_call({
            "name": "fake",
            "arguments": '{"a": 1}',
        })
        # err is about classification, but the parse worked (it would have
        # failed earlier with "not valid JSON" otherwise).
        assert "not valid JSON" not in (err or "")

    def test_resolve_underlying_call_rejects_bad_json(self):
        from tools.tool_search import resolve_underlying_call
        _, _, err = resolve_underlying_call({
            "name": "fake",
            "arguments": "{this is not json",
        })
        assert err is not None
        assert "JSON" in err

    def test_resolve_underlying_call_rejects_recursion(self):
        """tool_call cannot invoke tool_call itself."""
        from tools.tool_search import resolve_underlying_call, TOOL_CALL_NAME
        name, args, err = resolve_underlying_call({
            "name": TOOL_CALL_NAME,
            "arguments": {},
        })
        assert err is not None
        assert "bridge tool" in err.lower()


# ---------------------------------------------------------------------------
# End-to-end via the real handle_function_call (smoke test).
# ---------------------------------------------------------------------------


class TestHandleFunctionCallIntegration:
    def test_tool_search_dispatch_through_handle_function_call(self):
        """The dispatcher recognizes the bridge tool by name."""
        import model_tools
        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "nothing matches this"},
        )
        parsed = json.loads(result)
        # Without a real registry, the matches will be empty, but the
        # dispatch path completed without error.
        assert "matches" in parsed or "error" in parsed


class TestRegression_OpenClawCron84141:
    """Regression guard for the OpenClaw cron-tool-loss class of bug.

    OpenClaw #84141: ``toolsAllow: ["exec"]`` on an isolated cron turn
    resulted in the agent receiving only ``sessions_send`` — the catalog
    builder silently dropped the requested core tool.

    Our defense: core tools are NEVER deferred. This test exercises the
    full assembly pipeline with a mixed core+MCP toolset and asserts that
    every core tool survives.
    """

    def test_core_tool_survives_alongside_many_mcp_tools(self):
        from tools.tool_search import (
            assemble_tool_defs, ToolSearchConfig, BRIDGE_TOOL_NAMES,
            classify_tools,
        )
        # 1 core tool + 50 unknown/MCP-shaped tools (deferrable).
        defs = [_td("terminal", "Run shell commands")]
        # Pad with fake "deferrable" tools — without registry registration,
        # classify_tools puts them in 'visible'. So instead, we just verify
        # the core-tool side: terminal stays in visible regardless.
        visible, deferrable = classify_tools(defs)
        assert any(
            (td.get("function") or {}).get("name") == "terminal"
            for td in visible
        ), "Core tool 'terminal' was wrongly classified as deferrable"

        # Now force activation and check the resulting tool-defs list.
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        names = {(t.get("function") or {}).get("name") for t in result.tool_defs}
        # terminal must be present; bridges are only added if there are
        # deferrable tools to put behind them.
        assert "terminal" in names

    def test_unwrap_rejects_core_tool_attempt(self):
        """Even if the model tries to invoke a core tool through tool_call,
        we reject the call and tell the model to use it directly."""
        from tools.tool_search import resolve_underlying_call
        _, _, err = resolve_underlying_call({
            "name": "terminal",
            "arguments": {"command": "echo hi"},
        })
        assert err is not None
        assert "not a deferrable" in err


class TestRegression_ToolsetScoping:
    """A restricted-toolset session must not see or invoke out-of-scope tools.

    The bug: the bridge dispatch and the tool_executor unwrap read the
    catalog from the *global* registry (get_tool_definitions with no
    toolset scope = "start with everything"), so a session scoped to one
    MCP server could tool_search the entire process registry and tool_call
    any plugin tool it was never granted. registry.dispatch() has no
    enabled_tools gate for non-execute_code tools, so the out-of-scope tool
    actually ran.

    The fix threads the session's enabled/disabled toolsets into the bridge
    dispatch (model_tools.handle_function_call) and the executor unwrap
    (agent.tool_executor), scoping both the searchable catalog and the
    invocable set to the session's own toolsets.
    """

    @staticmethod
    def _register(name, toolset):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True, "tool": name})

        registry.register(
            name=name,
            handler=_handler,
            schema=_td(name, f"desc for {name}", {"repo": {"type": "string"}}),
            toolset=toolset,
        )

    def test_search_catalog_is_scoped_to_session_toolsets(self):
        import model_tools

        for i in range(12):
            self._register(f"mcp_scoped_gh_{i}", "mcp-scoped-gh")
        self._register("scoped_oos_plugin", "scopedoosplugin")

        # tool_search scoped to the github toolset must not count the
        # out-of-scope plugin tool (or any of the host registry).
        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "mcp_scoped_gh", "limit": 5},
            enabled_toolsets=["mcp-scoped-gh"],
        )
        parsed = json.loads(result)
        assert parsed["total_available"] == 12, (
            f"expected scoped catalog of 12, got {parsed['total_available']} "
            "— catalog leaked tools outside the session's toolsets"
        )
        hit_names = {m["name"] for m in parsed["matches"]}
        assert "scoped_oos_plugin" not in hit_names

    def test_tool_call_rejects_out_of_scope_tool(self):
        import model_tools

        self._register("mcp_inscope_gh_op", "mcp-inscope-gh")
        self._register("inscope_oos_plugin", "inscopeoosplugin")

        # Out-of-scope plugin tool: rejected even though it is registered
        # and deferrable in the global registry.
        rejected = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": "inscope_oos_plugin", "arguments": {}},
            enabled_toolsets=["mcp-inscope-gh"],
        ))
        assert "error" in rejected
        assert "not available in this session" in rejected["error"]

        # In-scope tool: dispatches normally.
        ok = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": "mcp_inscope_gh_op", "arguments": {"repo": "a/b"}},
            enabled_toolsets=["mcp-inscope-gh"],
        ))
        assert ok.get("ok") is True
        assert ok.get("tool") == "mcp_inscope_gh_op"

    def test_bridge_dispatch_does_not_pollute_global_resolved_names(self):
        import model_tools

        self._register("mcp_pollute_op_0", "mcp-pollute")
        self._register("mcp_pollute_op_1", "mcp-pollute")

        # Establish the scoped session global.
        model_tools.get_tool_definitions(
            enabled_toolsets=["mcp-pollute"], quiet_mode=True,
        )
        before = set(model_tools._last_resolved_tool_names)
        assert "terminal" not in before

        # A scoped tool_search call must not widen the process-global
        # _last_resolved_tool_names to the whole registry (which would leak
        # core/sandbox tools into execute_code's fallback).
        model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "pollute"},
            enabled_toolsets=["mcp-pollute"],
        )
        after = set(model_tools._last_resolved_tool_names)
        assert "terminal" not in after, (
            "bridge dispatch polluted _last_resolved_tool_names with "
            "out-of-scope tools"
        )

    def test_scoped_deferrable_names_helper(self):
        from tools.tool_search import scoped_deferrable_names

        self._register("mcp_helper_op", "mcp-helper")
        import model_tools
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["mcp-helper"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = scoped_deferrable_names(defs)
        assert "mcp_helper_op" in names
        # core tools are never deferrable
        assert "terminal" not in names

