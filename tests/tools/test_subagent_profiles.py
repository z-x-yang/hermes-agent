from pathlib import Path

import pytest

from tools.subagent_profiles import (
    DEFAULT_SUBAGENT_TYPE,
    SUPPORTED_SUBAGENT_TYPES,
    get_subagent_profile,
    resolve_subagent_type,
    resolve_profile_config,
)


def test_only_claude_aligned_builtin_types_are_exposed():
    assert SUPPORTED_SUBAGENT_TYPES == (
        "Explore",
        "Plan",
        "general-purpose",
    )


def test_omitted_type_resolves_to_general_purpose():
    assert DEFAULT_SUBAGENT_TYPE == "general-purpose"
    assert resolve_subagent_type(None) == "general-purpose"
    assert resolve_subagent_type("") == "general-purpose"


@pytest.mark.parametrize("name", SUPPORTED_SUBAGENT_TYPES)
def test_builtin_profile_round_trip(name):
    profile = get_subagent_profile(name)
    assert profile.name == name
    assert profile.model == "inherit"


def test_profiles_use_claude_like_type_specific_final_guidance():
    explore = get_subagent_profile("Explore").system_instructions
    plan = get_subagent_profile("Plan").system_instructions
    gp = get_subagent_profile("general-purpose").system_instructions

    for prompt in (explore, plan, gp):
        assert "recommended_next_step" not in prompt
        assert "files_changed" not in prompt
        assert "side_effects" not in prompt
    assert "clearly and concisely" in explore
    assert "absolute file paths" in explore
    assert "### Critical Files for Implementation" in plan
    assert "3-5" in plan
    assert "exact return requirements in the task prompt" in gp


def test_profile_metadata_has_no_redundant_capability_booleans_or_context_capsule():
    profile = get_subagent_profile("general-purpose")
    for removed in (
        "result_contract",
        "context_policy",
        "can_write_files",
        "can_external_side_effects",
        "can_delegate",
        "default_scheduling",
    ):
        assert not hasattr(profile, removed)


def test_read_only_profiles_remain_hard_no_external_side_effect():
    for name in ("Explore", "Plan"):
        profile = get_subagent_profile(name)
        assert "terminal" not in profile.allowed_tool_names
        assert "process" not in profile.allowed_tool_names
        assert "web_search" not in profile.allowed_tool_names
        assert "web_extract" not in profile.allowed_tool_names
        assert "vision_analyze" not in profile.allowed_tool_names
        assert profile.allowed_tool_names is not None
        assert {
            "web_search_readonly",
            "web_extract_readonly",
            "skills_list_readonly",
            "skill_view_readonly",
        }.issubset(profile.allowed_tool_names)
        assert {
            "mcp_notion_ai_notion_ai_ask",
            "mcp_apple_mail_search_messages",
            "mcp_apple_mail_get_message",
            "mcp_apple_mail_get_thread",
            "mcp_apple_mail_fetch_attachment",
        }.issubset(profile.allowed_tool_names)
        assert "mcp_apple_mail_send_email" not in profile.allowed_tool_names
        assert "mcp_apple_mail_delete_message" not in profile.allowed_tool_names
        assert "mcp_apple_mail_mark_as_read" not in profile.allowed_tool_names
        assert "mode=readonly" in profile.system_instructions
        assert "Never send, reply, forward, move, delete, flag, or mark mail" in (
            profile.system_instructions
        )


def test_readonly_aliases_activate_skill_prompt_capability_detection():
    from agent.system_prompt import (
        _adapt_skills_prompt_for_readonly_aliases,
        _has_skill_read_tools,
    )

    names = {"skills_list_readonly", "skill_view_readonly", "web_search_readonly"}
    assert _has_skill_read_tools(names)
    assert not _has_skill_read_tools({"read_file", "search_files"})

    prompt = _adapt_skills_prompt_for_readonly_aliases(
        "load skill_view(name); use skills_list and web_search; "
        "If a skill has issues, fix it with skill_manage(action='patch').\n",
        names,
    )
    assert "skill_view_readonly" in prompt
    assert "skills_list_readonly" in prompt
    assert "web_search_readonly" in prompt
    assert "skill_manage" not in prompt


def _force_web_tools_available(monkeypatch, registry):
    for name in (
        "web_search",
        "web_extract",
        "web_search_readonly",
        "web_extract_readonly",
    ):
        entry = registry.get_entry(name)
        assert entry is not None
        monkeypatch.setattr(entry, "check_fn", lambda: True)


def test_readonly_aliases_are_no_spill_reads_derived_from_raw_tools(monkeypatch):
    import tools.skills_tool  # noqa: F401 — register aliases
    import tools.web_tools  # noqa: F401 — register aliases
    from tools.registry import registry
    from tools.tool_effects import ResultRetention, ToolEffect

    _force_web_tools_available(monkeypatch, registry)

    expected = {
        "web_search_readonly": ("web_search", ToolEffect.READ_REMOTE),
        "web_extract_readonly": ("web_extract", ToolEffect.READ_REMOTE),
        "skills_list_readonly": ("skills_list", ToolEffect.READ_LOCAL),
        "skill_view_readonly": ("skill_view", ToolEffect.READ_LOCAL),
    }
    for alias, (raw, effect) in expected.items():
        metadata = registry.resolved_policy_metadata(alias)
        assert metadata is not None
        _, descriptor = metadata
        assert descriptor.effects == frozenset({effect})
        assert descriptor.retention is ResultRetention.NO_SPILL
        assert descriptor.required_parent_any_of == frozenset(
            {registry.resolved_policy_identity(raw)}
        )


def test_explore_materializes_readonly_aliases_from_raw_parent_authority(monkeypatch):
    from types import SimpleNamespace

    import tools.skills_tool  # noqa: F401 — register aliases
    import tools.web_tools  # noqa: F401 — register aliases
    from agent.subagent_tool_policy import build_child_tool_policy
    from tools.registry import registry
    from tools.tool_effects import build_authority_snapshot

    _force_web_tools_available(monkeypatch, registry)

    raw_names = {"web_search", "web_extract", "skills_list", "skill_view"}
    alias_names = {
        "web_search_readonly",
        "web_extract_readonly",
        "skills_list_readonly",
        "skill_view_readonly",
    }
    raw_ids = []
    alias_ids = []
    for name in raw_names:
        identity = registry.resolved_policy_identity(name)
        assert isinstance(identity, str)
        raw_ids.append(identity)
    for name in alias_names:
        identity = registry.resolved_policy_identity(name)
        assert isinstance(identity, str)
        alias_ids.append(identity)

    parent = SimpleNamespace(
        _parent_tool_authority_snapshot=build_authority_snapshot(
            raw_ids,
            registry_generation=registry._generation,
        )
    )
    child = SimpleNamespace(
        _parent_tool_authority_snapshot=build_authority_snapshot(
            alias_ids,
            registry_generation=registry._generation,
        )
    )
    profile_allowed_names = get_subagent_profile("Explore").allowed_tool_names
    assert profile_allowed_names is not None

    policy = build_child_tool_policy(
        child=child,
        parent=parent,
        profile_name="Explore",
        profile_allowed_names=profile_allowed_names,
    )

    assert policy.allowed_names is not None
    assert alias_names.issubset(policy.allowed_names)
    assert policy.allowed_names.isdisjoint(raw_names)


def test_general_purpose_truthfully_describes_raw_shell_external_effect_capability():
    profile = get_subagent_profile("general-purpose")
    assert profile.allowed_tool_names is None
    assert "exact parent tool surface" in profile.system_instructions
    assert "Named external-side-effect tools are not available" not in (
        profile.system_instructions
    )
    assert "not a no-side-effect sandbox" in profile.system_instructions
    assert "normal tool and terminal approvals" in profile.system_instructions


def test_delegation_docs_match_simplified_contract():
    root = Path(__file__).resolve().parents[2]
    docs = [
        root / "website/docs/user-guide/features/delegation.md",
        root
        / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/features"
        / "delegation.md",
        root / "website/docs/guides/delegation-patterns.md",
        root
        / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides"
        / "delegation-patterns.md",
        root / "website/docs/user-guide/configuration.md",
        root
        / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide"
        / "configuration.md",
        root
        / "website/docs/user-guide/skills/bundled/autonomous-ai-agents"
        / "autonomous-ai-agents-hermes-agent.md",
        root
        / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/skills/bundled/autonomous-ai-agents"
        / "autonomous-ai-agents-hermes-agent.md",
    ]
    required_claims = (
        "description",
        "prompt",
        "run_in_background",
        "Explore",
        "Plan",
        "general-purpose",
    )
    stale_claims = (
        "retain_session",
        'scheduling="auto"',
        "scheduling='auto'",
        'role="orchestrator"',
        "recommended_next_step",
    )
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert all(claim in text for claim in required_claims), path
        assert not any(claim in text for claim in stale_claims), path

    feature_docs = docs[:2]
    semantic_claims = (
        "one batch handle",
        "one consolidated completion",
        "one-shot",
        "automatically retained",
        "project context",
        "complete governance",
        "runtime-derived",
    )
    zh_semantic_claims = (
        "一个 batch handle",
        "一次合并完成通知",
        "一次性",
        "自动保留",
        "项目上下文",
        "完整 governance",
        "运行时派生",
    )
    en_text = feature_docs[0].read_text(encoding="utf-8")
    zh_text = feature_docs[1].read_text(encoding="utf-8")
    assert all(claim in en_text for claim in semantic_claims)
    assert all(claim in zh_text for claim in zh_semantic_claims)

    pattern_docs = docs[2:4]
    for path in pattern_docs:
        text = path.read_text(encoding="utf-8")
        assert "exact current-parent tool authority" in text or (
            "current parent 精确工具权限" in text
        )


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError, match="Unknown subagent_type"):
        get_subagent_profile("review-readonly")


def test_per_agent_config_overrides_global_without_exposing_to_model():
    cfg = {
        "model": "global-model",
        "provider": "openrouter",
        "foreground_wait_timeout_seconds": 1200,
        "child_run_timeout_seconds": 2400,
        "max_foreground_wait_timeout_seconds": 7200,
        "agents": {
            "Explore": {
                "model": "cheap-model",
                "foreground_wait_timeout_seconds": 900,
                "child_run_timeout_seconds": 1800,
            }
        },
    }
    resolved = resolve_profile_config("Explore", cfg)
    assert resolved.model == "cheap-model"
    assert resolved.provider == "openrouter"
    assert resolved.foreground_wait_timeout_seconds == 900
    assert resolved.child_run_timeout_seconds == 1800


def test_global_timeouts_override_profile_defaults():
    resolved = resolve_profile_config(
        "Explore",
        {
            "foreground_wait_timeout_seconds": 1234,
            "child_run_timeout_seconds": 2345,
        },
    )
    assert resolved.foreground_wait_timeout_seconds == 1234
    assert resolved.child_run_timeout_seconds == 2345


def test_foreground_wait_timeout_is_clamped_by_positive_maximum():
    resolved = resolve_profile_config(
        "Plan",
        {
            "foreground_wait_timeout_seconds": 9000,
            "max_foreground_wait_timeout_seconds": 4000,
            "agents": {"Plan": {"foreground_wait_timeout_seconds": 8000}},
        },
    )
    assert resolved.foreground_wait_timeout_seconds == 4000
