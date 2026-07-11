from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional


@dataclass(frozen=True)
class SubagentProfile:
    name: str
    description: str
    model: str
    context_policy: str
    allowed_tool_names: frozenset[str] | None
    can_write_files: bool
    can_external_side_effects: bool
    can_delegate: bool
    default_scheduling: str
    foreground_wait_timeout_seconds: int
    child_run_timeout_seconds: int
    system_instructions: str
    result_contract: str


@dataclass(frozen=True)
class ResolvedProfileConfig:
    model: Optional[str]
    provider: Optional[str]
    foreground_wait_timeout_seconds: int
    child_run_timeout_seconds: int


NOTION_PROMPT_READ_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {"mcp_notion_ai_notion_ai_ask"}
)

APPLE_MAIL_READ_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "mcp_apple_mail_list_accounts",
        "mcp_apple_mail_list_mailboxes",
        "mcp_apple_mail_search_messages",
        "mcp_apple_mail_list_messages",
        "mcp_apple_mail_get_message",
        "mcp_apple_mail_get_thread",
        "mcp_apple_mail_get_unread_count",
        "mcp_apple_mail_get_mail_stats",
        "mcp_apple_mail_get_sync_status",
        "mcp_apple_mail_health_check",
        "mcp_apple_mail_list_attachments",
        "mcp_apple_mail_fetch_attachment",
        "mcp_apple_mail_search_contacts",
    }
)

_READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "search_files",
        "web_search_readonly",
        "web_extract_readonly",
        "skills_list_readonly",
        "skill_view_readonly",
    }
) | NOTION_PROMPT_READ_TOOL_NAMES | APPLE_MAIL_READ_TOOL_NAMES

_DATA_SOURCE_READ_INSTRUCTIONS = (
    "For Notion, call notion_ai_ask only with mode=readonly and explicitly tell "
    "it not to create, edit, move, or delete workspace content. For Apple Mail, "
    "use only the provided search/list/get/fetch tools. Never send, reply, forward, "
    "move, delete, flag, or mark mail."
)

_COMPLETE_RESULT_FIELDS = (
    "Return all fields: outcome; evidence; actions; files_changed; tests_run; "
    "verification; blockers; open_questions; confidence; limitations; "
    "side_effects; recommended_next_step. Use empty lists or 'none' when a "
    "field does not apply; never omit fields."
)


def _result_contract(profile_guidance: str) -> str:
    return f"{_COMPLETE_RESULT_FIELDS} {profile_guidance}"


_PROFILES = {
    "Explore": SubagentProfile(
        name="Explore",
        description=(
            "Search and understand code/files or permitted data sources without "
            "changes; use for focused lookup and exploration."
        ),
        model="inherit",
        context_policy="lean",
        allowed_tool_names=_READ_ONLY_TOOLS,
        can_write_files=False,
        can_external_side_effects=False,
        can_delegate=False,
        default_scheduling="foreground",
        foreground_wait_timeout_seconds=900,
        child_run_timeout_seconds=1800,
        system_instructions=(
            "You are the Explore subagent. Search and understand files/code. "
            "Do not review, plan implementation, or modify anything. "
            + _DATA_SOURCE_READ_INSTRUCTIONS
        ),
        result_contract=_result_contract(
            "For Explore, evidence cites source/file/symbol/line handles; actions "
            "enumerate searches/lookups; files_changed, tests_run, and side_effects "
            "are normally empty."
        ),
    ),
    "Plan": SubagentProfile(
        name="Plan",
        description=(
            "Research the codebase/data sources and prepare implementation-plan "
            "inputs; use for planning research without edits."
        ),
        model="inherit",
        context_policy="project_summary",
        allowed_tool_names=_READ_ONLY_TOOLS,
        can_write_files=False,
        can_external_side_effects=False,
        can_delegate=False,
        default_scheduling="foreground",
        foreground_wait_timeout_seconds=1800,
        child_run_timeout_seconds=3600,
        system_instructions=(
            "You are the Plan subagent. Research the codebase for a later plan. "
            "Do not modify files or claim implementation is complete. "
            + _DATA_SOURCE_READ_INSTRUCTIONS
        ),
        result_contract=_result_contract(
            "For Plan, outcome is the implementation-plan shape; evidence names "
            "critical files; actions summarize planning research; verification "
            "assesses feasibility without claiming execution."
        ),
    ),
    "general-purpose": SubagentProfile(
        name="general-purpose",
        description=(
            "Handle complex multi-step work, including edits, tests, "
            "terminal/process, or permitted external actions."
        ),
        model="inherit",
        context_policy="normal",
        allowed_tool_names=None,
        can_write_files=True,
        can_external_side_effects=True,
        can_delegate=True,
        default_scheduling="background",
        foreground_wait_timeout_seconds=1800,
        child_run_timeout_seconds=7200,
        system_instructions=(
            "You are a general-purpose subagent. Complete the scoped task with "
            "repo-local actions and tests. You may use the exact parent tool surface "
            "that survives the profile ceiling, including named external tools when "
            "the user/task scope and normal tool contracts permit them. Raw "
            "terminal/process access can also reach external systems: this is not a "
            "no-side-effect sandbox. Follow normal tool and terminal approvals. Do "
            "not re-delegate the whole task."
        ),
        result_contract=_result_contract(
            "For general-purpose, actions list executed actions; files_changed and "
            "tests_run name concrete artifacts/commands; verification reports real "
            "outputs; side_effects include externally verifiable handles/status."
        ),
    ),
}

SUPPORTED_SUBAGENT_TYPES = tuple(_PROFILES)
DEFAULT_SUBAGENT_TYPE: Final[str] = "general-purpose"


def resolve_subagent_type(value: str | None) -> str:
    normalized = str(value or "").strip() or DEFAULT_SUBAGENT_TYPE
    if normalized not in SUPPORTED_SUBAGENT_TYPES:
        raise ValueError(f"Unsupported subagent_type: {normalized}")
    return normalized


def get_subagent_profile(name: str) -> SubagentProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        allowed = ", ".join(SUPPORTED_SUBAGENT_TYPES)
        raise ValueError(
            f"Unknown subagent_type {name!r}; expected one of: {allowed}"
        ) from exc


def resolve_profile_config(
    name: str,
    delegation_config: Mapping[str, Any],
) -> ResolvedProfileConfig:
    profile = get_subagent_profile(name)
    agent_cfg = dict((delegation_config.get("agents") or {}).get(name) or {})
    model = agent_cfg.get("model", delegation_config.get("model"))
    provider = agent_cfg.get("provider", delegation_config.get("provider"))
    wait_timeout = int(
        agent_cfg.get(
            "foreground_wait_timeout_seconds",
            delegation_config.get(
                "foreground_wait_timeout_seconds",
                profile.foreground_wait_timeout_seconds,
            ),
        )
    )
    max_wait_timeout = int(
        delegation_config.get("max_foreground_wait_timeout_seconds", 7200)
    )
    if max_wait_timeout <= 0:
        max_wait_timeout = 7200
    return ResolvedProfileConfig(
        model=model,
        provider=provider,
        foreground_wait_timeout_seconds=min(wait_timeout, max_wait_timeout),
        child_run_timeout_seconds=int(
            agent_cfg.get(
                "child_run_timeout_seconds",
                delegation_config.get(
                    "child_run_timeout_seconds",
                    profile.child_run_timeout_seconds,
                ),
            )
        ),
    )
