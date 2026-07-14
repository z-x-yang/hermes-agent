from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, FrozenSet, Mapping, Optional


@dataclass(frozen=True)
class SubagentProfile:
    name: str
    description: str
    system_instructions: str
    allowed_tool_names: Optional[FrozenSet[str]]
    default_wait_timeout_seconds: int
    default_run_timeout_seconds: int
    default_run_in_background: bool
    retain_on_success: bool
    context_policy: str
    allow_delegation: bool


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

REVIEWER_REQUIRED_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {"read_file", "search_files", "terminal"}
)
REVIEWER_OPTIONAL_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {"web_search_readonly", "web_extract_readonly"}
)
REVIEWER_TOOL_NAMES: Final[frozenset[str]] = (
    REVIEWER_REQUIRED_TOOL_NAMES | REVIEWER_OPTIONAL_TOOL_NAMES
)

_DATA_SOURCE_READ_INSTRUCTIONS = (
    "For public web research, use web_search_readonly and web_extract_readonly. "
    "For Notion, call mcp_notion_ai_notion_ai_ask only with mode=readonly and explicitly tell "
    "it not to create, edit, move, or delete workspace content. For Apple Mail, "
    "use only the provided search/list/get/fetch tools. Never send, reply, forward, "
    "move, delete, flag, or mark mail."
)

EXPLORE_FINAL = (
    "Report findings clearly and concisely in one final message. Include absolute "
    "file paths and relevant symbols or line ranges for claims the parent must verify."
)

PLAN_FINAL = (
    "Return an actionable implementation plan without making changes. End with "
    "`### Critical Files for Implementation` and list 3-5 files that are central "
    "to the plan, each with a one-line reason."
)

GENERAL_FINAL = (
    "Return one concise final message that follows the exact return requirements in "
    "the task prompt. The parent will verify claimed changes and side effects."
)

REVIEWER_FINAL = (
    "You are an independent code reviewer working from a fresh context. Read the "
    "ordinary self-contained task prompt, inspect the requested Git diff/files and "
    "relevant surrounding code, and use repository rules from the bound workspace. "
    "Report only newly introduced, evidence-backed Critical or Important candidate "
    "blockers; never edit files or decide merge readiness. You may use terminal for "
    "inspection and relevant checks, but it is not a no-side-effect sandbox: do not "
    "write files, install packages, change Git state, publish, or access Notion, Mail, "
    "session history, personal memory, or other private sources. The controller will "
    "verify every candidate."
)

_PROFILES = {
    "Explore": SubagentProfile(
        name="Explore",
        description=(
            "Search and understand code/files or permitted data sources without "
            "changes; use for focused lookup and exploration."
        ),
        system_instructions=(
            "You are the Explore subagent. Search and understand files/code. "
            "Do not review, plan implementation, or modify anything. "
            + _DATA_SOURCE_READ_INSTRUCTIONS
            + " "
            + EXPLORE_FINAL
        ),
        allowed_tool_names=_READ_ONLY_TOOLS,
        default_wait_timeout_seconds=900,
        default_run_timeout_seconds=1800,
        default_run_in_background=True,
        retain_on_success=False,
        context_policy="lean",
        allow_delegation=False,
    ),
    "Plan": SubagentProfile(
        name="Plan",
        description=(
            "Research the codebase/data sources and prepare implementation-plan "
            "inputs; use for planning research without edits."
        ),
        system_instructions=(
            "You are the Plan subagent. Research the codebase for a later plan. "
            "Do not modify files or claim implementation is complete. "
            + _DATA_SOURCE_READ_INSTRUCTIONS
            + " "
            + PLAN_FINAL
        ),
        allowed_tool_names=_READ_ONLY_TOOLS,
        default_wait_timeout_seconds=1800,
        default_run_timeout_seconds=3600,
        default_run_in_background=True,
        retain_on_success=False,
        context_policy="project_summary",
        allow_delegation=False,
    ),
    "Reviewer": SubagentProfile(
        name="Reviewer",
        description=(
            "Fresh-context independent code reviewer. Use with an ordinary "
            "self-contained prompt to inspect a scoped change and return "
            "evidence-backed candidate blockers without editing."
        ),
        system_instructions=REVIEWER_FINAL,
        allowed_tool_names=REVIEWER_TOOL_NAMES,
        default_wait_timeout_seconds=1800,
        default_run_timeout_seconds=3600,
        default_run_in_background=False,
        retain_on_success=False,
        context_policy="reviewer_project",
        allow_delegation=False,
    ),
    "general-purpose": SubagentProfile(
        name="general-purpose",
        description=(
            "Handle complex multi-step work, including edits, tests, "
            "terminal/process, or permitted external actions."
        ),
        system_instructions=(
            "You are a general-purpose subagent. Complete the scoped task with "
            "appropriate actions and verification. You may use the exact parent tool surface "
            "that survives the profile ceiling, including named external tools when "
            "the user/task scope and normal tool contracts permit them. Raw "
            "terminal/process access can also reach external systems: this is not a "
            "no-side-effect sandbox. Follow normal tool and terminal approvals. Do "
            "not re-delegate the whole task. "
            + GENERAL_FINAL
        ),
        allowed_tool_names=None,
        default_wait_timeout_seconds=1800,
        default_run_timeout_seconds=7200,
        default_run_in_background=True,
        retain_on_success=True,
        context_policy="project_context",
        allow_delegation=True,
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


def _positive_timeout(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"delegation.{key} must be a positive integer; got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"delegation.{key} must be a positive integer; got {value!r}"
        ) from exc
    if isinstance(value, float) and value != parsed:
        raise ValueError(f"delegation.{key} must be a positive integer; got {value!r}")
    if parsed <= 0:
        raise ValueError(f"delegation.{key} must be a positive integer; got {value!r}")
    return parsed


def resolve_profile_config(
    name: str,
    delegation_config: Mapping[str, Any],
) -> ResolvedProfileConfig:
    profile = get_subagent_profile(name)
    agent_cfg = dict((delegation_config.get("agents") or {}).get(name) or {})
    model = agent_cfg.get("model", delegation_config.get("model"))
    provider = agent_cfg.get("provider", delegation_config.get("provider"))
    wait_is_per_agent = "foreground_wait_timeout_seconds" in agent_cfg
    wait_timeout = _positive_timeout(
        agent_cfg.get(
            "foreground_wait_timeout_seconds",
            delegation_config.get(
                "foreground_wait_timeout_seconds",
                profile.default_wait_timeout_seconds,
            ),
        ),
        key=(
            f"agents.{name}.foreground_wait_timeout_seconds"
            if wait_is_per_agent
            else "foreground_wait_timeout_seconds"
        ),
    )
    max_wait_timeout = _positive_timeout(
        delegation_config.get("max_foreground_wait_timeout_seconds", 7200),
        key="max_foreground_wait_timeout_seconds",
    )
    run_is_per_agent = "child_run_timeout_seconds" in agent_cfg
    run_timeout = _positive_timeout(
        agent_cfg.get(
            "child_run_timeout_seconds",
            delegation_config.get(
                "child_run_timeout_seconds",
                profile.default_run_timeout_seconds,
            ),
        ),
        key=(
            f"agents.{name}.child_run_timeout_seconds"
            if run_is_per_agent
            else "child_run_timeout_seconds"
        ),
    )
    return ResolvedProfileConfig(
        model=model,
        provider=provider,
        foreground_wait_timeout_seconds=min(wait_timeout, max_wait_timeout),
        child_run_timeout_seconds=run_timeout,
    )
