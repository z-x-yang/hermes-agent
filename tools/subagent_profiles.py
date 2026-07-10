from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SubagentProfile:
    name: str
    description: str
    model: str
    context_policy: str
    allowed_tool_names: frozenset[str]
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


_CODE_READ_TOOLS = frozenset({
    "read_file", "search_files", "web_search", "web_extract",
})
_CODE_WORKER_TOOLS = _CODE_READ_TOOLS | frozenset({
    "write_file", "patch", "terminal", "process", "todo",
    "skills_list", "skill_view", "vision_analyze",
})

_PROFILES = {
    "Explore": SubagentProfile(
        name="Explore",
        description="Search and understand code/files without making changes.",
        model="inherit",
        context_policy="lean",
        allowed_tool_names=_CODE_READ_TOOLS,
        can_write_files=False,
        can_external_side_effects=False,
        can_delegate=False,
        default_scheduling="foreground",
        foreground_wait_timeout_seconds=900,
        child_run_timeout_seconds=1800,
        system_instructions=(
            "You are the Explore subagent. Search and understand files/code. "
            "Do not review, plan implementation, or modify anything."
        ),
        result_contract=(
            "Return: bottom line; relevant file:symbol:line evidence; searches "
            "performed; what was not found; uncertainty and next lookup."
        ),
    ),
    "Plan": SubagentProfile(
        name="Plan",
        description="Research the codebase and prepare implementation-plan inputs.",
        model="inherit",
        context_policy="project_summary",
        allowed_tool_names=_CODE_READ_TOOLS,
        can_write_files=False,
        can_external_side_effects=False,
        can_delegate=False,
        default_scheduling="foreground",
        foreground_wait_timeout_seconds=1800,
        child_run_timeout_seconds=3600,
        system_instructions=(
            "You are the Plan subagent. Research the codebase for a later plan. "
            "Do not modify files or claim implementation is complete."
        ),
        result_contract=(
            "Return: problem understanding; critical files; proposed shape; "
            "risks; tests; open questions."
        ),
    ),
    "general-purpose": SubagentProfile(
        name="general-purpose",
        description="Handle complex multi-step repository work, including edits.",
        model="inherit",
        context_policy="normal",
        allowed_tool_names=_CODE_WORKER_TOOLS,
        can_write_files=True,
        can_external_side_effects=False,
        can_delegate=False,
        default_scheduling="background",
        foreground_wait_timeout_seconds=1800,
        child_run_timeout_seconds=7200,
        system_instructions=(
            "You are a general-purpose subagent. Complete the scoped task with "
            "repo-local actions and tests. Do not re-delegate the whole task."
        ),
        result_contract=(
            "Return: outcome; files changed; commands/tests and results; evidence; "
            "uncertainty/blockers; side effects; next action."
        ),
    ),
}

SUPPORTED_SUBAGENT_TYPES = tuple(_PROFILES)


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
    return ResolvedProfileConfig(
        model=model,
        provider=provider,
        foreground_wait_timeout_seconds=int(
            agent_cfg.get(
                "foreground_wait_timeout_seconds",
                profile.foreground_wait_timeout_seconds,
            )
        ),
        child_run_timeout_seconds=int(
            agent_cfg.get(
                "child_run_timeout_seconds",
                profile.child_run_timeout_seconds,
            )
        ),
    )
