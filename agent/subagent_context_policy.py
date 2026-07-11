"""Trusted, observable context-policy capsules for delegated agents."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, Sequence, cast

from tools.subagent_profiles import SubagentProfile


class ContextPolicyError(RuntimeError):
    """Raised when trusted context-policy inputs cannot be safely materialized."""


class _ImmutableStringMap(dict[str, str]):
    """A copied string map that rejects ordinary mutation operations."""

    def _reject_mutation(self, *args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise TypeError("Context policy maps are immutable")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __ior__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation


@dataclass(frozen=True)
class ContextPolicyCapsule:
    policy: Literal["lean", "project_summary", "normal"]
    workspace_path: str
    project_routes: Sequence[dict[str, str]]
    workspace_metadata: dict[str, str]
    must_query_project_memory: bool


def _trusted_project_routes(parent_agent: Any) -> tuple[dict[str, str], ...]:
    """Copy only the runtime-populated trusted route seam, never conversation data."""
    missing = object()
    declared_routes = inspect.getattr_static(
        parent_agent, "_trusted_project_routes", missing
    )
    if declared_routes is missing:
        return ()
    raw_routes = getattr(parent_agent, "_trusted_project_routes", None)
    if raw_routes is None:
        return ()
    if not isinstance(raw_routes, (list, tuple)):
        raise ContextPolicyError("Trusted project routes have an invalid container type")

    routes: list[dict[str, str]] = []
    for candidate in raw_routes:
        if not isinstance(candidate, dict):
            raise ContextPolicyError("Trusted project routes contain an invalid entry type")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in candidate.items()
        ):
            raise ContextPolicyError("Trusted project routes contain invalid field types")
        routes.append(dict(candidate))
    return tuple(routes)


def _normal_workspace_metadata(workspace_path: str) -> dict[str, str]:
    """Resolve bounded workspace/repo metadata through the coding-context seam."""
    if not workspace_path:
        return {}
    try:
        resolved_workspace = str(Path(workspace_path).expanduser().resolve())
    except (OSError, RuntimeError):
        raise ContextPolicyError("Workspace path metadata could not be resolved") from None

    metadata = {"workspace_path": resolved_workspace}
    try:
        from agent.coding_context import project_facts_for

        facts = project_facts_for(resolved_workspace)
    except Exception:
        raise ContextPolicyError("Workspace project metadata could not be loaded") from None
    if isinstance(facts, dict):
        repo_root = facts.get("root")
        if isinstance(repo_root, str) and repo_root:
            metadata["repo_root"] = repo_root
    return metadata


def build_context_policy_capsule(
    *,
    profile: SubagentProfile,
    goal: str,
    context: str | None,
    parent_agent: Any,
    workspace_path: str,
) -> ContextPolicyCapsule:
    """Build trusted routing metadata without consuming task or transcript content.

    ``goal`` and ``context`` remain explicit parameters to keep the trust boundary
    visible at the call site, but are deliberately never inspected or serialized.
    """
    del goal, context
    policy = cast(
        Literal["lean", "project_summary", "normal"], profile.context_policy
    )
    if policy not in {"lean", "project_summary", "normal"}:
        raise ValueError(f"Unsupported subagent context_policy: {policy}")

    if policy == "lean":
        routes: tuple[dict[str, str], ...] = ()
    else:
        routes = _trusted_project_routes(parent_agent)

    metadata = (
        _normal_workspace_metadata(workspace_path) if policy == "normal" else {}
    )
    return ContextPolicyCapsule(
        policy=policy,
        workspace_path=workspace_path,
        project_routes=tuple(_ImmutableStringMap(route) for route in routes),
        workspace_metadata=_ImmutableStringMap(metadata),
        must_query_project_memory=not bool(routes),
    )
