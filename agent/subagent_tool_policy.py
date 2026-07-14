from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

from tools.tool_effects import (
    ResultRetention,
    ToolAuthoritySnapshot,
    ToolEffect,
)


class ToolAuthorizationError(RuntimeError):
    """A resolved tool call does not fit the child's frozen authority."""


class FrozenArgs(Mapping[str, Any]):
    """Recursively frozen JSON-like tool arguments."""

    __slots__ = ("_items", "_mapping")

    def __init__(self, values: Mapping[str, Any]):
        items = tuple((str(key), _freeze(value)) for key, value in values.items())
        self._items = items
        self._mapping = dict(items)

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenArgs(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ToolAuthorizationError(
        f"tool arguments contain unsupported value type {type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, FrozenArgs):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def thaw_authorized_args(call: "AuthorizedToolCall") -> dict[str, Any]:
    return _thaw(call.frozen_args)


def _arguments_digest(args: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            _thaw(_freeze(args)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ToolAuthorizationError("tool arguments are not strict JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AuthorizedToolCall:
    tool_name: str
    frozen_args: Mapping[str, Any]
    effects: frozenset[ToolEffect]
    policy_identity: str
    retention: ResultRetention
    requires_confirmation: bool
    arguments_digest: str


@dataclass(frozen=True)
class ToolNamePolicy:
    """Compatibility name for the child's exact name/effect/identity policy."""

    allowed_names: Optional[frozenset[str]] = None
    denied_names: frozenset[str] = frozenset()
    allowed_effects: Optional[frozenset[ToolEffect]] = None
    authority_snapshot: Optional[ToolAuthoritySnapshot] = None
    profile_name: str = "general-purpose"

    def allows(self, name: str) -> bool:
        if name in self.denied_names:
            return False
        if self.allowed_names is not None and name not in self.allowed_names:
            return False
        return True

    @property
    def is_readonly(self) -> bool:
        if self.allowed_effects is None:
            return False
        return self.allowed_effects.issubset(
            frozenset({ToolEffect.READ_LOCAL, ToolEffect.READ_REMOTE})
        )


def build_child_tool_policy(
    *,
    child: Any,
    parent: Any,
    profile_name: str,
    profile_allowed_names: Optional[frozenset[str]],
    denied_names: frozenset[str] = frozenset(),
) -> ToolNamePolicy:
    """Build an intersection-only exact policy-identity ceiling."""
    allowed_effects = None
    if profile_name in {"Explore", "Plan", "Reviewer"}:
        allowed_effects = frozenset({ToolEffect.READ_LOCAL, ToolEffect.READ_REMOTE})

    parent_snapshot = getattr(parent, "_parent_tool_authority_snapshot", None)
    parent_policy = getattr(parent, "_subagent_tool_policy", None)
    parent_allowed_names: Optional[frozenset[str]] = None
    parent_denied_names = frozenset()
    if isinstance(parent_policy, ToolNamePolicy):
        if isinstance(parent_policy.authority_snapshot, ToolAuthoritySnapshot):
            parent_snapshot = parent_policy.authority_snapshot
        parent_allowed_names = parent_policy.allowed_names
        parent_denied_names = parent_policy.denied_names
    child_snapshot = getattr(child, "_parent_tool_authority_snapshot", None)
    from tools.registry import registry
    from tools.tool_effects import build_authority_snapshot, is_derived_tool_eligible

    if not isinstance(parent_snapshot, ToolAuthoritySnapshot) or not isinstance(
        child_snapshot, ToolAuthoritySnapshot
    ):
        return ToolNamePolicy(
            allowed_names=frozenset(),
            denied_names=denied_names,
            allowed_effects=allowed_effects,
            authority_snapshot=build_authority_snapshot(
                frozenset(), registry_generation=registry._generation
            ),
            profile_name=profile_name,
        )

    parent_ids = parent_snapshot.policy_identities
    child_ids = child_snapshot.policy_identities
    ceiling_ids: set[str] = set()
    allowed_names: set[str] = set()
    for name in registry.get_all_tool_names():
        if name in denied_names:
            continue
        if profile_allowed_names is not None and name not in profile_allowed_names:
            continue
        if parent_allowed_names is not None and name not in parent_allowed_names:
            continue
        if name in parent_denied_names:
            continue
        metadata = registry.resolved_policy_metadata(name)
        if metadata is None:
            continue
        identity, descriptor = metadata
        if identity not in child_ids:
            continue
        if identity in parent_ids:
            ceiling_ids.add(identity)
            allowed_names.add(name)
            continue
        if descriptor.required_parent_any_of and is_derived_tool_eligible(
            descriptor, parent_snapshot
        ):
            sources = descriptor.required_parent_any_of.intersection(parent_ids)
            ceiling_ids.update(sources)
            ceiling_ids.add(identity)
            allowed_names.add(name)

    return ToolNamePolicy(
        allowed_names=frozenset(allowed_names),
        denied_names=denied_names,
        allowed_effects=allowed_effects,
        authority_snapshot=build_authority_snapshot(
            ceiling_ids,
            registry_generation=registry._generation,
        ),
        profile_name=profile_name,
    )


def apply_tool_policy_to_agent(agent, policy: ToolNamePolicy) -> None:
    """Persist and expose only tools permitted by a subagent capability policy.

    Restricted profiles must filter the complete resolved registry surface before
    Tool Search schema compaction.  Filtering an already-compacted surface can
    silently hide authorized deferred tools while leaving their policy metadata
    apparently correct.
    """
    agent._subagent_tool_policy = policy
    definitions = list(getattr(agent, "tools", []) or [])
    if policy.allowed_names is not None:
        resolved = getattr(agent, "_resolved_tool_definitions", None)
        if isinstance(resolved, (list, tuple)):
            definitions = list(resolved)
    definitions = filter_tool_definitions_for_policy(agent, definitions)
    agent.tools = definitions
    agent.valid_tool_names = {
        definition["function"]["name"] for definition in definitions
    }
    # Late refresh is allowed, but Task 6 refresh filtering must preserve the
    # original exact identity ceiling. Name-only refresh disabling is removed.
    agent._skip_mcp_refresh = False


def tool_policy_block_message(agent, tool_name: str) -> Optional[str]:
    policy = getattr(agent, "_subagent_tool_policy", None)
    if policy is None or policy.allows(tool_name):
        return None
    return (
        f"Tool {tool_name!r} is blocked by subagent capability policy. "
        "Do not work around this restriction or spawn another agent."
    )


def _resolved_effects(descriptor, normalized_args: Mapping[str, Any]) -> frozenset[ToolEffect]:
    effects = descriptor.effects
    if descriptor.argument_resolver is not None:
        resolved = descriptor.argument_resolver(dict(normalized_args))
        if not isinstance(resolved, frozenset) or any(
            not isinstance(effect, ToolEffect) for effect in resolved
        ):
            raise ToolAuthorizationError(
                "trusted tool argument resolver returned invalid effects"
            )
        effects = resolved
    if not effects:
        raise ToolAuthorizationError("resolved tool call has no declared effects")
    return effects


def filter_tool_definitions_for_policy(
    agent: Any, definitions: list[dict]
) -> list[dict]:
    """Filter a rebuilt/catalog surface through the child's frozen ceiling."""
    policy = getattr(agent, "_subagent_tool_policy", None)
    if policy is None:
        return definitions
    if policy.authority_snapshot is None:
        return [
            definition
            for definition in definitions
            if tool_policy_block_message(
                agent, (definition.get("function") or {}).get("name", "")
            )
            is None
        ]

    from tools.registry import registry

    allowed: list[dict] = []
    for definition in definitions:
        name = (definition.get("function") or {}).get("name", "")
        if not name or tool_policy_block_message(agent, name) is not None:
            continue
        metadata = registry.resolved_policy_metadata(name)
        if metadata is None:
            continue
        identity, _descriptor = metadata
        if identity in policy.authority_snapshot.policy_identities:
            allowed.append(definition)
    return allowed


def authorize_subagent_call(
    agent: Any,
    tool_name: str,
    args: dict[str, Any],
) -> AuthorizedToolCall:
    """Normalize and authorize one resolved tool call before any side effect."""
    policy = getattr(agent, "_subagent_tool_policy", None)
    if policy is not None and not policy.allows(tool_name):
        raise ToolAuthorizationError(
            f"tool {tool_name!r} is blocked by subagent capability policy"
        )

    from tools.registry import registry

    metadata = registry.resolved_policy_metadata(tool_name)
    if metadata is None:
        raise ToolAuthorizationError(
            f"tool {tool_name!r} has no currently resolved policy identity"
        )
    policy_identity, descriptor = metadata

    if (
        policy is not None
        and policy.authority_snapshot is not None
        and policy_identity not in policy.authority_snapshot.policy_identities
    ):
        raise ToolAuthorizationError(
            f"tool {tool_name!r} identity is outside the original authority ceiling"
        )

    from model_tools import coerce_tool_args

    normalized = coerce_tool_args(tool_name, dict(args))

    effects = _resolved_effects(descriptor, normalized)
    if policy is not None and policy.allowed_effects is not None:
        if ToolEffect.UNKNOWN in effects or not effects.issubset(policy.allowed_effects):
            values = ", ".join(sorted(effect.value for effect in effects))
            raise ToolAuthorizationError(
                f"tool {tool_name!r} effect set [{values}] is blocked by "
                f"{policy.profile_name} capability policy"
            )

    frozen = FrozenArgs(normalized)
    return AuthorizedToolCall(
        tool_name=tool_name,
        frozen_args=frozen,
        effects=effects,
        policy_identity=policy_identity,
        retention=descriptor.retention,
        requires_confirmation=descriptor.requires_confirmation,
        arguments_digest=_arguments_digest(normalized),
    )


def verify_authorized_tool_call(
    call: AuthorizedToolCall,
    tool_name: str,
    args: Mapping[str, Any],
    *,
    current_policy_identity: Optional[str] = None,
) -> dict[str, Any]:
    """Recheck identity and frozen arguments immediately before dispatch.

    Registry dispatch passes the identity of the ToolEntry captured under its
    lock. Other callers may omit it for a current-registry precheck, but that
    precheck alone is not an atomic dispatch guarantee.
    """
    if not isinstance(call, AuthorizedToolCall):
        raise ToolAuthorizationError("missing trusted tool authorization token")
    if call.tool_name != tool_name:
        raise ToolAuthorizationError("authorized tool name changed before dispatch")

    if current_policy_identity is None:
        from tools.registry import registry

        current_policy_identity = registry.resolved_policy_identity(tool_name)
    if current_policy_identity != call.policy_identity:
        raise ToolAuthorizationError("authorized tool identity changed before dispatch")
    if _arguments_digest(args) != call.arguments_digest:
        raise ToolAuthorizationError("authorized tool arguments changed before dispatch")
    return thaw_authorized_args(call)
