"""Trusted metadata for resolved tool effects and parent authority.

This module is intentionally descriptive only.  Runtime enforcement belongs to
``tool_executor``; registration and assembly use these immutable records to
bind later authorization to an exact tool source and registry generation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit


POLICY_VERSION = 1


class ToolEffect(StrEnum):
    READ_LOCAL = "READ_LOCAL"
    READ_REMOTE = "READ_REMOTE"
    WRITE_LOCAL = "WRITE_LOCAL"
    WRITE_REMOTE = "WRITE_REMOTE"
    EXECUTE = "EXECUTE"
    DESTRUCTIVE = "DESTRUCTIVE"
    UNKNOWN = "UNKNOWN"


class ResultRetention(StrEnum):
    DEFAULT = "DEFAULT"
    NO_SPILL = "NO_SPILL"
    HANDLE_ONLY = "HANDLE_ONLY"


@dataclass(frozen=True)
class ToolPolicyDescriptor:
    effects: frozenset[ToolEffect]
    requires_confirmation: bool
    retention: ResultRetention
    source_identity: str
    schema_digest: str
    policy_version: int
    argument_resolver: Callable[[Mapping[str, Any]], frozenset[ToolEffect]] | None
    required_parent_any_of: frozenset[str]


@dataclass(frozen=True)
class ToolAuthoritySnapshot:
    policy_identities: frozenset[str]
    registry_generation: int
    fingerprint: str


def _canonical_json(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except TypeError as exc:
        raise ValueError("value must be JSON serializable") from exc
    except ValueError as exc:
        raise ValueError("JSON numbers must be finite") from exc
    return encoded.encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _normalize_schema_for_identity(value: Any) -> Any:
    """Remove provider-sanitized, semantically inert schema syntax.

    Tool schemas with no required properties may express that as either an
    absent ``required`` key or ``required: []``. Provider sanitization removes
    the empty form before the model sees it, so policy identity must treat the
    two representations as equivalent while preserving every non-empty
    requirement and all other schema content.
    """
    if isinstance(value, Mapping):
        return {
            key: _normalize_schema_for_identity(item)
            for key, item in value.items()
            if not (key == "required" and item == [])
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_schema_for_identity(item) for item in value]
    return value


def schema_digest(schema: Mapping[str, Any]) -> str:
    """Return the SHA-256 of normalized compact, sorted-key UTF-8 JSON.

    ``default=str`` is deliberately forbidden: malformed schemas must fail
    registration instead of silently acquiring a misleading identity.
    """
    if not isinstance(schema, Mapping):
        raise TypeError("tool schema must be a mapping")
    return hashlib.sha256(
        _canonical_json(_normalize_schema_for_identity(schema))
    ).hexdigest()


def _callable_parts(fn: Callable[..., Any], *, role: str) -> tuple[str, str]:
    if not callable(fn):
        raise TypeError(f"{role} must be callable")
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
        callable_type = type(fn)
        module = getattr(callable_type, "__module__", None)
        qualname = getattr(callable_type, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
        raise ValueError(f"{role} must expose module and qualname metadata")
    return module, qualname


def callable_identity(
    fn: Callable[..., Any],
    *,
    registered_name: str,
    role: str = "handler",
    trusted_resolver: bool = False,
) -> str:
    """Build a safe deterministic callable identity without source or repr."""
    module, qualname = _callable_parts(fn, role=role)
    if trusted_resolver:
        if not (inspect.isfunction(fn) or inspect.isbuiltin(fn)):
            raise ValueError(f"{role} must be a locally registered function")
        if "<lambda>" in qualname or "<locals>" in qualname:
            raise ValueError(f"{role} must be a named module-level trusted resolver")
    payload = {
        "module": module,
        "qualname": qualname,
        "registered_name": registered_name,
        "role": role,
    }
    return f"callable:{module}:{qualname}:{_sha256(payload)}"


def argument_resolver_identity(
    resolver: Callable[[Mapping[str, Any]], frozenset[ToolEffect]] | None,
) -> str | None:
    if resolver is None:
        return None
    return callable_identity(
        resolver,
        registered_name="argument_resolver",
        role="argument resolver",
        trusted_resolver=True,
    )


def _normalize_effects(effects: Iterable[ToolEffect]) -> frozenset[ToolEffect]:
    try:
        normalized = frozenset(effects)
    except TypeError as exc:
        raise TypeError("effects must be an iterable of ToolEffect") from exc
    if not normalized or any(not isinstance(effect, ToolEffect) for effect in normalized):
        raise ValueError("effects must be a non-empty set of ToolEffect values")
    return normalized


def validate_policy_descriptor(
    descriptor: ToolPolicyDescriptor,
    *,
    expected_schema_digest: str | None = None,
) -> ToolPolicyDescriptor:
    if not isinstance(descriptor, ToolPolicyDescriptor):
        raise TypeError("descriptor must be a ToolPolicyDescriptor")
    if not isinstance(descriptor.effects, frozenset):
        raise TypeError("tool policy effects must be a frozenset")
    _normalize_effects(descriptor.effects)
    if type(descriptor.requires_confirmation) is not bool:
        raise TypeError("requires_confirmation must be bool")
    if not isinstance(descriptor.retention, ResultRetention):
        raise TypeError("retention must be ResultRetention")
    if not isinstance(descriptor.source_identity, str) or not descriptor.source_identity:
        raise ValueError("source_identity must be a non-empty string")
    if (
        not isinstance(descriptor.schema_digest, str)
        or len(descriptor.schema_digest) != 64
        or any(char not in "0123456789abcdef" for char in descriptor.schema_digest)
    ):
        raise ValueError("schema digest must be a lowercase SHA-256 hex digest")
    if expected_schema_digest is not None and descriptor.schema_digest != expected_schema_digest:
        raise ValueError("descriptor schema digest does not match registered schema digest")
    if type(descriptor.policy_version) is not int or descriptor.policy_version < 1:
        raise ValueError("policy_version must be a positive integer")
    argument_resolver_identity(descriptor.argument_resolver)
    if not isinstance(descriptor.required_parent_any_of, frozenset) or any(
        not isinstance(identity, str) or not identity.startswith("policy:")
        for identity in descriptor.required_parent_any_of
    ):
        raise ValueError("required_parent_any_of must contain exact policy identities")
    return descriptor


def builtin_policy_descriptor(
    *,
    name: str,
    schema: Mapping[str, Any],
    handler: Callable[..., Any],
    effects: Iterable[ToolEffect],
    requires_confirmation: bool = False,
    retention: ResultRetention = ResultRetention.DEFAULT,
    policy_version: int = POLICY_VERSION,
    argument_resolver: Callable[[Mapping[str, Any]], frozenset[ToolEffect]] | None = None,
    required_parent_any_of: Iterable[str] = (),
) -> ToolPolicyDescriptor:
    handler_identity = callable_identity(handler, registered_name=name)
    source_identity = f"builtin:{handler_identity}:{name}"
    descriptor = ToolPolicyDescriptor(
        effects=_normalize_effects(effects),
        requires_confirmation=requires_confirmation,
        retention=retention,
        source_identity=source_identity,
        schema_digest=schema_digest(schema),
        policy_version=policy_version,
        argument_resolver=argument_resolver,
        required_parent_any_of=frozenset(required_parent_any_of),
    )
    return validate_policy_descriptor(descriptor)


def _mcp_transport_locator(config: Mapping[str, Any]) -> tuple[str, str]:
    """Return transport kind and a credential-safe locator digest."""
    if not isinstance(config, Mapping):
        raise TypeError("MCP config must be a mapping")
    if config.get("url"):
        raw_url = str(config["url"])
        parsed = urlsplit(raw_url)
        hostname = parsed.hostname or "unknown"
        if parsed.port is not None:
            hostname = f"{hostname}:{parsed.port}"
        safe_url = urlunsplit((parsed.scheme.lower(), hostname.lower(), parsed.path, "", ""))
        transport = str(config.get("transport") or "http").lower()
        locator = {"transport": transport, "url": safe_url}
    else:
        transport = "stdio"
        # Hash the full process locator as a whole. Raw command/args/cwd may
        # contain credentials or private paths and never enter the identity.
        locator = {
            "transport": transport,
            "process_locator_digest": _sha256(
                {
                    "command": config.get("command"),
                    "args": config.get("args", []),
                    "cwd": config.get("cwd"),
                }
            ),
        }
    return transport, _sha256(locator)


def mcp_policy_descriptor(
    *,
    registered_name: str,
    remote_tool_name: str,
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
    server_info_name: str | None,
    server_info_version: str | None,
    effects: Iterable[ToolEffect] = (ToolEffect.UNKNOWN,),
    requires_confirmation: bool = False,
    retention: ResultRetention = ResultRetention.DEFAULT,
    policy_version: int = POLICY_VERSION,
    argument_resolver: Callable[[Mapping[str, Any]], frozenset[ToolEffect]] | None = None,
    required_parent_any_of: Iterable[str] = (),
) -> ToolPolicyDescriptor:
    transport, locator_digest = _mcp_transport_locator(config)
    remote_identity_digest = _sha256(
        {
            "server_name": str(server_info_name or "unknown"),
            "server_version": str(server_info_version or "unknown"),
            "remote_tool_name": str(remote_tool_name),
        }
    )
    source_identity = (
        f"mcp:{transport}:{locator_digest}:{remote_identity_digest}:{registered_name}"
    )
    descriptor = ToolPolicyDescriptor(
        effects=_normalize_effects(effects),
        requires_confirmation=requires_confirmation,
        retention=retention,
        source_identity=source_identity,
        schema_digest=schema_digest(schema),
        policy_version=policy_version,
        argument_resolver=argument_resolver,
        required_parent_any_of=frozenset(required_parent_any_of),
    )
    return validate_policy_descriptor(descriptor)


def derive_policy_identity(
    descriptor: ToolPolicyDescriptor,
    *,
    entry_generation: int,
) -> str:
    validate_policy_descriptor(descriptor)
    if type(entry_generation) is not int or entry_generation < 1:
        raise ValueError("entry_generation must be a positive integer")
    payload = {
        "source_identity": descriptor.source_identity,
        "schema_digest": descriptor.schema_digest,
        "policy_version": descriptor.policy_version,
        "effects": sorted(effect.value for effect in descriptor.effects),
        "retention": descriptor.retention.value,
        "requires_confirmation": descriptor.requires_confirmation,
        "argument_resolver_identity": argument_resolver_identity(
            descriptor.argument_resolver
        ),
        "required_parent_any_of": sorted(descriptor.required_parent_any_of),
        "entry_generation": entry_generation,
    }
    return f"policy:{_sha256(payload)}"


def build_authority_snapshot(
    policy_identities: Iterable[str],
    registry_generation: int,
) -> ToolAuthoritySnapshot:
    identities = frozenset(policy_identities)
    if any(
        not isinstance(identity, str) or not identity.startswith("policy:")
        for identity in identities
    ):
        raise ValueError("authority snapshot requires exact policy identities")
    if type(registry_generation) is not int or registry_generation < 0:
        raise ValueError("registry_generation must be a non-negative integer")
    fingerprint = _sha256(
        {
            "policy_identities": sorted(identities),
            "registry_generation": registry_generation,
        }
    )
    return ToolAuthoritySnapshot(
        policy_identities=identities,
        registry_generation=registry_generation,
        fingerprint=fingerprint,
    )


def is_derived_tool_eligible(
    descriptor: ToolPolicyDescriptor,
    parent_authority: ToolAuthoritySnapshot,
) -> bool:
    validate_policy_descriptor(descriptor)
    if not isinstance(parent_authority, ToolAuthoritySnapshot):
        raise TypeError("parent_authority must be ToolAuthoritySnapshot")
    required = descriptor.required_parent_any_of
    return not required or bool(required & parent_authority.policy_identities)


__all__ = [
    "POLICY_VERSION",
    "ToolEffect",
    "ResultRetention",
    "ToolPolicyDescriptor",
    "ToolAuthoritySnapshot",
    "schema_digest",
    "callable_identity",
    "argument_resolver_identity",
    "validate_policy_descriptor",
    "builtin_policy_descriptor",
    "mcp_policy_descriptor",
    "derive_policy_identity",
    "build_authority_snapshot",
    "is_derived_tool_eligible",
]
