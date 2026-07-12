"""Trusted tool policy metadata and authority snapshot contracts."""

import math

import pytest

from tools.tool_effects import (
    ResultRetention,
    ToolAuthoritySnapshot,
    ToolEffect,
    ToolPolicyDescriptor,
    build_authority_snapshot,
    builtin_policy_descriptor,
    derive_policy_identity,
    is_derived_tool_eligible,
    mcp_policy_descriptor,
    schema_digest,
    validate_policy_descriptor,
)


def _handler(args):
    return args


def _resolver(args):
    return frozenset({ToolEffect.READ_LOCAL})


def _schema(order: bool = False):
    properties = {"z": {"type": "string"}, "a": {"type": "integer"}}
    if order:
        properties = {"a": {"type": "integer"}, "z": {"type": "string"}}
    return {"type": "object", "properties": properties}


def test_schema_digest_is_compact_sorted_json_sha256_and_rejects_unsafe_values():
    assert schema_digest(_schema()) == schema_digest(_schema(order=True))
    assert len(schema_digest(_schema())) == 64

    with pytest.raises(ValueError, match="serializable"):
        schema_digest({"bad": object()})
    with pytest.raises(ValueError, match="finite"):
        schema_digest({"bad": math.nan})


def test_schema_digest_treats_empty_required_as_semantic_absence():
    without_required = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }
    empty_required = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": [],
    }
    nonempty_required = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    assert schema_digest(without_required) == schema_digest(empty_required)
    assert schema_digest(without_required) != schema_digest(nonempty_required)


def test_builtin_descriptor_roundtrips_sensitive_retention_and_argument_resolver():
    descriptor = builtin_policy_descriptor(
        name="derived_read",
        schema=_schema(),
        handler=_handler,
        effects={ToolEffect.READ_LOCAL},
        retention=ResultRetention.HANDLE_ONLY,
        requires_confirmation=True,
        argument_resolver=_resolver,
        required_parent_any_of={"policy:source-a"},
    )

    assert descriptor.effects == frozenset({ToolEffect.READ_LOCAL})
    assert descriptor.retention is ResultRetention.HANDLE_ONLY
    assert descriptor.argument_resolver is _resolver
    assert descriptor.required_parent_any_of == frozenset({"policy:source-a"})
    assert "test_tool_effects" in descriptor.source_identity
    assert "0x" not in descriptor.source_identity


def test_policy_identity_covers_all_policy_fields_and_entry_generation():
    descriptor = builtin_policy_descriptor(
        name="read",
        schema=_schema(),
        handler=_handler,
        effects={ToolEffect.READ_LOCAL},
        retention=ResultRetention.NO_SPILL,
        argument_resolver=_resolver,
    )
    first = derive_policy_identity(descriptor, entry_generation=7)
    assert first == derive_policy_identity(descriptor, entry_generation=7)
    assert first != derive_policy_identity(descriptor, entry_generation=8)

    changed = ToolPolicyDescriptor(
        effects=frozenset({ToolEffect.READ_REMOTE}),
        requires_confirmation=descriptor.requires_confirmation,
        retention=descriptor.retention,
        source_identity=descriptor.source_identity,
        schema_digest=descriptor.schema_digest,
        policy_version=descriptor.policy_version,
        argument_resolver=descriptor.argument_resolver,
        required_parent_any_of=descriptor.required_parent_any_of,
    )
    assert first != derive_policy_identity(changed, entry_generation=7)


def test_snapshot_fingerprint_is_deterministic_frozen_and_generation_bound():
    source = {"policy:b", "policy:a"}
    snapshot = build_authority_snapshot(source, registry_generation=11)
    source.add("policy:later")

    assert isinstance(snapshot, ToolAuthoritySnapshot)
    assert snapshot.policy_identities == frozenset({"policy:a", "policy:b"})
    assert snapshot == build_authority_snapshot(
        ["policy:a", "policy:b"], registry_generation=11
    )
    assert snapshot.fingerprint != build_authority_snapshot(
        ["policy:a", "policy:b"], registry_generation=12
    ).fingerprint


def test_derived_eligibility_requires_exact_policy_identity_not_tool_name():
    descriptor = builtin_policy_descriptor(
        name="derived",
        schema=_schema(),
        handler=_handler,
        effects={ToolEffect.READ_LOCAL},
        required_parent_any_of={"policy:exact-source"},
    )
    exact = build_authority_snapshot({"policy:exact-source"}, 1)
    with pytest.raises(ValueError, match="exact policy identities"):
        build_authority_snapshot({"derived", "source_tool"}, 1)
    name_only = ToolAuthoritySnapshot(
        policy_identities=frozenset({"derived", "source_tool"}),
        registry_generation=1,
        fingerprint="fabricated-name-only-snapshot",
    )

    assert is_derived_tool_eligible(descriptor, exact)
    assert not is_derived_tool_eligible(descriptor, name_only)


def test_malformed_descriptor_and_resolver_fail_closed():
    with pytest.raises((TypeError, ValueError)):
        builtin_policy_descriptor(
            name="bad",
            schema=_schema(),
            handler=_handler,
            effects={"READ_LOCAL"},
        )
    with pytest.raises((TypeError, ValueError), match="resolver"):
        builtin_policy_descriptor(
            name="bad",
            schema=_schema(),
            handler=_handler,
            effects={ToolEffect.READ_LOCAL},
            argument_resolver=lambda args: frozenset({ToolEffect.READ_LOCAL}),
        )


def test_descriptor_rejects_mutable_or_noncanonical_effect_collections():
    descriptor = ToolPolicyDescriptor(
        effects=(ToolEffect.READ_LOCAL,),  # type: ignore[arg-type]
        requires_confirmation=False,
        retention=ResultRetention.DEFAULT,
        source_identity="builtin:test:handler",
        schema_digest=schema_digest(_schema()),
        policy_version=1,
        argument_resolver=None,
        required_parent_any_of=frozenset(),
    )

    with pytest.raises(TypeError, match="frozenset"):
        validate_policy_descriptor(descriptor)


def test_mcp_identity_hashes_untrusted_metadata_and_binds_cwd():
    secret = "do-not-persist-this-secret"

    def make_descriptor(*, cwd: str, version: str):
        return mcp_policy_descriptor(
            registered_name="mcp_safe_lookup",
            remote_tool_name=f"lookup-{secret}",
            schema=_schema(),
            server_info_name=f"server-{secret}",
            server_info_version=f"{version}-{secret}",
            config={
                "command": "python",
                "args": ["server.py", "--token", secret],
                "cwd": cwd,
            },
        )

    first = make_descriptor(cwd="/srv/first", version="v1")
    moved = make_descriptor(cwd="/srv/second", version="v1")
    upgraded = make_descriptor(cwd="/srv/first", version="v2")

    assert secret not in first.source_identity
    assert "/srv/first" not in first.source_identity
    assert first.source_identity != moved.source_identity
    assert first.source_identity != upgraded.source_identity
