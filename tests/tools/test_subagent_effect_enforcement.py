from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from model_tools import handle_function_call as _real_handle_function_call
from tools.registry import registry
from tools.tool_effects import (
    ResultRetention,
    ToolEffect,
    build_authority_snapshot,
    builtin_policy_descriptor,
)


def _schema(name: str) -> dict:
    return {
        "name": name,
        "description": name,
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
    }


@pytest.fixture
def registered_effect_tools():
    names = ["task6_read_probe", "task6_unknown_probe", "task6_mutable_probe"]
    calls: dict[str, list[dict]] = {name: [] for name in names}
    for name in names:
        effect = ToolEffect.READ_LOCAL if name == "task6_read_probe" else ToolEffect.UNKNOWN

        def handler(args, *, _name=name, **_kwargs):
            calls[_name].append(dict(args))
            return json.dumps({"ok": True, "name": _name, "args": args})

        registry.register(
            name=name,
            toolset="task6-test",
            schema=_schema(name),
            handler=handler,
            descriptor=builtin_policy_descriptor(
                name=name,
                schema=_schema(name),
                handler=handler,
                effects={effect},
                retention=ResultRetention.NO_SPILL,
            ),
            override=True,
        )
    try:
        yield calls
    finally:
        for name in names:
            registry.deregister(name)


def _readonly_agent(*names: str):
    from agent.subagent_tool_policy import ToolNamePolicy

    identities = {registry.resolved_policy_identity(name) for name in names}
    assert None not in identities
    return SimpleNamespace(
        _subagent_tool_policy=ToolNamePolicy(
            allowed_names=frozenset(names),
            allowed_effects=frozenset({ToolEffect.READ_LOCAL, ToolEffect.READ_REMOTE}),
            authority_snapshot=build_authority_snapshot(
                identities,
                registry_generation=registry._generation,
            ),
            profile_name="Explore",
        )
    )


def test_child_policy_without_exact_authority_snapshot_fails_closed():
    from agent.subagent_tool_policy import build_child_tool_policy

    policy = build_child_tool_policy(
        child=SimpleNamespace(),
        parent=SimpleNamespace(),
        profile_name="Explore",
        profile_allowed_names=frozenset({"read_file"}),
    )

    assert policy.allowed_names == frozenset()
    assert policy.authority_snapshot is not None
    assert policy.authority_snapshot.policy_identities == frozenset()


def test_nested_child_intersects_parent_current_allowed_names(
    registered_effect_tools,
):
    from agent.subagent_tool_policy import (
        ToolNamePolicy,
        build_child_tool_policy,
    )

    broad_parent = _readonly_agent(
        "task6_read_probe", "task6_unknown_probe"
    )
    snapshot = broad_parent._subagent_tool_policy.authority_snapshot
    parent = SimpleNamespace(
        _parent_tool_authority_snapshot=snapshot,
        _subagent_tool_policy=ToolNamePolicy(
            allowed_names=frozenset({"task6_read_probe"}),
            allowed_effects=None,
            authority_snapshot=snapshot,
            profile_name="general-purpose",
        ),
    )
    child = SimpleNamespace(_parent_tool_authority_snapshot=snapshot)

    policy = build_child_tool_policy(
        child=child,
        parent=parent,
        profile_name="general-purpose",
        profile_allowed_names=None,
    )

    assert policy.allowed_names == frozenset({"task6_read_probe"})


def test_read_call_authorizes_and_unknown_effect_denies_backend(registered_effect_tools):
    from agent.subagent_tool_policy import (
        ToolAuthorizationError,
        authorize_subagent_call,
    )

    agent = _readonly_agent("task6_read_probe", "task6_unknown_probe")
    authorized = authorize_subagent_call(
        agent, "task6_read_probe", {"value": "safe"}
    )
    assert authorized.effects == frozenset({ToolEffect.READ_LOCAL})

    with pytest.raises(ToolAuthorizationError, match="effect"):
        authorize_subagent_call(agent, "task6_unknown_probe", {"value": "blocked"})
    assert registered_effect_tools["task6_unknown_probe"] == []


def test_same_name_replacement_invalidates_original_ceiling(registered_effect_tools):
    from agent.subagent_tool_policy import ToolAuthorizationError, authorize_subagent_call

    agent = _readonly_agent("task6_read_probe")
    old_identity = registry.resolved_policy_identity("task6_read_probe")

    def replacement(args):
        registered_effect_tools["task6_read_probe"].append(dict(args))
        return "replacement"

    registry.register(
        name="task6_read_probe",
        toolset="task6-test",
        schema=_schema("task6_read_probe"),
        handler=replacement,
        descriptor=builtin_policy_descriptor(
            name="task6_read_probe",
            schema=_schema("task6_read_probe"),
            handler=replacement,
            effects={ToolEffect.READ_LOCAL},
            retention=ResultRetention.NO_SPILL,
        ),
        override=True,
    )
    assert registry.resolved_policy_identity("task6_read_probe") != old_identity

    with pytest.raises(ToolAuthorizationError, match="identity"):
        authorize_subagent_call(agent, "task6_read_probe", {"value": "stale"})
    assert registered_effect_tools["task6_read_probe"] == []


def test_identity_precheck_precedes_argument_coercion(
    registered_effect_tools, monkeypatch
):
    from agent.subagent_tool_policy import ToolAuthorizationError, authorize_subagent_call

    agent = _readonly_agent("task6_read_probe")

    def replacement(args, **kwargs):
        return "replacement"

    registry.register(
        name="task6_read_probe",
        toolset="task6-test",
        schema=_schema("task6_read_probe"),
        handler=replacement,
        descriptor=builtin_policy_descriptor(
            name="task6_read_probe",
            schema=_schema("task6_read_probe"),
            handler=replacement,
            effects={ToolEffect.READ_LOCAL},
            retention=ResultRetention.NO_SPILL,
        ),
        override=True,
    )
    monkeypatch.setattr(
        "model_tools.coerce_tool_args",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("coercion ran before identity precheck")
        ),
    )

    with pytest.raises(ToolAuthorizationError, match="identity"):
        authorize_subagent_call(
            agent, "task6_read_probe", {"value": "stale"}
        )


def test_final_registry_dispatch_rejects_mutated_arguments(registered_effect_tools):
    from agent.subagent_tool_policy import (
        ToolAuthorizationError,
        authorize_subagent_call,
    )

    agent = _readonly_agent("task6_read_probe")
    authorized = authorize_subagent_call(
        agent, "task6_read_probe", {"value": "original"}
    )

    with pytest.raises(ToolAuthorizationError, match="arguments"):
        registry.dispatch(
            "task6_read_probe",
            {"value": "mutated"},
            authorization=authorized,
        )
    assert registered_effect_tools["task6_read_probe"] == []

    result = registry.dispatch(
        "task6_read_probe",
        {"value": "original"},
        authorization=authorized,
    )
    assert json.loads(result)["ok"] is True
    assert registered_effect_tools["task6_read_probe"] == [{"value": "original"}]


def test_final_dispatch_uses_the_atomically_verified_entry(
    registered_effect_tools, monkeypatch
):
    from agent import subagent_tool_policy as policy_module

    agent = _readonly_agent("task6_read_probe")
    authorized = policy_module.authorize_subagent_call(
        agent, "task6_read_probe", {"value": "safe"}
    )
    replacement_calls = []
    original_verify = policy_module.verify_authorized_tool_call

    def replacement(args, **_kwargs):
        replacement_calls.append(dict(args))
        return json.dumps({"replacement": True})

    def verify_then_replace(*args, **kwargs):
        normalized = original_verify(*args, **kwargs)
        registry.register(
            name="task6_read_probe",
            toolset="task6-test",
            schema=_schema("task6_read_probe"),
            handler=replacement,
            descriptor=builtin_policy_descriptor(
                name="task6_read_probe",
                schema=_schema("task6_read_probe"),
                handler=replacement,
                effects={ToolEffect.READ_LOCAL},
                retention=ResultRetention.NO_SPILL,
            ),
            override=True,
        )
        return normalized

    monkeypatch.setattr(policy_module, "verify_authorized_tool_call", verify_then_replace)
    result = json.loads(
        registry.dispatch(
            "task6_read_probe",
            {"value": "safe"},
            authorization=authorized,
        )
    )

    assert result["ok"] is True
    assert registered_effect_tools["task6_read_probe"] == [{"value": "safe"}]
    assert replacement_calls == []


def test_dynamic_schema_tools_dispatch_with_the_resolved_identity(monkeypatch):
    from agent.subagent_tool_policy import ToolNamePolicy, authorize_subagent_call
    from tools.registry import ToolRegistry

    import tools.delegate_tool  # noqa: F401
    import tools.image_generation_tool  # noqa: F401
    import tools.video_generation_tool  # noqa: F401

    source_registry = registry
    cases = {
        "delegate_task": {"goal": "inspect"},
        "image_generate": {"prompt": "draw"},
        "video_generate": {"prompt": "animate"},
    }

    for name, args in cases.items():
        source_entry = source_registry.get_entry(name)
        assert source_entry is not None
        assert source_entry.dynamic_schema_overrides is not None
        local = ToolRegistry()
        calls = []

        def handler(call_args, *, _name=name, **_kwargs):
            calls.append((_name, dict(call_args)))
            return json.dumps({"ok": _name})

        local.register(
            name=name,
            toolset=source_entry.toolset,
            schema=source_entry.schema,
            handler=handler,
            check_fn=lambda: True,
            dynamic_schema_overrides=source_entry.dynamic_schema_overrides,
            descriptor=source_entry.policy_descriptor,
        )
        definitions = local.get_definitions({name})
        snapshot = local.authority_snapshot_for_definitions(definitions)
        child = SimpleNamespace(
            _subagent_tool_policy=ToolNamePolicy(
                allowed_names=frozenset({name}),
                allowed_effects=None,
                authority_snapshot=snapshot,
                profile_name="general-purpose",
            )
        )
        monkeypatch.setattr("tools.registry.registry", local)
        authorized = authorize_subagent_call(child, name, args)

        result = json.loads(local.dispatch(name, args, authorization=authorized))

        assert result == {"ok": name}
        assert calls == [(name, args)]


def test_direct_model_tools_path_enforces_readonly_policy(registered_effect_tools):
    import model_tools

    agent = _readonly_agent("task6_read_probe", "task6_unknown_probe")
    ok = json.loads(
        _real_handle_function_call(
            "task6_read_probe",
            {"value": "safe"},
            agent=agent,
            enabled_toolsets=["task6-test"],
        )
    )
    denied = json.loads(
        _real_handle_function_call(
            "task6_unknown_probe",
            {"value": "blocked"},
            agent=agent,
            enabled_toolsets=["task6-test"],
        )
    )

    assert ok["ok"] is True
    assert "error" in denied
    assert "effect" in denied["error"]
    assert registered_effect_tools["task6_read_probe"] == [{"value": "safe"}]
    assert registered_effect_tools["task6_unknown_probe"] == []


def test_readonly_direct_path_skips_request_and_execution_middleware(
    registered_effect_tools, monkeypatch
):
    import model_tools

    agent = _readonly_agent("task6_read_probe")
    monkeypatch.setattr(
        "hermes_cli.middleware.apply_tool_request_middleware",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("readonly request middleware ran")
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_tool_execution_middleware",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("readonly execution middleware ran")
        ),
    )

    result = json.loads(
        _real_handle_function_call(
            "task6_read_probe",
            {"value": "safe"},
            agent=agent,
            enabled_toolsets=["task6-test"],
        )
    )

    assert result["ok"] is True
    assert registered_effect_tools["task6_read_probe"] == [{"value": "safe"}]


def test_no_spill_retention_never_writes_generic_result_storage():
    from tools.tool_result_storage import maybe_persist_tool_result

    class RecordingEnv:
        def __init__(self):
            self.calls = []

        def execute(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"returncode": 0}

    env = RecordingEnv()
    content = "sensitive" * 100
    retained = maybe_persist_tool_result(
        content,
        "task6_read_probe",
        "call-1",
        env=env,
        threshold=1,
        retention=ResultRetention.NO_SPILL,
    )

    assert retained == content
    assert env.calls == []


def test_handle_only_observer_projection_is_bounded_without_changing_live_result():
    from tools.tool_result_storage import project_result_for_retention

    content = "visible-prefix-" + ("secret-payload-" * 100)
    projected = project_result_for_retention(
        content,
        ResultRetention.HANDLE_ONLY,
        excerpt_chars=16,
    )

    assert content.startswith("visible-prefix-")
    assert len(projected) < 400
    assert "secret-payload-secret-payload" not in projected
    payload = json.loads(projected)
    assert payload["retention"] == "handle_only"
    assert payload["size_chars"] == len(content)
    assert payload["sha256"]
    assert payload["excerpt"] == content[:16]


def test_handle_only_live_result_uses_bounded_head_tail_projection():
    from tools.tool_result_storage import bound_live_result_for_retention

    content = "HEAD-" + ("private-middle-" * 100) + "-TAIL"
    bounded = bound_live_result_for_retention(
        content,
        ResultRetention.HANDLE_ONLY,
        max_chars=240,
    )

    assert len(bounded) <= 240
    assert bounded.startswith("HEAD-")
    assert bounded.endswith("-TAIL")
    assert "HANDLE_ONLY content truncated" in bounded
    assert hashlib.sha256(content.encode("utf-8")).hexdigest() in bounded


def test_handle_only_direct_path_projects_post_hook_but_returns_live_content(monkeypatch):
    import model_tools
    from agent.subagent_tool_policy import ToolNamePolicy

    name = "task6_handle_only_probe"
    schema = _schema(name)
    live_content = "private-body-" * 100

    def handler(args, **kwargs):
        return live_content

    registry.register(
        name=name,
        toolset="task6-test",
        schema=schema,
        handler=handler,
        descriptor=builtin_policy_descriptor(
            name=name,
            schema=schema,
            handler=handler,
            effects={ToolEffect.READ_REMOTE},
            retention=ResultRetention.HANDLE_ONLY,
        ),
        override=True,
    )
    try:
        identity = registry.resolved_policy_identity(name)
        assert isinstance(identity, str)
        agent = SimpleNamespace(
            _subagent_tool_policy=ToolNamePolicy(
                allowed_names=frozenset({name}),
                allowed_effects=frozenset(
                    {ToolEffect.READ_LOCAL, ToolEffect.READ_REMOTE}
                ),
                authority_snapshot=build_authority_snapshot(
                    {identity}, registry_generation=registry._generation
                ),
                profile_name="Explore",
            )
        )
        observed = []
        monkeypatch.setattr(
            model_tools,
            "_emit_post_tool_call_hook",
            lambda **kwargs: observed.append(kwargs["result"]),
        )

        result = _real_handle_function_call(
            name,
            {"value": "safe"},
            agent=agent,
            enabled_toolsets=["task6-test"],
        )

        assert result == live_content
        assert len(observed) == 1
        projection = json.loads(observed[0])
        assert projection["retention"] == "handle_only"
        assert live_content not in observed[0]
    finally:
        registry.deregister(name)


def test_tool_search_scope_cache_is_bound_to_authority_fingerprint(monkeypatch):
    from agent.subagent_tool_policy import ToolNamePolicy
    from agent.tool_executor import _tool_search_scoped_names

    names = ("mcp_task6_scope_a", "mcp_task6_scope_b")
    try:
        for name in names:
            schema = _schema(name)

            def handler(args, **kwargs):
                return "ok"

            registry.register(
                name=name,
                toolset="task6-test",
                schema=schema,
                handler=handler,
                descriptor=builtin_policy_descriptor(
                    name=name,
                    schema=schema,
                    handler=handler,
                    effects={ToolEffect.READ_REMOTE},
                    retention=ResultRetention.NO_SPILL,
                ),
                override=True,
            )
        identity_values = [registry.resolved_policy_identity(name) for name in names]
        assert all(isinstance(identity, str) for identity in identity_values)
        identities = {
            name: identity
            for name, identity in zip(names, identity_values)
            if isinstance(identity, str)
        }
        broad = ToolNamePolicy(
            allowed_names=frozenset(names),
            allowed_effects=frozenset(
                {ToolEffect.READ_LOCAL, ToolEffect.READ_REMOTE}
            ),
            authority_snapshot=build_authority_snapshot(
                identities.values(), registry_generation=registry._generation
            ),
            profile_name="Explore",
        )
        agent = SimpleNamespace(
            enabled_toolsets=None,
            disabled_toolsets=None,
            _subagent_tool_policy=broad,
        )
        import model_tools

        monkeypatch.setattr(
            model_tools,
            "get_tool_definitions",
            lambda **kwargs: [
                {"type": "function", "function": _schema(name)}
                for name in names
            ],
        )
        assert _tool_search_scoped_names(agent) == frozenset(names)

        only_a = ToolNamePolicy(
            allowed_names=frozenset({names[0]}),
            allowed_effects=broad.allowed_effects,
            authority_snapshot=build_authority_snapshot(
                {identities[names[0]]}, registry_generation=registry._generation
            ),
            profile_name="Explore",
        )
        agent._subagent_tool_policy = only_a

        assert _tool_search_scoped_names(agent) == frozenset({names[0]})
    finally:
        for name in names:
            registry.deregister(name)
