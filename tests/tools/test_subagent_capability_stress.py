from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from types import SimpleNamespace

import pytest

from agent.subagent_tool_policy import (
    ToolAuthorizationError,
    ToolNamePolicy,
    authorize_subagent_call,
)
from tools.registry import registry
from tools.subagent_sessions import (
    RetainedSubagentSession,
    claim_retained_subagent_session,
    clear_retained_subagent_sessions,
    release_retained_subagent_session,
    retain_subagent_session,
    retained_subagent_transcript_bytes,
    update_retained_history,
)
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
def stress_read_probe():
    name = "task12_parallel_read_probe"
    registry.deregister(name)
    calls: list[str] = []
    lock = threading.Lock()

    def handler(args, **_kwargs):
        with lock:
            calls.append(str(args["value"]))
        return json.dumps({"ok": True})

    registry.register(
        name=name,
        toolset="task12-test",
        schema=_schema(name),
        handler=handler,
        descriptor=builtin_policy_descriptor(
            name=name,
            schema=_schema(name),
            handler=handler,
            effects={ToolEffect.READ_LOCAL},
            retention=ResultRetention.NO_SPILL,
        ),
    )
    try:
        yield name, calls, handler
    finally:
        registry.deregister(name)
        clear_retained_subagent_sessions()


def _readonly_agent(name: str):
    identity = registry.resolved_policy_identity(name)
    assert isinstance(identity, str)
    return SimpleNamespace(
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


def test_parallel_authorize_dispatch_never_uses_stale_same_name_identity(
    stress_read_probe,
):
    name, calls, original_handler = stress_read_probe
    agent = _readonly_agent(name)
    start_replacement = threading.Event()

    def replacement(args, **_kwargs):
        calls.append(f"replacement:{args['value']}")
        return json.dumps({"replacement": True})

    def replace_repeatedly():
        start_replacement.wait(timeout=2)
        for _ in range(30):
            registry.register(
                name=name,
                toolset="task12-test",
                schema=_schema(name),
                handler=replacement,
                descriptor=builtin_policy_descriptor(
                    name=name,
                    schema=_schema(name),
                    handler=replacement,
                    effects={ToolEffect.READ_LOCAL},
                    retention=ResultRetention.NO_SPILL,
                ),
                override=True,
            )

    replacer = threading.Thread(target=replace_repeatedly)
    replacer.start()

    def one_call(index: int) -> str:
        args = {"value": str(index)}
        try:
            authorized = authorize_subagent_call(agent, name, args)
            if index == 0:
                start_replacement.set()
                time.sleep(0.002)
            registry.dispatch(name, args, authorization=authorized)
            return "success"
        except ToolAuthorizationError:
            return "denied"

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(one_call, range(100)))
    start_replacement.set()
    replacer.join(timeout=2)
    assert not replacer.is_alive()

    assert len(outcomes) == 100
    assert set(outcomes).issubset({"success", "denied"})
    assert len(calls) == outcomes.count("success")
    assert all(not call.startswith("replacement:") for call in calls)
    assert original_handler is not replacement


def test_two_hundred_retained_session_cycles_stay_bounded_and_do_not_leak_fds(
    stress_read_probe,
    tmp_path,
):
    psutil = pytest.importorskip("psutil")
    process = psutil.Process()
    if not hasattr(process, "num_fds"):
        pytest.skip("num_fds is unavailable on this platform")

    name, _calls, _handler = stress_read_probe
    identity = registry.resolved_policy_identity(name)
    assert isinstance(identity, str)
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    baseline_fds = process.num_fds()
    byte_budget = 64 * 1024

    for index in range(200):
        now = time.time()
        agent_id = f"stress-agent-{index}"
        retain_subagent_session(
            RetainedSubagentSession(
                agent_id=agent_id,
                parent_session_id="stress-parent",
                subagent_type="general-purpose",
                workspace_path=str(tmp_path),
                model="fake-model",
                provider="fake-provider",
                conversation_history=[
                    {"role": "user", "content": f"cycle-{index}"}
                ],
                created_at=now,
                expires_at=now + 60,
                profile_id="stress-profile",
                canonical_profile_home=str(profile_home.resolve()),
                original_policy_identities=frozenset({identity}),
                original_governance_fingerprint="a" * 64,
                effective_allowed_tool_names=frozenset({name}),
            ),
            max_records=8,
            max_total_bytes=byte_budget,
        )
        claim_retained_subagent_session(agent_id)
        update_retained_history(
            agent_id,
            [{"role": "assistant", "content": f"updated-{index}"}],
            max_total_bytes=byte_budget,
        )
        release_retained_subagent_session(agent_id)

    assert retained_subagent_transcript_bytes() <= byte_budget
    assert process.num_fds() <= baseline_fds + 3
