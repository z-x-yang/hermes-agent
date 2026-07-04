"""End-to-end tests for persistent origin-overload (502/503/524/529) self-heal.

After N consecutive overload errors on one origin, the loop compresses history
and retries with a lighter request (mirrors manual /compress); if compression
can't reduce (bad window 502s the aux call too), it fails honestly. Fixture
mirrors tests/run_agent/test_413_compression.py.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.conversation_loop as conversation_loop
import run_agent
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Short-circuit retry backoff so consecutive 502s run instantly."""
    monkeypatch.setattr(conversation_loop, "jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0, raising=False)
    monkeypatch.setattr(conversation_loop.time, "sleep", lambda *_a, **_k: None)


def _mock_response(content="ok", finish_reason="stop"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp


def _make_overload(status=502, message=None):
    err = Exception(message or f"HTTP {status}: origin_bad_gateway")
    err.status_code = status
    return err


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://gptcodex.top/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = True
        a.save_trajectories = False
        a._persistent_overload_threshold = 5
        # Self-heal needs the retry budget to outlast the streak threshold
        # (production config.yaml sets api_max_retries: 10 explicitly; the
        # code default is 3, under which a 5-streak can never accumulate).
        a._api_max_retries = 10
        return a


def _prefill():
    return [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]


def test_5_consecutive_502_triggers_compression(agent):
    """5 × 502 → compress → lighter request succeeds on the 6th call."""
    agent.client.chat.completions.create.side_effect = [
        _make_overload(), _make_overload(), _make_overload(),
        _make_overload(), _make_overload(),
        _mock_response(content="recovered after compression"),
    ]
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        mock_compress.return_value = (
            [{"role": "user", "content": "summary"}], "compressed prompt",
        )
        result = agent.run_conversation("hello", conversation_history=_prefill())

    mock_compress.assert_called_once()
    assert result["completed"] is True
    assert result["final_response"] == "recovered after compression"


def test_below_threshold_no_compression(agent):
    """2 × 502 then success — a transient blip must NOT compress."""
    agent.client.chat.completions.create.side_effect = [
        _make_overload(), _make_overload(), _mock_response(content="ok"),
    ]
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=_prefill())

    mock_compress.assert_not_called()
    assert result["completed"] is True


def test_503_overload_also_triggers(agent):
    """503 (FailoverReason.overloaded) self-heals identically to 502."""
    agent.client.chat.completions.create.side_effect = [
        _make_overload(status=503), _make_overload(status=503),
        _make_overload(status=503), _make_overload(status=503),
        _make_overload(status=503),
        _mock_response(content="ok after overload"),
    ]
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        mock_compress.return_value = (
            [{"role": "user", "content": "summary"}], "compressed",
        )
        result = agent.run_conversation("hello", conversation_history=_prefill())

    mock_compress.assert_called_once()
    assert result["completed"] is True


def test_compression_noop_fails_honestly(agent):
    """No-op compression must not loop forever and must explain the attempt."""
    agent.client.chat.completions.create.side_effect = _make_overload()
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        mock_compress.side_effect = lambda msgs, *a, **k: (msgs, agent._cached_system_prompt)
        result = agent.run_conversation("hello", conversation_history=_prefill())

    assert result.get("completed") is not True
    blob = (result.get("final_response", "") or "") + (result.get("error", "") or "")
    assert "压缩" in blob or "中转站持续过载" in blob


def test_compression_disabled_no_self_heal(agent):
    """compression_enabled=False → never self-heals."""
    agent.compression_enabled = False
    agent.client.chat.completions.create.side_effect = _make_overload()
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=_prefill())

    mock_compress.assert_not_called()
    assert result.get("completed") is not True


def test_origin_switch_resets_streak(agent):
    """A fallback that switches base_url starts a fresh overload streak."""
    calls = {"n": 0}
    first_compress_at = {"n": None}

    def _side(**_kwargs):
        calls["n"] += 1
        # Before the 5th call, simulate a fallback that switched origin.
        if calls["n"] == 5:
            agent.base_url = "https://other-relay.example/v1"
        raise _make_overload()

    agent.client.chat.completions.create.side_effect = _side

    def _compress(messages, system_message, **_k):
        if first_compress_at["n"] is None:
            first_compress_at["n"] = calls["n"]
        return ([{"role": "user", "content": "summary"}], "compressed")

    with (
        patch.object(agent, "_compress_context", side_effect=_compress),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("hello", conversation_history=_prefill())

    assert first_compress_at["n"] is not None, "self-heal should eventually fire"
    assert first_compress_at["n"] >= 9, (
        f"first compression fired at call {first_compress_at['n']} — the new "
        f"origin's streak leaked from the old origin"
    )


def test_no_self_heal_no_overload_label(agent):
    """If self-heal never fires, terminal error must not use the overload label."""
    agent._persistent_overload_threshold = 100  # unreachable within max_retries
    agent.client.chat.completions.create.side_effect = _make_overload()
    with (
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=_prefill())

    mock_compress.assert_not_called()
    blob = (result.get("final_response", "") or "") + (result.get("error", "") or "")
    assert "中转站持续过载" not in blob
    assert result.get("completed") is not True