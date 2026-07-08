from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.conversation_loop as conversation_loop
import run_agent
from agent.runtime_context_status import build_post_compression_notice, queue_runtime_context_status
from hermes_cli.config import DEFAULT_CONFIG
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(conversation_loop, "jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0, raising=False)
    monkeypatch.setattr(conversation_loop.time, "sleep", lambda *_a, **_k: None)


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://example.test/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        a._runtime_context_status_mode = "inject"
        a._runtime_context_status_audit_enabled = False
        a._pending_runtime_context_statuses = []
        a._queued_runtime_context_status_keys = set()
        return a


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


def test_pending_runtime_status_is_appended_to_api_copy_not_persisted(agent):
    captured = {}

    def _create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _mock_response("done")

    agent.client.chat.completions.create.side_effect = _create
    queue_runtime_context_status(
        agent,
        build_post_compression_notice(),
        kind="post_compression_completed",
        dedupe_key="post:1",
        metadata={"compression_count": 1},
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "current user request",
            conversation_history=[
                {"role": "user", "content": "previous user"},
                {"role": "assistant", "content": "previous assistant"},
            ],
        )

    assert result["completed"] is True
    sent_messages = captured["messages"]
    assert sent_messages[-1]["role"] == "user"
    assert sent_messages[-1]["content"].startswith("current user request\n\n<hermes-runtime-context>")
    assert "immediately before this model call" in sent_messages[-1]["content"]
    persisted_blob = json.dumps(result["messages"], ensure_ascii=False)
    assert "<hermes-runtime-context>" not in persisted_blob
    assert agent._pending_runtime_context_statuses == []


def test_shadow_runtime_status_does_not_change_api_payload(agent):
    captured = {}
    agent._runtime_context_status_mode = "shadow"

    def _create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _mock_response("done")

    agent.client.chat.completions.create.side_effect = _create
    queue_runtime_context_status(
        agent,
        build_post_compression_notice(),
        kind="post_compression_completed",
        dedupe_key="post:1",
        metadata={"compression_count": 1},
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("current user request")

    assert captured["messages"][-1] == {"role": "user", "content": "current user request"}
    assert agent._pending_runtime_context_statuses == []


def test_default_config_keeps_runtime_context_status_disabled():
    cfg = DEFAULT_CONFIG["compression"]["runtime_context_status"]
    assert cfg["mode"] == "off"
    assert cfg["audit"] is True
    assert cfg["near_threshold_ratio"] == 0.90


def test_agent_initializes_runtime_context_status_from_config(monkeypatch):
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["compression"] = deepcopy(DEFAULT_CONFIG["compression"])
    cfg["compression"]["runtime_context_status"] = {
        "mode": "shadow",
        "audit": False,
        "near_threshold_ratio": 0.87,
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://example.test/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert getattr(a, "_runtime_context_status_mode") == "shadow"
    assert getattr(a, "_runtime_context_status_audit_enabled") is False
    assert getattr(a, "_runtime_context_status_near_threshold_ratio") == 0.87
    assert getattr(a, "_pending_runtime_context_statuses") == []
    assert getattr(a, "_queued_runtime_context_status_keys") == set()
