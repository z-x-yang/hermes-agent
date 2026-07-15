from __future__ import annotations

from types import SimpleNamespace

from agent.conversation_compression import compress_context


class _TodoStore:
    def format_for_injection(self):
        return None


class _Compressor:
    def __init__(self, *, aborted: bool = False, deferred: bool = False):
        self._last_compress_aborted = aborted
        self._last_compress_deferred = deferred
        self._last_summary_error = "summary failed" if aborted else None
        self._last_aux_model_failure_model = None
        self._last_aux_model_failure_error = None
        self.compression_count = 1
        self.last_compression_rough_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.awaiting_real_usage_after_compression = False

    def compress(self, messages, **_kwargs):
        if self._last_compress_aborted or self._last_compress_deferred:
            return messages
        return [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}]


class _Agent(SimpleNamespace):
    pass


def _agent(*, aborted: bool = False, deferred: bool = False):
    return _Agent(
        session_id="sess-1",
        model="test/model",
        platform="cli",
        tools=[],
        log_prefix="",
        compression_in_place=False,
        compression_repeated_warning=False,
        _compression_feasibility_checked=True,
        _memory_manager=None,
        _session_db=None,
        _cached_system_prompt="SYSTEM",
        _todo_store=_TodoStore(),
        _runtime_context_status_mode="inject",
        _runtime_context_status_audit_enabled=False,
        _pending_runtime_context_statuses=[],
        _queued_runtime_context_status_keys=set(),
        context_compressor=_Compressor(aborted=aborted, deferred=deferred),
        _emit_status=lambda *_a, **_k: None,
        _emit_warning=lambda *_a, **_k: None,
        _invalidate_system_prompt=lambda: None,
        _build_system_prompt=lambda _system_message=None: "SYSTEM-NEW",
        commit_memory_session=lambda *_a, **_k: None,
        event_callback=None,
    )


def test_successful_compression_queues_post_runtime_context_status():
    agent = _agent()

    compressed, new_system = compress_context(
        agent,
        [{"role": "user", "content": "hello"}],
        None,
        approx_tokens=1000,
        trigger_reason="token_threshold",
        trigger_tokens=1000,
        trigger_threshold_tokens=900,
        trigger_message_count=1,
    )

    assert new_system == "SYSTEM-NEW"
    assert compressed == [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}]
    assert len(agent._pending_runtime_context_statuses) == 1
    pending = agent._pending_runtime_context_statuses[0]
    assert pending["kind"] == "post_compression_completed"
    assert "immediately before this model call" in pending["content"]
    assert pending["metadata"]["trigger_reason"] == "token_threshold"


def test_aborted_compression_does_not_queue_post_runtime_context_status():
    agent = _agent(aborted=True)
    messages = [{"role": "user", "content": "hello"}]

    compressed, new_system = compress_context(
        agent,
        messages,
        None,
        approx_tokens=1000,
        trigger_reason="token_threshold",
    )

    assert compressed is messages
    assert new_system == "SYSTEM"
    assert agent._pending_runtime_context_statuses == []


def test_deferred_compression_does_not_rotate_or_queue_runtime_status():
    warnings = []
    agent = _agent(deferred=True)
    agent._emit_warning = warnings.append
    messages = [{"role": "user", "content": "hello"}]

    compressed, new_system = compress_context(
        agent,
        messages,
        None,
        approx_tokens=1000,
        trigger_reason="token_threshold",
    )

    assert compressed is messages
    assert new_system == "SYSTEM"
    assert warnings == []
    assert agent._pending_runtime_context_statuses == []
