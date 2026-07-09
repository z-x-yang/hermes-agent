"""Tests for recording accepted-request rough baselines before provider calls."""

from types import SimpleNamespace

from agent.conversation_loop import _record_pending_compression_request_estimate


class _RecordingCompressor:
    def __init__(self):
        self.estimate_args = None
        self.fingerprint_args = None
        self.recorded = None

    def estimate_provider_request_tokens(self, messages, *, system_prompt="", tools=None):
        self.estimate_args = (messages, system_prompt, tools)
        return 123_456

    def preflight_request_fingerprint(self, *, system_prompt="", tools=None):
        self.fingerprint_args = (system_prompt, tools)
        return "route-fp"

    def record_pending_request_estimate(self, rough_tokens, *, fingerprint=""):
        self.recorded = (rough_tokens, fingerprint)


def test_does_not_double_count_system_prompt_when_api_messages_already_include_system():
    compressor = _RecordingCompressor()
    agent = SimpleNamespace(
        context_compressor=compressor,
        tools=[{"type": "function", "function": {"name": "demo"}}],
    )
    api_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]

    _record_pending_compression_request_estimate(
        agent,
        api_messages,
        active_system_prompt="system",
    )

    assert compressor.estimate_args == (api_messages, "", agent.tools)
    assert compressor.fingerprint_args == ("system", agent.tools)
    assert compressor.recorded == (123_456, "route-fp")


def test_records_pending_compression_request_estimate_from_same_shape_as_preflight():
    compressor = _RecordingCompressor()
    agent = SimpleNamespace(
        context_compressor=compressor,
        tools=[{"type": "function", "function": {"name": "demo"}}],
    )
    api_messages = [{"role": "user", "content": "hello"}]

    _record_pending_compression_request_estimate(
        agent,
        api_messages,
        active_system_prompt="system",
    )

    assert compressor.estimate_args == (api_messages, "system", agent.tools)
    assert compressor.fingerprint_args == ("system", agent.tools)
    assert compressor.recorded == (123_456, "route-fp")
