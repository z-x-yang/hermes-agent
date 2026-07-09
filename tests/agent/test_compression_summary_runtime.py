from types import SimpleNamespace

from agent.compression_summary_runtime import make_summary_runtime, extract_summary_cache_stats


class FakeSummaryAgent:
    provider = "gptcodex"
    model = "gpt-5.5"
    api_mode = "codex_responses"
    base_url = "https://example.invalid/v1"
    reasoning_effort = None
    tools = []
    session_api_calls = 0
    _fallback_chain = []
    _fallback_index = 0

    def __init__(self):
        self.context_compressor = SimpleNamespace(context_length=100_000)
        self.streamed: list[str] = []
        self.secondary_streamed: list[str] = []
        self.reasoning_streamed: list[str] = []
        self.stream_delta_callback = self.streamed.append
        self._stream_callback = self.secondary_streamed.append
        self.reasoning_callback = self.reasoning_streamed.append

    def _build_api_kwargs(self, messages, max_tokens=None):
        return {"messages": messages, "max_tokens": max_tokens}


def test_summary_runtime_suppresses_main_stream_callbacks(monkeypatch):
    """Append-cached summarizer deltas are internal and must not reach chat streams."""
    agent = FakeSummaryAgent()

    def fake_interruptible_api_call(agent_arg, api_kwargs):
        # If make_summary_runtime does not suppress callbacks, this emulates the
        # Codex Responses streaming path leaking the summary body to Discord.
        if agent_arg.stream_delta_callback is not None:
            agent_arg.stream_delta_callback("SUMMARY_STREAM_LEAK")
        if getattr(agent_arg, "_stream_callback", None) is not None:
            agent_arg._stream_callback("SUMMARY_STREAM_LEAK")
        if getattr(agent_arg, "reasoning_callback", None) is not None:
            agent_arg.reasoning_callback("SUMMARY_STREAM_LEAK")
        return SimpleNamespace(output_text="internal summary", output=[], usage=None)

    monkeypatch.setattr(
        "agent.chat_completion_helpers.interruptible_api_call",
        fake_interruptible_api_call,
    )

    original_stream_delta_callback = agent.stream_delta_callback
    original_stream_callback = agent._stream_callback
    original_reasoning_callback = agent.reasoning_callback

    runtime = make_summary_runtime(agent)
    response = runtime.invoke({"messages": []})

    assert response.output_text == "internal summary"
    assert agent.streamed == []
    assert agent.secondary_streamed == []
    assert agent.reasoning_streamed == []
    assert agent.stream_delta_callback == original_stream_delta_callback
    assert agent._stream_callback == original_stream_callback
    assert agent.reasoning_callback == original_reasoning_callback


def test_extract_summary_cache_stats_reports_provider_actual_denominator():
    response = SimpleNamespace(
        usage={
            "input_tokens": 107_010,
            "output_tokens": 3_000,
            "input_tokens_details": {"cached_tokens": 86_400},
        }
    )

    stats = extract_summary_cache_stats(response)

    assert stats["reported"] is True
    assert stats["read_tokens"] == 86_400
    assert stats["provider_input_tokens"] == 107_010
    assert stats["provider_output_tokens"] == 3_000
    assert stats["hit_rate_provider_actual"] == 0.8074


def test_extract_summary_cache_stats_does_not_infer_actual_hit_rate_without_input_tokens():
    response = SimpleNamespace(
        usage={
            "input_tokens_details": {"cached_tokens": 0},
        }
    )

    stats = extract_summary_cache_stats(response)

    assert stats["reported"] is True
    assert stats["read_tokens"] == 0
    assert stats["provider_input_tokens"] is None
    assert stats["hit_rate_provider_actual"] is None
