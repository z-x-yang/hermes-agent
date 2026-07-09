"""Regression tests for issue #27405.

The preflight compression gate must trigger when *either* the message
count exceeds the protected ranges OR the cheap char-based token
estimate already crosses the configured threshold. Pre-fix, only the
message-count condition was checked, so a session with a small number
of huge messages would silently skip compression and eventually hit a
hard context-overflow error.
"""

from types import SimpleNamespace

from agent.turn_context import build_turn_context, _should_run_preflight_estimate


# Protected-range counts mirror the compressor defaults. THRESHOLD_TOKENS is an
# arbitrary test threshold passed explicitly into the helper — it is NOT the
# live runtime threshold (which is max(0.5*window, MINIMUM_CONTEXT_LENGTH) per
# model); the helper takes the threshold as a parameter so the tests are
# self-contained and independent of model metadata.
PROTECT_FIRST_N = 3
PROTECT_LAST_N = 20
THRESHOLD_TOKENS = 64_000


def _msg(content: str) -> dict:
    return {"role": "user", "content": content}


def test_few_messages_huge_content_triggers_gate():
    """The bug from #27405: 8 messages with one massive content blob."""
    # ~280K chars in one message ~= 70K tokens at 4 chars/token.
    big = "x" * 280_000
    messages = [_msg("hi")] * 7 + [_msg(big)]
    assert len(messages) <= PROTECT_FIRST_N + PROTECT_LAST_N + 1  # would fail old gate
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is True


def test_few_messages_small_content_does_not_trigger():
    """Regression guard: tiny sessions should not pay the estimator cost."""
    messages = [_msg("hello world")] * 8
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is False


def test_many_small_messages_still_triggers_via_count():
    """The historical path: > protect_first + protect_last + 1 messages."""
    messages = [_msg("ok")] * (PROTECT_FIRST_N + PROTECT_LAST_N + 2)  # 25
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is True


def test_content_above_threshold_triggers():
    """A single message comfortably above the threshold trips branch (b)."""
    # ~threshold*4 chars => ~threshold tokens; +1000 tokens of margin so the
    # test doesn't depend on per-message dict-wrapping overhead in the
    # shared estimator's (chars+3)//4 rounding.
    messages = [_msg("x" * ((THRESHOLD_TOKENS + 1000) * 4))]
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is True


def test_content_below_threshold_does_not_trigger():
    """A single message comfortably below the threshold (and few messages)
    must not trigger — the estimator stays under and the count gate is not
    tripped."""
    messages = [_msg("x" * ((THRESHOLD_TOKENS - 1000) * 4))]
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is False


def test_message_with_none_content_is_treated_as_empty():
    """Assistant turns mid-tool-call carry content=None -- must not crash."""
    messages = [{"role": "assistant", "content": None}] * 5
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is False


def test_message_with_list_content_counts_text_parts():
    """Multimodal content lists: the shared estimator digs into text parts.

    estimate_messages_tokens_rough walks list content (rather than str()-ing
    the whole list), so a huge text part is counted by its real length and an
    image part is counted at a flat per-image cost — not its base64 length.
    """
    parts = [{"type": "text", "text": "x" * 300_000}]
    messages = [{"role": "user", "content": parts}]
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is True


def test_large_base64_image_does_not_falsely_trip_gate():
    """Regression for the inline-estimator bug: a single ~1MB base64 image
    must NOT be mistaken for ~250K tokens. The shared estimator counts images
    at a flat per-image cost, so one screenshot in a tiny session stays below
    the threshold and the gate does not fire on content size alone.
    """
    big_b64 = "A" * 1_000_000  # ~1MB base64 payload
    parts = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_b64}"}}]
    messages = [{"role": "user", "content": parts}]
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is False


class _FakePreflightCompressor:
    threshold_tokens = 85_000
    protect_first_n = 3
    protect_last_n = 20
    last_prompt_tokens = 0
    last_real_prompt_tokens = 50_000
    compression_count = 0
    context_length = 272_000

    def __init__(self):
        self.defer_fingerprint = ""

    def estimate_provider_request_tokens(self, messages, *, system_prompt="", tools=None):
        return 93_000

    def preflight_request_fingerprint(self, *, system_prompt="", tools=None):
        return "route-a"

    def should_defer_preflight_to_real_usage(self, rough_tokens, *, fingerprint=""):
        self.defer_fingerprint = fingerprint
        return rough_tokens == 93_000 and fingerprint == "route-a"

    def get_active_compression_failure_cooldown(self):
        return None

    def should_compress(self, prompt_tokens=None):
        return (prompt_tokens or 0) >= self.threshold_tokens


def _fake_agent(compressor):
    return SimpleNamespace(
        session_id="sess",
        _memory_write_origin="assistant_tool",
        _restore_primary_runtime=lambda: None,
        provider="gptcodex",
        model="gpt-5.5",
        base_url="https://example.invalid",
        api_key="",
        api_mode="codex_responses",
        _skip_mcp_refresh=True,
        _stream_callback=None,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _current_task_id="",
        _current_turn_id="",
        _current_api_request_id="",
        _tool_guardrails=SimpleNamespace(reset_for_turn=lambda: None),
        _memory_store=SimpleNamespace(reset_consolidation_failures=lambda: None),
        _cleanup_dead_connections=lambda: False,
        _emit_status=lambda _message: None,
        _compression_warning=None,
        _replay_compression_warning=lambda: None,
        max_iterations=3,
        platform="test",
        _todo_store=SimpleNamespace(has_items=lambda: True),
        _hydrate_todo_store=lambda _history: None,
        _user_turn_count=1,
        _memory_nudge_interval=0,
        _turns_since_memory=0,
        valid_tool_names=set(),
        _stream_context_scrubber=None,
        _stream_think_scrubber=None,
        quiet_mode=True,
        _safe_print=lambda _text: None,
        _cached_system_prompt="system",
        _ensure_db_session=lambda: None,
        _persist_session=lambda _messages, _history: None,
        compression_enabled=True,
        context_compressor=compressor,
        tools=[{"type": "function", "function": {"name": "demo"}}],
        _runtime_context_status_mode="off",
        _last_context_pressure_notice_compression_count=None,
        _empty_content_retries=0,
        _invalid_tool_retries=0,
        _invalid_json_retries=0,
        _incomplete_scratchpad_retries=0,
        _codex_incomplete_retries=0,
        _thinking_prefill_retries=0,
        _post_tool_empty_retried=False,
        _last_content_with_tools=None,
        _last_content_tools_all_housekeeping=False,
        _mute_post_response=False,
        _unicode_sanitization_passes=0,
        _tool_guardrail_halt_decision=None,
        _turn_failed_file_mutations={},
        _turn_file_mutation_paths=set(),
        _verification_stop_nudges=0,
        _pre_verify_nudges=0,
        _execution_thread_id=None,
        _interrupt_requested=False,
        _interrupt_thread_signal_pending=False,
        _interrupt_message=None,
        _memory_manager=None,
        _user_id="user",
        _compress_context=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("preflight compression should have been deferred")
        ),
    )


def test_build_turn_context_passes_preflight_fingerprint_to_defer_gate():
    compressor = _FakePreflightCompressor()
    agent = _fake_agent(compressor)

    ctx = build_turn_context(
        agent,
        "x" * 380_000,
        None,
        [],
        None,
        None,
        None,
        restore_or_build_system_prompt=lambda *_args, **_kwargs: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s[:20],
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _origin: None,
        ra=lambda: SimpleNamespace(_set_interrupt=lambda *_args, **_kwargs: None),
    )

    assert compressor.defer_fingerprint == "route-a"
    assert ctx.active_system_prompt == "system"
