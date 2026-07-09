from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SLASH_COMMANDS = REPO_ROOT / "gateway" / "slash_commands.py"


def test_manual_compress_binds_append_cached_summary_to_full_request_shape_agent():
    """Manual /compress must not summarize through the memory-only temp agent.

    The append-cached summary call is only cache-friendly when its provider-visible
    system/tools/messages prefix matches a normal main request. The temporary
    compression agent is intentionally side-effect-light and memory-only, so the
    slash handler must bind its ContextCompressor summary runtime to the full
    request-shape agent used for platform prompt/tool accounting.
    """
    src = SLASH_COMMANDS.read_text()
    assert "make_summary_runtime(request_estimate_agent)" in src
    assert "bind_summary_runtime_factory" in src
    assert "summary_runtime_shape" in src
    assert "full_toolset_internal" in src
    assert "compress-estimate" not in src


def test_manual_compress_full_request_shape_agent_inherits_gateway_fallback_model():
    """Append-cached retries must stay append-cached when the primary route 429s."""
    src = SLASH_COMMANDS.read_text()
    request_kwargs_idx = src.index("request_estimate_kwargs = {")
    request_agent_idx = src.index("request_estimate_agent = AIAgent", request_kwargs_idx)
    block = src[request_kwargs_idx:request_agent_idx]
    assert 'request_estimate_kwargs["fallback_model"] = _fallback_model' in block


def test_manual_compress_keeps_internal_summary_off_normal_delivery_path():
    """The full-toolset fix must still avoid the ordinary chat delivery path."""
    src = SLASH_COMMANDS.read_text()
    bind_idx = src.index("bind_summary_runtime_factory") if "bind_summary_runtime_factory" in src else -1
    compress_idx = src.index("tmp_agent._compress_context")
    assert bind_idx != -1 and bind_idx < compress_idx
    nearby = src[max(0, bind_idx - 500): compress_idx + 500]
    assert "run_conversation" not in nearby
    assert "make_summary_runtime(request_estimate_agent)" in nearby
