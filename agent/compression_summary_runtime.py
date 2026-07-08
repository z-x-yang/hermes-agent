from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SummaryRuntime:
    """Lightweight bridge from the compressor to the active main-model runtime."""

    provider: str
    model: str
    api_mode: str
    base_url: str
    reasoning_effort: str | None
    context_limit_tokens: int | None
    tools_included: bool
    build_kwargs: Callable[[list[dict[str, Any]], int], dict[str, Any]]
    invoke: Callable[[dict[str, Any]], Any]
    estimate_request_tokens: Callable[[dict[str, Any]], int]


def make_summary_runtime(agent: Any) -> SummaryRuntime:
    """Return a main-runtime bridge for append-cached compression summary calls."""
    from agent.chat_completion_helpers import (
        estimate_request_context_tokens,
        interruptible_api_call,
    )

    def _build_kwargs(messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        old_ephemeral = getattr(agent, "_ephemeral_max_output_tokens", None)
        try:
            agent._ephemeral_max_output_tokens = max_tokens
            return agent._build_api_kwargs(messages)
        finally:
            agent._ephemeral_max_output_tokens = old_ephemeral

    def _invoke(api_kwargs: dict[str, Any]) -> Any:
        return interruptible_api_call(agent, api_kwargs)

    return SummaryRuntime(
        provider=getattr(agent, "provider", "") or "",
        model=getattr(agent, "model", "") or "",
        api_mode=getattr(agent, "api_mode", "") or "",
        base_url=getattr(agent, "base_url", "") or "",
        reasoning_effort=getattr(agent, "reasoning_effort", None),
        context_limit_tokens=getattr(getattr(agent, "context_compressor", None), "context_length", None),
        tools_included=bool(getattr(agent, "tools", None)),
        build_kwargs=_build_kwargs,
        invoke=_invoke,
        estimate_request_tokens=estimate_request_context_tokens,
    )


def apply_summary_tool_choice_none(
    api_kwargs: dict[str, Any],
    api_mode: str,
) -> tuple[dict[str, Any], bool]:
    """Return kwargs with provider-appropriate no-tool choice when tools are present."""
    if "tools" not in api_kwargs:
        return api_kwargs, False
    if api_mode == "codex_responses":
        # Codex Responses already emits ``tool_choice: auto`` when tools are
        # present. Keep the main-runtime payload shape unchanged for cache reuse
        # and rely on the summary instruction + post-response tool-call
        # validation instead of sending a provider-rejected no-tool override.
        return api_kwargs, False

    updated = dict(api_kwargs)
    if api_mode == "anthropic_messages":
        updated["tool_choice"] = {"type": "none"}
    else:
        updated["tool_choice"] = "none"
    return updated, True


def extract_summary_response_content(response: Any) -> tuple[str, bool]:
    """Return response text and whether the model attempted a tool call."""
    def _get(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    if part:
                        parts.append(part)
                    continue
                part_type = str(_get(part, "type", "") or "").strip().lower()
                if part_type not in {"output_text", "text"}:
                    continue
                text = _get(part, "text", "")
                if not isinstance(text, str):
                    text = str(text or "")
                if text:
                    parts.append(text)
            return "\n".join(parts).strip()
        return str(content) if content else ""

    choices = getattr(response, "choices", None) or []
    if not choices:
        output = _get(response, "output") or []
        content_parts: list[str] = []
        if isinstance(output, list):
            for item in output:
                item_type = str(_get(item, "type", "") or "").strip().lower()
                if item_type in {"function_call", "custom_tool_call"}:
                    return "", True
                if item_type != "message":
                    continue
                text = _content_text(_get(item, "content", ""))
                if text:
                    content_parts.append(text)
        if content_parts:
            return "\n".join(content_parts).strip(), False
        output_text = _get(response, "output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip(), False
        return "", False
    message = getattr(choices[0], "message", None)
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")
    else:
        tool_calls = getattr(message, "tool_calls", None) or []
        content = getattr(message, "content", message)
    if tool_calls:
        return "", True
    return _content_text(content), False


def extract_summary_cache_stats(response: Any) -> dict[str, Any]:
    """Normalize provider cache usage fields for compression audit records."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {"reported": False, "read_tokens": None, "write_tokens": None, "hit_rate_estimate": None}

    def _get(obj: Any, name: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def _has(obj: Any, name: str) -> bool:
        if obj is None:
            return False
        if isinstance(obj, dict):
            return name in obj
        return hasattr(obj, name)

    def _first_present(*paths: tuple[Any, str]) -> tuple[Any, bool]:
        for obj, name in paths:
            if _has(obj, name):
                value = _get(obj, name)
                if value is not None:
                    return value, True
        return None, False

    def _int_or_none(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details")
    read_tokens, read_reported = _first_present(
        (details, "cached_tokens"),
        (details, "cache_read_input_tokens"),
        (usage, "cache_read_tokens"),
        (usage, "cache_read_input_tokens"),
    )
    write_tokens, write_reported = _first_present(
        (details, "cache_write_tokens"),
        (details, "cache_creation_tokens"),
        (usage, "cache_write_tokens"),
        (usage, "cache_creation_input_tokens"),
        (usage, "cache_creation_tokens"),
    )
    prompt_tokens, _prompt_reported = _first_present(
        (usage, "prompt_tokens"),
        (usage, "input_tokens"),
    )
    read_int = _int_or_none(read_tokens) if read_reported else None
    write_int = _int_or_none(write_tokens) if write_reported else None
    prompt_int = _int_or_none(prompt_tokens)
    hit_rate = None
    if read_int is not None and prompt_int:
        hit_rate = round(read_int / max(prompt_int, 1), 4)
    elif read_int == 0 and prompt_int is not None:
        hit_rate = 0.0
    return {
        "reported": read_int is not None or write_int is not None,
        "read_tokens": read_int,
        "write_tokens": write_int,
        "hit_rate_estimate": hit_rate,
    }
