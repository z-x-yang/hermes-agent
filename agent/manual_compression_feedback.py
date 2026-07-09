"""User-facing summaries for manual compression commands."""

from __future__ import annotations

from typing import Any, Sequence


def materialize_manual_compression_system_prompt(
    agent: Any,
    system_message: str | None = None,
) -> str:
    """Return the system prompt used for manual-compression token estimates.

    Manual compression paths create or reuse agents before making a real model
    request.  At that point ``_cached_system_prompt`` may still be empty even
    though the subsequent compression call rebuilds it.  Token feedback must not
    compare a pre-compression request estimated with an empty prompt against a
    post-compression request estimated with the rebuilt prompt.
    """
    prompt = getattr(agent, "_cached_system_prompt", "") or ""
    if prompt:
        return prompt

    build_prompt = getattr(agent, "_build_system_prompt", None)
    if not callable(build_prompt):
        return ""

    prompt = build_prompt(system_message) or ""
    if prompt:
        try:
            agent._cached_system_prompt = prompt
        except Exception:
            pass
    return prompt


def estimate_manual_compression_request_tokens(
    agent: Any,
    messages: Sequence[dict[str, Any]],
    *,
    system_prompt: str = "",
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Estimate /compress request size using provider-visible accounting.

    The user-facing number should describe the payload that can be replayed to
    the provider, not raw DB/storage fields. Prefer the bound compressor's
    provider-visible estimator so Codex Responses replay items, chat-completions
    sanitization, and storage-only metadata match automatic compression audit
    accounting.
    """
    compressor = getattr(agent, "context_compressor", None)
    estimator = getattr(compressor, "estimate_provider_request_tokens", None)
    if callable(estimator):
        value = estimator(
            list(messages),
            system_prompt=system_prompt,
            tools=tools,
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)

    from agent.model_metadata import estimate_request_tokens_rough

    return int(
        estimate_request_tokens_rough(
            list(messages),
            system_prompt=system_prompt,
            tools=tools,
        )
    )


def summarize_manual_compression(
    before_messages: Sequence[dict[str, Any]],
    after_messages: Sequence[dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
) -> dict[str, Any]:
    """Return consistent user-facing feedback for manual compression."""
    before_count = len(before_messages)
    after_count = len(after_messages)
    noop = list(after_messages) == list(before_messages)

    if noop:
        headline = f"No changes from compression: {before_count} messages"
        if after_tokens == before_tokens:
            token_line = (
                f"Approx request size: ~{before_tokens:,} tokens (unchanged)"
            )
        else:
            token_line = (
                f"Approx request size: ~{before_tokens:,} → "
                f"~{after_tokens:,} tokens"
            )
    else:
        headline = f"Compressed: {before_count} → {after_count} messages"
        token_line = (
            f"Approx request size: ~{before_tokens:,} → "
            f"~{after_tokens:,} tokens"
        )

    note = None
    if not noop and after_count < before_count and after_tokens > before_tokens:
        note = (
            "Note: fewer messages can still raise this estimate when "
            "compression rewrites the transcript into denser summaries."
        )

    return {
        "noop": noop,
        "headline": headline,
        "token_line": token_line,
        "note": note,
    }
