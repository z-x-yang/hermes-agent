"""Shared middleware and backend path for one provider attempt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class PreparedProviderAttempt:
    payload: dict[str, Any]
    original_payload: dict[str, Any]
    middleware_trace: tuple[Any, ...]


def prepare_provider_attempt(
    agent,
    api_kwargs: dict[str, Any],
    *,
    task_id: str,
    turn_id: str,
    api_request_id: str,
    api_call_count: int,
) -> PreparedProviderAttempt:
    """Apply request middleware with the ordinary no-middleware fallback."""
    try:
        from hermes_cli.middleware import apply_llm_request_middleware

        middleware_result = apply_llm_request_middleware(
            api_kwargs,
            task_id=task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            session_id=agent.session_id or "",
            platform=agent.platform or "",
            model=agent.model,
            provider=agent.provider,
            base_url=agent.base_url,
            api_mode=agent.api_mode,
            api_call_count=api_call_count,
        )
        return PreparedProviderAttempt(
            payload=middleware_result.payload,
            original_payload=middleware_result.original_payload,
            middleware_trace=tuple(middleware_result.trace),
        )
    except Exception:
        return PreparedProviderAttempt(
            payload=dict(api_kwargs),
            original_payload=dict(api_kwargs),
            middleware_trace=(),
        )


def execute_provider_attempt(
    agent,
    prepared: PreparedProviderAttempt,
    perform_backend: Callable[[dict[str, Any]], Any],
    *,
    task_id: str,
    turn_id: str,
    api_request_id: str,
    api_call_count: int,
    pre_api_observer: Callable[[dict[str, Any], tuple[Any, ...]], None] | None = None,
    final_payload_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
):
    """Run execution middleware, then observe and execute its final payload."""

    def _terminal(final_payload: dict[str, Any]):
        if final_payload_transform is not None:
            final_payload = final_payload_transform(final_payload)
        if pre_api_observer is not None:
            pre_api_observer(final_payload, prepared.middleware_trace)
        return perform_backend(final_payload)

    from hermes_cli.middleware import run_llm_execution_middleware

    return run_llm_execution_middleware(
        prepared.payload,
        _terminal,
        original_request=prepared.original_payload,
        task_id=task_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        session_id=agent.session_id or "",
        platform=agent.platform or "",
        model=agent.model,
        provider=agent.provider,
        base_url=agent.base_url,
        api_mode=agent.api_mode,
        api_call_count=api_call_count,
        middleware_trace=list(prepared.middleware_trace),
    )
