"""Gateway response filtering helpers.

These helpers operate at the gateway boundary: they decide whether a completed
agent turn should be delivered to the chat, not what should be persisted in the
conversation history.
"""

from __future__ import annotations

from typing import Any

# Canonical model-emitted control token for intentional silence.
SILENT_REPLY_TOKEN = "NO_REPLY"

# Exact whole-response markers that mean "the agent intentionally chose not to
# reply".  Keep this list small and explicit; arbitrary empty output remains an
# error/empty-response path, not silence.
LIVE_GATEWAY_SILENT_MARKERS = frozenset({
    "[SILENT]",
    "SILENT",
    "NO_REPLY",
    "NO REPLY",
})

_CONTEXT_COMPACTION_BODY_MARKERS = (
    "## Primary Request and Intent",
    "## All User Messages",
    "## Current Work",
)


def _canonical_silence_candidate(text: str) -> str:
    return " ".join(text.strip().upper().split())


def is_intentional_silence_response(response: Any) -> bool:
    """Return True only when ``response`` is exactly a silence marker.

    Substantive prose that merely mentions ``NO_REPLY`` or ``[SILENT]`` must be
    delivered normally.  A blank response is also not silence; blank output is
    handled by the empty-response failure path.
    """
    if not isinstance(response, str):
        return False
    stripped = response.strip()
    if not stripped:
        return False
    if len(stripped) > 64:
        return False
    return _canonical_silence_candidate(stripped) in LIVE_GATEWAY_SILENT_MARKERS


def is_intentional_silence_agent_result(agent_result: dict | None, response: Any) -> bool:
    """Silence markers suppress delivery only for successful agent turns."""
    if not isinstance(agent_result, dict):
        return False
    if agent_result.get("failed"):
        return False
    return is_intentional_silence_response(response)


def is_context_compaction_response(response: Any) -> bool:
    """Return True for internal context-compaction summaries.

    These summaries are provider-visible recovery state, not a user-facing
    assistant reply.  A live Discord incident showed that a compacted summary
    can enter the normal streaming/final-delivery path either with the banner or
    after a boundary stripped it, so detect both the explicit prefix and the
    canonical nine-section body shape.
    """
    if not isinstance(response, str):
        return False
    text = response.strip()
    if not text:
        return False
    if text.startswith("[CONTEXT COMPACTION"):
        return True
    if "--- END OF COMPACTED CONTEXT ---" not in text:
        return False
    head = text[:4096]
    return all(marker in head or marker in text for marker in _CONTEXT_COMPACTION_BODY_MARKERS)


def is_partial_context_compaction_response(response: Any) -> bool:
    """Return True while a stream buffer still looks like a compaction summary.

    Streaming paths may see only the leading ``## Primary Request and Intent``
    body before the end marker arrives. Hold those buffers until completion, at
    which point :func:`is_context_compaction_response` either suppresses the
    internal summary or ordinary prose flushes normally.
    """
    if not isinstance(response, str):
        return False
    text = response.lstrip()
    if not text:
        return False
    if text.startswith("[CONTEXT COMPACTION"):
        return True
    return text.startswith("## Primary Request and Intent")


def is_partial_silence_marker(text: Any) -> bool:
    """Return True while ``text`` could still resolve to a silence marker.

    The streaming path accumulates the reply delta-by-delta and must decide,
    before the whole response is known, whether to show what it has so far.
    A buffer whose canonical form is a non-empty *prefix* of a silence marker
    (e.g. ``"NO"`` on the way to ``"NO_REPLY"``, or an exact marker that has
    not yet been terminated by stream-end) is held back so a raw marker is
    never edited onto the screen and then belatedly retracted.

    Anything that has already diverged from every marker (ordinary prose) —
    and anything longer than the marker cap — returns False so normal
    streaming resumes immediately.  This is the streaming counterpart to
    :func:`is_intentional_silence_response`, sharing the same marker set and
    canonicalization so the two never drift.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > 64:
        return False
    candidate = _canonical_silence_candidate(stripped)
    if not candidate:
        return False
    return any(marker.startswith(candidate) for marker in LIVE_GATEWAY_SILENT_MARKERS)
