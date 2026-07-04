"""Regression coverage for resumed sessions with inherited handoffs.

A compacted handoff can sit in the protected head after a lineage resumes. The
new contract deliberately makes the handoff working context, not reference-only
history, while still ensuring later user messages are newer and take precedence
on conflict. Older reference-only/resume-exactly handoffs must be detected and
renormalized so stale directives do not survive forever.
"""

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    LEGACY_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
    ContextCompressor,
)


# The conflicting prefix that shipped before the #35344 fix. A handoff
# persisted in a resumed lineage could carry this verbatim.
_OLD_CONFLICTING_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is identified in the '## Active Task' section of the "
    "summary — resume exactly from there. "
    "Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:"
)


def test_later_user_messages_precede_inherited_handoff_on_conflict():
    """The current prefix keeps active-work continuation but still privileges
    fresh user deltas over inherited summarized state on conflict."""
    lower = SUMMARY_PREFIX.lower()
    assert "working context" in lower
    assert "current work" in lower
    assert "pending tasks" in lower
    assert "later user messages" in lower
    assert "newer than the summary" in lower
    assert "take precedence on conflict" in lower
    assert "do not revive completed or cancelled work" in lower


def test_no_resume_exactly_or_reference_only_directive_can_hijack():
    """The old hijack directive and old reference-only framing must be gone
    from the live prefix."""
    lower = SUMMARY_PREFIX.lower()
    assert "resume exactly" not in lower
    assert "reference only" not in lower
    assert "respond only to the latest user message" not in lower
    assert "active task" not in lower


def test_resumed_stale_handoff_gets_renormalized_to_current_prefix():
    """A handoff persisted under the OLD conflicting prefix is upgraded to
    the current working-context prefix when re-normalized on re-compaction.
    """
    stale_body = (
        f"{HISTORICAL_TASK_HEADING}\n"
        "User asked: 'Migrate the billing module to Stripe'\n\n"
        "## Goal\nMigrate billing.\n"
    )
    stale_handoff = f"{_OLD_CONFLICTING_PREFIX}\n{stale_body}"

    # Sanity: the fixture really does carry the old directive.
    assert "resume exactly" in stale_handoff.lower()

    renormalized = ContextCompressor._with_summary_prefix(stale_handoff)

    # The body is preserved...
    assert "Migrate the billing module to Stripe" in renormalized
    # ...but the conflicting directive is stripped and replaced with the
    # current working-context framing.
    assert "resume exactly" not in renormalized.lower()
    assert "reference only" not in renormalized.lower()
    assert renormalized.startswith(SUMMARY_PREFIX)
    assert "working context" in renormalized.lower()
    assert "take precedence on conflict" in renormalized.lower()


def test_legacy_prefix_handoff_also_renormalized():
    """The same upgrade applies to the oldest ``[CONTEXT SUMMARY]:`` handoff
    format that may sit in a long-lived resumed lineage."""
    legacy = f"{LEGACY_SUMMARY_PREFIX} {HISTORICAL_TASK_HEADING}\nUser asked: 'task A'"
    renormalized = ContextCompressor._with_summary_prefix(legacy)
    assert renormalized.startswith(SUMMARY_PREFIX)
    assert LEGACY_SUMMARY_PREFIX not in renormalized
    assert "task A" in renormalized


def test_inherited_handoff_detected_in_resumed_protected_head():
    """On a resumed lineage the handoff commonly sits right after the system
    prompt. ``_find_latest_context_summary`` must detect it there so
    re-compaction rehydrates state from it rather than serializing it as a
    fresh user turn."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n## Current Work\nContinue task A."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Unrelated task B: what's the capital of France?"},
    ]
    # Search the whole post-system range.
    idx, body = ContextCompressor._find_latest_context_summary(
        messages, 1, len(messages)
    )
    assert idx == 1, "handoff in protected head must be found"
    assert "Continue task A" in body
    # The detected body is stripped of the prefix (treated as state, not a
    # standalone instruction message).
    assert not body.startswith(SUMMARY_PREFIX)


def test_historical_prefixed_handoff_detected_and_stripped():
    """A pre-fix handoff inherited into a resumed lineage must still be
    recognized as a context summary AND have its old directive stripped on
    detection."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{_OLD_CONFLICTING_PREFIX}\n{HISTORICAL_TASK_HEADING}\nUser asked: 'task A'"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Unrelated task B"},
    ]
    idx, body = ContextCompressor._find_latest_context_summary(
        messages, 1, len(messages)
    )
    assert idx == 1
    assert "task A" in body
    assert "resume exactly" not in body.lower()
