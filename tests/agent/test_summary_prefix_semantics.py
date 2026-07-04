"""Pin the working-context semantics of SUMMARY_PREFIX.

The prefix is intentionally short: it explains that the compacted summary is
working context, not a fresh user request; active work may continue from the
nine-section checkpoint; later user messages are newer and win on conflict.
The detailed handoff contract belongs in the nine-section summary body, not in
an edge-case rule list in the prefix.
"""

from agent.context_compressor import (
    SUMMARY_PREFIX,
    ContextCompressor,
)


def test_prefix_is_short_working_context_contract():
    lines = SUMMARY_PREFIX.splitlines()
    assert len(lines) == 3
    assert len(SUMMARY_PREFIX) < 500
    lower = SUMMARY_PREFIX.lower()
    assert "working context" in lower
    assert "not as a new user request" in lower
    assert "current work" in lower
    assert "pending tasks" in lower


def test_later_user_messages_take_precedence_on_conflict():
    lower = SUMMARY_PREFIX.lower()
    assert "later user messages" in lower
    assert "newer than the summary" in lower
    assert "take precedence on conflict" in lower
    assert "changes, narrows, cancels, or replaces" in lower


def test_prefix_does_not_reintroduce_reference_only_or_latest_user_only_contract():
    lower = SUMMARY_PREFIX.lower()
    forbidden = [
        "reference only",
        "background reference",
        "not as active instructions",
        "respond only to the latest user message",
        "single source of truth",
        "resume exactly",
        "active task",
        "topic overlap",
    ]
    for phrase in forbidden:
        assert phrase not in lower


def test_prefix_does_not_expand_into_reverse_signal_rule_list():
    """Reverse-signal examples made the old prefix brittle and long.

    The new contract keeps this generic (changes/narrows/cancels/replaces)
    and lets the model infer specific stop/undo/rollback cases.
    """
    lower = SUMMARY_PREFIX.lower()
    reverse_terms = ["stop", "undo", "roll back", "never mind", "just verify"]
    assert not any(term in lower for term in reverse_terms)


def test_replaced_prefixes_are_frozen_for_renormalization():
    """Every retired SUMMARY_PREFIX must be frozen into
    _HISTORICAL_SUMMARY_PREFIXES, otherwise summaries persisted by older
    builds lose detection/renormalization after an upgrade.
    """
    from agent.context_compressor import _HISTORICAL_SUMMARY_PREFIXES

    assert SUMMARY_PREFIX not in _HISTORICAL_SUMMARY_PREFIXES
    assert any("single source of truth" in p for p in _HISTORICAL_SUMMARY_PREFIXES)
    assert any("you may use the summary as background" in p for p in _HISTORICAL_SUMMARY_PREFIXES)
    assert any("resume exactly" in p for p in _HISTORICAL_SUMMARY_PREFIXES)

    # Detection + strip must work for every frozen prefix.
    for old_prefix in _HISTORICAL_SUMMARY_PREFIXES:
        content = old_prefix + "\n## Summary body"
        assert ContextCompressor._is_context_summary_content(content)
        stripped = ContextCompressor._strip_summary_prefix(content)
        assert not stripped.startswith(old_prefix)
        assert stripped == "## Summary body"
