"""Tests for agent/efficiency_review.py — the deterministic efficiency-review
support for the background skill review.

Design under test (division of labour): scripts do the deterministic part
(digest detection, sightings ledger, escalation), the review model does the
judgment (whether a flagged repeat is real waste, what rule to write).

Key behaviours locked here:
  • Path-aggregated re-read detection: reading the same file in ≥3 segments
    (different offsets) is flagged — tool-level mtime dedup never fires for
    those, and "walk a source file in chunks to learn CLI flags" was the
    measured #1 avoidable waste in the email-triage cron.
  • Mutation-aware exemption: re-reads of a file the session itself modified
    (write_file/patch between reads) are verification, not waste — exempt.
  • The gate: a clean session produces NO digest → prompt carries zero
    efficiency text.
  • Ledger lifecycle: observed (1st sighting) → encode_now (≥2 distinct
    sessions) → recurred_after_encode (sighted again after a rule landed),
    with same-session repeats counted once.
"""

import json

import pytest

from agent import efficiency_review as er


# ---------------------------------------------------------------------------
# Message-snapshot builders (OpenAI-style dicts, as stored in _session_messages)
# ---------------------------------------------------------------------------

_ID = 0


def _call(name, args):
    global _ID
    _ID += 1
    return {
        "id": f"call_{_ID}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _assistant(*tool_calls):
    return {"role": "assistant", "content": "", "tool_calls": list(tool_calls)}


def _result(tool_call_id, content="ok"):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _msgs_with_results(*assistant_msgs):
    """Interleave assistant tool-call messages with matching tool results."""
    out = []
    for m in assistant_msgs:
        out.append(m)
        for tc in m.get("tool_calls", []):
            out.append(_result(tc["id"]))
    return out


RUNNER = "scripts/email_triage_runner.mjs"


def _segmented_read_snapshot(n_reads=3, path=RUNNER, interleave=None):
    """n_reads of the same path at different offsets; optionally interleave a
    mutating call between the first and second read."""
    msgs = []
    for i in range(n_reads):
        msgs.append(_assistant(_call("read_file", {"path": path, "offset": 1 + 300 * i, "limit": 220})))
        if i == 0 and interleave is not None:
            msgs.append(interleave)
    return _msgs_with_results(*msgs)


# ---------------------------------------------------------------------------
# Digest: detection
# ---------------------------------------------------------------------------

def test_segmented_reread_is_flagged():
    digest = er.build_tool_call_digest(_segmented_read_snapshot(3))
    assert digest is not None
    keys = [p["key"] for p in digest["patterns"]]
    assert f"reread:{RUNNER}" in keys
    text = "\n".join(digest["lines"])
    assert RUNNER in text
    assert "3" in text  # segment count surfaced


def test_two_reads_below_threshold_not_flagged():
    assert er.build_tool_call_digest(_segmented_read_snapshot(2)) is None


def test_reads_of_different_paths_not_flagged():
    msgs = _msgs_with_results(
        _assistant(_call("read_file", {"path": "a.py", "limit": 100})),
        _assistant(_call("read_file", {"path": "b.py", "limit": 100})),
        _assistant(_call("read_file", {"path": "c.py", "limit": 100})),
    )
    assert er.build_tool_call_digest(msgs) is None


def test_identical_args_duplicate_call_is_flagged():
    call_args = {"command": "node scripts/email_triage_run_artifacts.mjs summarize"}
    msgs = _msgs_with_results(
        _assistant(_call("terminal", call_args)),
        _assistant(_call("terminal", call_args)),
    )
    digest = er.build_tool_call_digest(msgs)
    assert digest is not None
    assert any(p["key"].startswith("dup:terminal:") for p in digest["patterns"])


def test_write_between_reads_exempts_the_path():
    """read → patch same file → re-read is verification, not waste."""
    mutate = _assistant(_call("patch", {"path": RUNNER, "old_string": "a", "new_string": "b"}))
    msgs = _segmented_read_snapshot(3, interleave=mutate)
    digest = er.build_tool_call_digest(msgs)
    assert digest is None or f"reread:{RUNNER}" not in [p["key"] for p in digest["patterns"]]


def test_clean_session_gates_to_none():
    msgs = _msgs_with_results(
        _assistant(_call("terminal", {"command": "node runner.mjs --mode cron"})),
        _assistant(_call("read_file", {"path": "review_packet.json", "limit": 500})),
    )
    assert er.build_tool_call_digest(msgs) is None


def test_digest_includes_largest_results():
    msgs = _segmented_read_snapshot(3)
    # inflate one tool result so the size ranking has something to show
    msgs[1]["content"] = "x" * 50_000
    digest = er.build_tool_call_digest(msgs)
    text = "\n".join(digest["lines"])
    assert "chars" in text


# ---------------------------------------------------------------------------
# Ledger lifecycle
# ---------------------------------------------------------------------------

@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    path = tmp_path / ".efficiency_observations.jsonl"
    monkeypatch.setattr(er, "ledger_path", lambda: path)
    return path


def _classify(keys, session, family="cron_test"):
    return {
        p["key"]: p["status"]
        for p in er.record_and_classify(
            [{"key": k, "desc": k} for k in keys], session_id=session, run_family=family
        )
    }


def test_first_sighting_is_observed(ledger):
    assert _classify(["reread:x"], "s1") == {"reread:x": "observed"}


def test_second_distinct_session_is_encode_now(ledger):
    _classify(["reread:x"], "s1")
    assert _classify(["reread:x"], "s2") == {"reread:x": "encode_now"}


def test_same_session_repeat_counts_once(ledger):
    _classify(["reread:x"], "s1")
    assert _classify(["reread:x"], "s1") == {"reread:x": "observed"}


def test_recurrence_after_encode_escalates(ledger):
    _classify(["reread:x"], "s1")
    _classify(["reread:x"], "s2")
    er.mark_encoded(["reread:x"])
    assert _classify(["reread:x"], "s3") == {"reread:x": "recurred_after_encode"}


def test_same_session_after_encode_does_not_escalate(ledger):
    """A later background review of the same snapshot must not look like
    a post-rule recurrence.

    The encoded marker is written after the review patches a skill. The next
    review can still see the same old tool calls from that session, but the
    pattern has not recurred in a new session yet.
    """
    _classify(["reread:x"], "s1")
    assert _classify(["reread:x"], "s2") == {"reread:x": "encode_now"}
    er.mark_encoded(["reread:x"])

    assert _classify(["reread:x"], "s2") == {"reread:x": "observed"}
    assert _classify(["reread:x"], "s3") == {"reread:x": "recurred_after_encode"}


def test_ledger_corrupt_lines_are_skipped(ledger):
    ledger.write_text('not json\n{"also": "wrong shape"}\n')
    assert _classify(["reread:x"], "s1") == {"reread:x": "observed"}


# ---------------------------------------------------------------------------
# Prompt block + outcome application
# ---------------------------------------------------------------------------

def test_build_efficiency_block_contents(ledger):
    msgs = _segmented_read_snapshot(3)
    block, ctx = er.build_efficiency_block(msgs, session_id="s1", run_family="cron_test")
    assert "EFFICIENCY" in block
    assert RUNNER in block
    # hard constraints present
    assert "replacement" in block or "cheaper" in block
    assert "references/" in block
    # first sighting → observe-only, nothing to encode or escalate
    assert ctx["encode_keys"] == []
    assert ctx["escalations"] == []


def test_build_efficiency_block_gates_on_clean_session(ledger):
    block, ctx = er.build_efficiency_block([], session_id="s1", run_family="cron_test")
    assert block is None and ctx is None


def test_run_family_derivation():
    assert er.run_family("cron_5958dff23566_20260702_170025", "cron") == "cron_5958dff23566"
    assert er.run_family("some-uuid", "discord") == "discord"
    assert er.run_family(None, None) == "interactive"


def _skill_write_messages():
    tc = _call("skill_manage", {"action": "write_file", "name": "email-triage-research",
                                "file_path": "references/cli.md", "file_content": "x"})
    return [
        _assistant(tc),
        _result(tc["id"], json.dumps({"success": True, "message": "File written"})),
    ]


def test_apply_outcome_marks_encoded_after_skill_write(ledger):
    _classify(["reread:x"], "s1")
    _classify(["reread:x"], "s2")  # → encode_now next time
    ctx = {"encode_keys": ["reread:x"], "escalations": []}
    actions = []
    er.apply_efficiency_outcome(ctx, _skill_write_messages(), prior_snapshot=[], actions=actions)
    # encoded marker written → next sighting escalates
    assert _classify(["reread:x"], "s3") == {"reread:x": "recurred_after_encode"}


def test_apply_outcome_no_write_no_mark(ledger):
    _classify(["reread:x"], "s1")
    ctx = {"encode_keys": ["reread:x"], "escalations": []}
    er.apply_efficiency_outcome(ctx, [], prior_snapshot=[], actions=[])
    # not marked encoded → second session is encode_now, not escalation
    assert _classify(["reread:x"], "s2") == {"reread:x": "encode_now"}


def test_apply_outcome_appends_escalation_line(ledger):
    ctx = {"encode_keys": [], "escalations": ["reread:x"]}
    actions = []
    er.apply_efficiency_outcome(ctx, [], prior_snapshot=[], actions=actions)
    assert len(actions) == 1 and "reread:x" in actions[0]


def test_apply_outcome_ignores_stale_skill_writes(ledger):
    """A skill_manage result inherited from the prior conversation snapshot
    must not count as 'the review encoded a rule'."""
    _classify(["reread:x"], "s1")
    stale = _skill_write_messages()
    ctx = {"encode_keys": ["reread:x"], "escalations": []}
    er.apply_efficiency_outcome(ctx, stale, prior_snapshot=stale, actions=[])
    assert _classify(["reread:x"], "s2") == {"reread:x": "encode_now"}


# ---------------------------------------------------------------------------
# Wiring: spawn_background_review_thread
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from agent.background_review import (
    spawn_background_review_thread,
    _SKILL_REVIEW_PROMPT,
    _MEMORY_REVIEW_PROMPT,
)


def _fake_agent():
    return SimpleNamespace(session_id="cron_5958dff23566_20260703_090000", platform="cron")


def test_spawn_appends_efficiency_block_for_wasteful_skill_review(ledger):
    msgs = _segmented_read_snapshot(3)
    _target, prompt = spawn_background_review_thread(
        _fake_agent(), msgs, review_memory=False, review_skills=True
    )
    assert prompt.startswith(_SKILL_REVIEW_PROMPT)
    assert "EFFICIENCY REVIEW" in prompt
    assert RUNNER in prompt


def test_spawn_clean_session_prompt_unchanged(ledger):
    _target, prompt = spawn_background_review_thread(
        _fake_agent(), [], review_memory=False, review_skills=True
    )
    assert prompt == _SKILL_REVIEW_PROMPT


def test_spawn_memory_only_review_never_gets_block(ledger):
    msgs = _segmented_read_snapshot(3)
    _target, prompt = spawn_background_review_thread(
        _fake_agent(), msgs, review_memory=True, review_skills=False
    )
    assert prompt == _MEMORY_REVIEW_PROMPT
    assert "EFFICIENCY" not in prompt


def test_spawn_survives_efficiency_failure(ledger, monkeypatch):
    """A digest/ledger crash must not cost the skill review itself."""
    import agent.efficiency_review as _er

    def _boom(*a, **k):
        raise RuntimeError("ledger disk full")

    monkeypatch.setattr(_er, "build_efficiency_block", _boom)
    msgs = _segmented_read_snapshot(3)
    _target, prompt = spawn_background_review_thread(
        _fake_agent(), msgs, review_memory=False, review_skills=True
    )
    assert prompt == _SKILL_REVIEW_PROMPT
