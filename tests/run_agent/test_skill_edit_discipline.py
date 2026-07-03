"""Tests for the skill edit discipline + compaction mechanism.

Design under test (user requirements, 2026-07-03):
  • Skills MAY grow with experience — growth buys efficiency/quality.
  • But they must stay CLEAN: no near-duplicate rules; a new lesson that
    contradicts old text OVERWRITES it (old and new never coexist).
  • Placement is economic: near-always-needed content lives in the SKILL.md
    body; low-frequency content lives in references/ behind a pointer that
    names its trigger condition.
  • Compaction is deterministically NOMINATED (patch_count accumulated since
    the last compaction baseline >= threshold), executed by the review as a
    full-read + full-rewrite, and its mandate is dedup/decontradiction —
    NEVER shortening.
  • Pinned skills are read-only for the autonomous review (guard-enforced);
    the prompts must say so instead of the stale "pinned CAN be improved".
"""

import json

import pytest

from agent import efficiency_review as er
from agent.background_review import (
    _COMBINED_REVIEW_PROMPT,
    _SKILL_REVIEW_PROMPT,
    spawn_background_review_thread,
)
from tools import skill_usage


# ---------------------------------------------------------------------------
# F1: prompt discipline text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", [_SKILL_REVIEW_PROMPT, _COMBINED_REVIEW_PROMPT])
def test_prompts_carry_edit_discipline(prompt):
    assert "Edit discipline" in prompt
    # read-before-write
    assert "READ BEFORE WRITE" in prompt
    # duplicate handling and contradiction overwrite
    assert "DUPLICATE" in prompt and "MERGE" in prompt
    assert "CONTRADICTION" in prompt and "OVERWRITE" in prompt
    # growth is explicitly allowed
    assert "Growth is fine" in prompt
    # frequency-based placement with trigger-condition pointers
    assert "PLACE BY FREQUENCY" in prompt
    assert "trigger condition" in prompt


@pytest.mark.parametrize("prompt", [_SKILL_REVIEW_PROMPT, _COMBINED_REVIEW_PROMPT])
def test_prompts_pinned_text_matches_write_guard(prompt):
    """The write guard refuses autonomous writes to pinned skills; the prompt
    must not claim the opposite (stale 'CAN be improved' text)."""
    assert "CAN be improved" not in prompt
    assert "read-only" in prompt
    # redirect path is named
    assert "unpinned companion" in prompt


def test_hard_constraints_updated():
    hc = er._HARD_CONSTRAINTS
    # net-zero growth requirement is withdrawn
    assert "do not grow" not in hc
    # dedup + contradiction overwrite discipline present
    assert "REPLACE the old text" in hc
    # frequency-based placement replaced the blanket references/ preference
    assert "frequency" in hc
    assert "trigger condition" in hc
    # pinned redirect present
    assert "pinned" in hc
    # kept from the original: replacement-path rule and evidence-gate rule
    assert "cheaper replacement path" in hc
    assert "Never remove evidence-gathering" in hc


# ---------------------------------------------------------------------------
# F2: compaction nomination
# ---------------------------------------------------------------------------

def _seed_usage(records):
    skill_usage.save_usage(records)


def _rec(patch_count=0, compacted=0, pinned=False, state="active"):
    r = skill_usage._empty_record()
    r["patch_count"] = patch_count
    r["compacted_patch_count"] = compacted
    r["pinned"] = pinned
    r["state"] = state
    return r


@pytest.fixture()
def skills_on_disk(monkeypatch):
    """Every skill exists locally, is curation-eligible, and its SKILL.md is
    comfortably above the compaction size floor."""
    monkeypatch.setattr(er, "_skill_body_bytes", lambda name: 20_000)
    monkeypatch.setattr(skill_usage, "is_curation_eligible", lambda name: True)


def _snap(*mentioned):
    """A conversation snapshot whose content mentions the given skill names —
    compaction grounding requires the session to have touched the skill's
    territory."""
    return [{"role": "user", "content": "worked on " + " and ".join(mentioned)}]


def test_nominate_most_overdue_mentioned_skill(skills_on_disk):
    _seed_usage({
        "email-triage-research": _rec(patch_count=149),
        "email-triage-closer": _rec(patch_count=45),
        "small-skill": _rec(patch_count=3),
    })
    nom = er.nominate_compaction(_snap("email-triage-research", "email-triage-closer"))
    assert nom["name"] == "email-triage-research"
    assert nom["overdue"] == 149


def test_nominate_requires_session_grounding(skills_on_disk):
    """A session that never touched the skill must not trigger a blind cold
    rewrite of it (the o2-cluster-workflow incident, 2026-07-03)."""
    _seed_usage({"o2-cluster-workflow": _rec(patch_count=223)})
    assert er.nominate_compaction(_snap("unrelated-topic")) is None
    assert er.nominate_compaction([]) is None
    # grounding also looks inside tool-call arguments, not just content
    tool_snap = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "skill_view", "arguments": '{"name": "o2-cluster-workflow"}'}}],
    }]
    assert er.nominate_compaction(tool_snap)["name"] == "o2-cluster-workflow"


def test_nominate_grounding_picks_mentioned_over_more_overdue(skills_on_disk):
    _seed_usage({
        "not-mentioned": _rec(patch_count=200),
        "mentioned": _rec(patch_count=20),
    })
    assert er.nominate_compaction(_snap("mentioned"))["name"] == "mentioned"


def test_nominate_respects_size_floor(skills_on_disk, monkeypatch):
    """Small skills can't hold meaningful redundancy — rewriting them is pure
    churn risk, so they are never nominated."""
    monkeypatch.setattr(er, "_skill_body_bytes", lambda name: er.COMPACTION_MIN_BYTES - 1)
    _seed_usage({"tiny": _rec(patch_count=99)})
    assert er.nominate_compaction(_snap("tiny")) is None


def test_nominate_none_below_threshold(skills_on_disk):
    _seed_usage({"a": _rec(patch_count=er.COMPACTION_THRESHOLD - 1)})
    assert er.nominate_compaction(_snap("a")) is None


def test_nominate_respects_baseline(skills_on_disk):
    _seed_usage({"a": _rec(patch_count=100, compacted=95)})
    assert er.nominate_compaction(_snap("a")) is None


def test_nominate_skips_pinned_and_inactive(skills_on_disk):
    _seed_usage({
        "pinned-one": _rec(patch_count=99, pinned=True),
        "archived-one": _rec(patch_count=99, state="archived"),
        "eligible": _rec(patch_count=20),
    })
    nom = er.nominate_compaction(_snap("pinned-one", "archived-one", "eligible"))
    assert nom["name"] == "eligible"


def test_nominate_skips_skills_missing_on_disk(skills_on_disk, monkeypatch):
    monkeypatch.setattr(er, "_skill_body_bytes", lambda name: None)
    _seed_usage({"ghost": _rec(patch_count=99)})
    assert er.nominate_compaction(_snap("ghost")) is None


def test_nominate_threshold_override(skills_on_disk):
    _seed_usage({"a": _rec(patch_count=5)})
    assert er.nominate_compaction(_snap("a"), threshold=5)["name"] == "a"
    assert er.nominate_compaction(_snap("a"), threshold=6) is None


def test_set_compaction_baseline_roundtrip(skills_on_disk):
    _seed_usage({"a": _rec(patch_count=42)})
    skill_usage.set_compaction_baseline("a")
    assert skill_usage.get_record("a")["compacted_patch_count"] == 42
    assert er.nominate_compaction(_snap("a")) is None


# ---------------------------------------------------------------------------
# F2: compaction prompt block mandate
# ---------------------------------------------------------------------------

def test_compaction_block_mandate():
    block = er.build_compaction_block({"name": "cluster-workflow", "overdue": 23})
    assert "cluster-workflow" in block and "23" in block
    # full read first, then JUDGE — rewrite must be earned, not automatic
    assert "skill_view" in block
    assert "action=edit" in block
    assert "do NOT edit" in block
    assert er.COMPACTION_CLEAN_SENTINEL in block
    # the objective is hygiene, NOT shortening
    assert "NOT shortening" in block
    assert "length is not a metric" in block
    assert "When in doubt, KEEP" in block
    # accounting requirement
    assert "account for what changed" in block
    # the false "folded N patches" claim is retired
    assert "folded" not in block


# ---------------------------------------------------------------------------
# F2: compaction outcome (baseline reset + accounting + shrink warning)
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


def _edit_messages(skill_name, success=True):
    tc = _call("skill_manage", {"action": "edit", "name": skill_name, "content": "..."})
    return [
        {"role": "assistant", "content": "", "tool_calls": [tc]},
        {"role": "tool", "tool_call_id": tc["id"],
         "content": json.dumps({"success": success, "message": "updated"})},
    ]


def test_compaction_outcome_rewrite_resets_baseline_and_accounts(skills_on_disk, monkeypatch):
    monkeypatch.setattr(er, "_compaction_shrink_pct", lambda name: None)
    _seed_usage({"a": _rec(patch_count=42)})
    actions = []
    er.apply_compaction_outcome(
        {"name": "a", "overdue": 42}, _edit_messages("a"), prior_snapshot=[], actions=actions
    )
    assert skill_usage.get_record("a")["compacted_patch_count"] == 42
    # honest accounting: says a rewrite happened + how many patches were
    # pending — never a false "folded N patches" claim
    assert any("rewrote skill 'a'" in a and "42 patches since last check" in a for a in actions)
    assert not any("folded" in a for a in actions)


def test_compaction_outcome_clean_verdict_resets_baseline(skills_on_disk):
    """Model reads the skill, judges it clean, states the sentinel instead of
    editing — that closes the loop too (earn-the-rewrite)."""
    _seed_usage({"a": _rec(patch_count=42)})
    actions = []
    verdict_messages = [
        {"role": "assistant",
         "content": f"Reviewed the skill in full. {er.COMPACTION_CLEAN_SENTINEL}"},
    ]
    er.apply_compaction_outcome(
        {"name": "a", "overdue": 42}, verdict_messages, prior_snapshot=[], actions=actions
    )
    assert skill_usage.get_record("a")["compacted_patch_count"] == 42
    assert any("judged clean" in a and "no rewrite needed" in a for a in actions)


def test_compaction_outcome_clean_verdict_in_prior_snapshot_ignored(skills_on_disk):
    """A sentinel inherited from the pre-review conversation must not count —
    only fresh assistant text from this review closes the loop."""
    _seed_usage({"a": _rec(patch_count=42)})
    prior = [{"role": "assistant", "content": er.COMPACTION_CLEAN_SENTINEL}]
    actions = []
    er.apply_compaction_outcome(
        {"name": "a", "overdue": 42}, list(prior), prior_snapshot=prior, actions=actions
    )
    assert skill_usage.get_record("a")["compacted_patch_count"] == 0
    assert actions == []


def test_compaction_outcome_no_rewrite_no_verdict_no_reset(skills_on_disk):
    _seed_usage({"a": _rec(patch_count=42)})
    actions = []
    er.apply_compaction_outcome(
        {"name": "a", "overdue": 42}, [], prior_snapshot=[], actions=actions
    )
    assert skill_usage.get_record("a")["compacted_patch_count"] == 0
    assert actions == []


def test_compaction_outcome_ignores_edit_of_other_skill(skills_on_disk):
    _seed_usage({"a": _rec(patch_count=42)})
    actions = []
    er.apply_compaction_outcome(
        {"name": "a", "overdue": 42}, _edit_messages("b"), prior_snapshot=[], actions=actions
    )
    assert skill_usage.get_record("a")["compacted_patch_count"] == 0


def test_compaction_outcome_failed_edit_no_reset(skills_on_disk):
    _seed_usage({"a": _rec(patch_count=42)})
    actions = []
    er.apply_compaction_outcome(
        {"name": "a", "overdue": 42},
        _edit_messages("a", success=False),
        prior_snapshot=[],
        actions=actions,
    )
    assert skill_usage.get_record("a")["compacted_patch_count"] == 0


def test_compaction_outcome_shrink_warning(skills_on_disk, monkeypatch):
    monkeypatch.setattr(er, "_compaction_shrink_pct", lambda name: 47)
    _seed_usage({"a": _rec(patch_count=42)})
    actions = []
    er.apply_compaction_outcome(
        {"name": "a", "overdue": 42}, _edit_messages("a"), prior_snapshot=[], actions=actions
    )
    assert any("shrank 47%" in a for a in actions)
    assert any(".history" in a for a in actions)


def test_compaction_outcome_mild_shrink_no_warning(skills_on_disk, monkeypatch):
    monkeypatch.setattr(er, "_compaction_shrink_pct", lambda name: 12)
    _seed_usage({"a": _rec(patch_count=42)})
    actions = []
    er.apply_compaction_outcome(
        {"name": "a", "overdue": 42}, _edit_messages("a"), prior_snapshot=[], actions=actions
    )
    assert not any("shrank" in a for a in actions)


# ---------------------------------------------------------------------------
# Wiring: spawn_background_review_thread injects the compaction block
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _fake_agent(**kw):
    return SimpleNamespace(session_id="s1", platform="cli", **kw)


def test_spawn_injects_compaction_block(skills_on_disk):
    _seed_usage({"hoarder": _rec(patch_count=99)})
    _target, prompt = spawn_background_review_thread(
        _fake_agent(), _snap("hoarder"), review_memory=False, review_skills=True
    )
    assert "SKILL COMPACTION" in prompt
    assert "hoarder" in prompt


def test_spawn_no_compaction_when_session_unrelated(skills_on_disk):
    """The o2-cluster-workflow incident: an overdue skill the session never
    touched must not be nominated by that session's review."""
    _seed_usage({"hoarder": _rec(patch_count=99)})
    _target, prompt = spawn_background_review_thread(
        _fake_agent(), _snap("something-else"), review_memory=False, review_skills=True
    )
    assert "SKILL COMPACTION" not in prompt


def test_spawn_no_compaction_when_none_overdue(skills_on_disk):
    _seed_usage({"a": _rec(patch_count=2)})
    _target, prompt = spawn_background_review_thread(
        _fake_agent(), _snap("a"), review_memory=False, review_skills=True
    )
    assert "SKILL COMPACTION" not in prompt


def test_spawn_memory_only_never_compacts(skills_on_disk):
    _seed_usage({"hoarder": _rec(patch_count=99)})
    _target, prompt = spawn_background_review_thread(
        _fake_agent(), _snap("hoarder"), review_memory=True, review_skills=False
    )
    assert "SKILL COMPACTION" not in prompt


def test_spawn_threshold_from_agent_attr(skills_on_disk):
    _seed_usage({"a": _rec(patch_count=5)})
    _target, prompt = spawn_background_review_thread(
        _fake_agent(skill_compaction_threshold=5), _snap("a"),
        review_memory=False, review_skills=True
    )
    assert "SKILL COMPACTION" in prompt


def test_spawn_survives_compaction_failure(skills_on_disk, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("usage file corrupted")

    monkeypatch.setattr(er, "nominate_compaction", _boom)
    _seed_usage({"hoarder": _rec(patch_count=99)})
    _target, prompt = spawn_background_review_thread(
        _fake_agent(), _snap("hoarder"), review_memory=False, review_skills=True
    )
    assert prompt.startswith(_SKILL_REVIEW_PROMPT)
