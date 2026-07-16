from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GOVERNANCE = (
    REPO_ROOT
    / "skills/autonomous-ai-agents/independent-review-governance/SKILL.md"
)
CODE_REVIEW = REPO_ROOT / "skills/software-development/code-review-execution/SKILL.md"
CODE_REVIEW_TEMPLATE = (
    REPO_ROOT / "skills/software-development/code-review-execution/code-reviewer.md"
)
SUBAGENT_DRIVEN = (
    REPO_ROOT
    / "optional-skills/software-development/subagent-driven-development/SKILL.md"
)
GOVERNANCE_DOC = (
    REPO_ROOT
    / "website/docs/user-guide/skills/bundled/autonomous-ai-agents/"
    "autonomous-ai-agents-independent-review-governance.md"
)
CODE_REVIEW_DOC = (
    REPO_ROOT
    / "website/docs/user-guide/skills/bundled/software-development/"
    "software-development-code-review-execution.md"
)
CODE_REVIEW_ZH_DOC = (
    REPO_ROOT
    / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/"
    "skills/bundled/software-development/software-development-code-review-execution.md"
)
SUBAGENT_DRIVEN_DOC = (
    REPO_ROOT
    / "website/docs/user-guide/skills/optional/software-development/"
    "software-development-subagent-driven-development.md"
)
PLAN_SKILL = REPO_ROOT / "skills/software-development/plan/SKILL.md"
PLAN_DOC = (
    REPO_ROOT
    / "website/docs/user-guide/skills/bundled/software-development/"
    "software-development-plan.md"
)
SIMPLIFY_CODE = REPO_ROOT / "skills/software-development/simplify-code/SKILL.md"
SIMPLIFY_CODE_DOC = (
    REPO_ROOT
    / "website/docs/user-guide/skills/bundled/software-development/"
    "software-development-simplify-code.md"
)
EN_SKILLS_CATALOG = REPO_ROOT / "website/docs/reference/skills-catalog.md"
ZH_SKILLS_CATALOG = (
    REPO_ROOT
    / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/"
    "skills-catalog.md"
)
KANBAN_SWARM = REPO_ROOT / "hermes_cli/kanban_swarm.py"


OLD_OWNER_NAMES = ("requesting-code-review", "multi-agent-review-governance")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).split()).lower()


def test_review_owner_descriptions_are_disjoint_and_routeable():
    governance = _text(GOVERNANCE)
    code_review = _text(CODE_REVIEW)

    assert (
        "description: \"Use before any independent Reviewer, Codex, Claude, human, "
        "or domain review is launched" in governance
    )
    assert (
        "description: \"Use only after independent-review-governance authorizes "
        "an independent pass" in code_review
    )


def test_governance_owns_authorization_budget_and_stop_not_code_mechanics():
    text = _normalized(GOVERNANCE)
    assert "canonical control plane for independent review across domains" in text
    assert "global pass count" in text
    assert "default budget" in text
    assert "targeted pass 2" in text
    assert "the budget does not reset" in text
    assert "if the next-pass gate is not met, stop reviewing" in text
    assert "git diff head" not in text
    assert "untracked files" not in text


def test_code_review_executes_one_authorized_pass_without_budget_policy():
    text = _normalized(CODE_REVIEW)
    assert "one software review pass already authorized" in text
    assert "does not decide whether review is needed" in text
    assert "count global passes" in text
    assert "authorize follow-up review" in text
    assert "git diff head --stat -- <task-files...>" in text
    assert "git diff head --check -- <task-files...>" in text
    assert "untracked files" in text
    assert "default budget" not in text
    assert "targeted pass 2" not in text
    assert "the budget does not reset" not in text


def test_reviewer_template_uses_canonical_one_shot_profile():
    text = _text(CODE_REVIEW_TEMPLATE)
    assert 'subagent_type="Reviewer"' in text
    assert 'subagent_type="Explore"' not in text
    assert "do not automatically launch another reviewer" in text.lower()


def test_subagent_driven_routes_authorization_before_execution():
    for path in (SUBAGENT_DRIVEN, SUBAGENT_DRIVEN_DOC):
        text = _normalized(path)
        governance_at = text.index("load `independent-review-governance`")
        execution_at = text.index("use `code-review-execution`", governance_at)
        assert governance_at < execution_at
        assert "parent/controller owns all independent review" in text
        assert "per-task independent reviewers are exceptional" in text


def test_final_reviewer_example_contains_the_review_inputs():
    for path in (SUBAGENT_DRIVEN, SUBAGENT_DRIVEN_DOC):
        text = _normalized(path)
        assert 'subagent_type="reviewer"' in text
        assert "approved contract:" in text
        assert "acceptance criteria / invariants:" in text
        assert "scoped integrated diff or review package:" in text
        assert "fresh test / build / runtime evidence:" in text


def test_plan_handoff_preserves_controller_checks_and_one_final_review():
    for path in (PLAN_SKILL, PLAN_DOC):
        text = _normalized(path)
        assert "two-stage review" not in text
        assert "controller-owned diff/test verification after each task" in text
        assert "one final whole-change independent review" in text


def test_simplify_code_routes_final_review_through_governance():
    for path in (SIMPLIFY_CODE, SIMPLIFY_CODE_DOC):
        text = _normalized(path)
        assert "does not replace or multiply that final review" in text
        assert "route the decision through `independent-review-governance`" in text
        assert "authorized software pass uses `code-review-execution`" in text


def test_runtime_verifier_preloads_governance_not_execution():
    text = _text(KANBAN_SWARM)
    assert 'skills=["independent-review-governance"]' in text
    assert 'skills=["code-review-execution"]' not in text


def test_generated_docs_and_catalogs_use_only_new_owner_names():
    for path in (
        GOVERNANCE_DOC,
        CODE_REVIEW_DOC,
        CODE_REVIEW_ZH_DOC,
        EN_SKILLS_CATALOG,
        ZH_SKILLS_CATALOG,
    ):
        text = _text(path)
        for old_name in OLD_OWNER_NAMES:
            assert old_name not in text, path

    en = _text(EN_SKILLS_CATALOG)
    zh = _text(ZH_SKILLS_CATALOG)
    assert "independent-review-governance" in en
    assert "code-review-execution" in en
    assert "independent-review-governance" in zh
    assert "code-review-execution" in zh


def test_old_bundled_owner_paths_are_gone():
    assert not (
        REPO_ROOT / "skills/software-development/requesting-code-review"
    ).exists()
    assert not (
        REPO_ROOT
        / "website/docs/user-guide/skills/bundled/software-development/"
        "software-development-requesting-code-review.md"
    ).exists()
