from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUESTING_REVIEW = (
    REPO_ROOT / "skills/software-development/requesting-code-review/SKILL.md"
)
SUBAGENT_DRIVEN = (
    REPO_ROOT
    / "optional-skills/software-development/subagent-driven-development/SKILL.md"
)
REQUESTING_REVIEW_DOC = (
    REPO_ROOT
    / "website/docs/user-guide/skills/bundled/software-development/"
    "software-development-requesting-code-review.md"
)
REQUESTING_REVIEW_ZH_DOC = (
    REPO_ROOT
    / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/"
    "skills/bundled/software-development/software-development-requesting-code-review.md"
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
ZH_SKILLS_CATALOG = (
    REPO_ROOT
    / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/"
    "skills-catalog.md"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reviewer_examples_use_review_capable_profile():
    for path in (
        REQUESTING_REVIEW,
        REQUESTING_REVIEW_DOC,
        REQUESTING_REVIEW_ZH_DOC,
        SUBAGENT_DRIVEN,
        SUBAGENT_DRIVEN_DOC,
    ):
        text = _text(path)
        assert 'subagent_type="Explore"' not in text, path


def test_subagent_driven_workflow_has_controller_owned_final_review_only():
    for path in (SUBAGENT_DRIVEN, SUBAGENT_DRIVEN_DOC):
        text = _text(path)
        normalized = " ".join(text.split()).lower()
        assert "two-stage review" not in normalized, path
        assert "parent/controller owns all independent review" in normalized, path
        assert "one final independent review" in normalized, path


def test_requesting_review_does_not_reintroduce_per_task_review():
    for path in (
        REQUESTING_REVIEW,
        REQUESTING_REVIEW_DOC,
        REQUESTING_REVIEW_ZH_DOC,
    ):
        normalized = " ".join(_text(path).split()).lower()
        assert "after each task in subagent-driven-development" not in normalized, path
        assert (
            "do not run reviewer subagents after every task" in normalized
            or "默认不要在每个 task 后都启动 reviewer subagent" in normalized
        ), path


def test_plan_handoff_uses_controller_checks_and_one_final_review():
    for path in (PLAN_SKILL, PLAN_DOC):
        normalized = " ".join(_text(path).split()).lower()
        assert "two-stage review" not in normalized, path
        assert "controller-owned diff/test verification after each task" in normalized, path
        assert "one final whole-change independent review" in normalized, path


def test_requesting_review_includes_staged_unstaged_and_untracked_changes():
    for path in (REQUESTING_REVIEW, REQUESTING_REVIEW_DOC, REQUESTING_REVIEW_ZH_DOC):
        normalized = " ".join(_text(path).split()).lower()
        assert "git diff head --stat -- <changed-files...>" in normalized, path
        assert "git diff head --check -- <changed-files...>" in normalized, path
        assert "untracked files" in normalized, path
        assert "commit-only range" in normalized or "只比较 commits" in normalized, path


def test_final_reviewer_example_contains_the_review_inputs():
    for path in (SUBAGENT_DRIVEN, SUBAGENT_DRIVEN_DOC):
        normalized = " ".join(_text(path).split()).lower()
        assert "approved contract:" in normalized, path
        assert "acceptance criteria / invariants:" in normalized, path
        assert "scoped integrated diff or review package:" in normalized, path
        assert "fresh test / build / runtime evidence:" in normalized, path


def test_simplify_code_does_not_reintroduce_per_task_independent_review():
    for path in (SIMPLIFY_CODE, SIMPLIFY_CODE_DOC):
        normalized = " ".join(_text(path).split()).lower()
        assert "parallel review during implementation, per task" not in normalized, path
        assert "one final whole-change independent review" in normalized, path
        assert "does not replace or multiply that final review" in normalized, path


def test_chinese_catalog_does_not_advertise_removed_auto_fix_workflow():
    text = _text(ZH_SKILLS_CATALOG)
    line = next(
        item for item in text.splitlines() if "software-development-requesting-code-review" in item
    )
    assert "自动修复" not in line
    assert "全新独立审查" in line
