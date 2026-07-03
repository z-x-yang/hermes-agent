"""Tests for the .history/ backup on autonomous (background-review) full
rewrites of SKILL.md.

Skills live outside git (skills/* is gitignored), and compaction is an
autonomous full rewrite — one bad pass could silently destroy months of
accumulated lessons. So: before a background-review `edit` overwrites
SKILL.md, the previous content is preserved under <skill>/.history/
(last 5 kept). Backup failure REFUSES the edit (fail fast) — an autonomous
destructive rewrite must not proceed without its safety net. Foreground
(user-directed) edits are unaffected.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import _create_skill, _edit_skill


@contextmanager
def _skill_dir(tmp_path):
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield


@contextmanager
def _as_background_review():
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        yield
    finally:
        reset_current_write_origin(token)


def _content(version):
    return (
        f"---\nname: test-skill\ndescription: Version {version}.\n---\n\n"
        f"# Test Skill v{version}\n\nStep 1: Do thing {version}.\n"
    )


def test_background_rewrite_backs_up_previous_content(tmp_path):
    with _skill_dir(tmp_path):
        assert _create_skill("test-skill", _content(1))["success"]
        with _as_background_review():
            assert _edit_skill("test-skill", _content(2))["success"]
        backups = sorted((tmp_path / "test-skill" / ".history").glob("SKILL-*.md"))
        assert len(backups) == 1
        assert "Version 1" in backups[0].read_text()
        assert "Version 2" in (tmp_path / "test-skill" / "SKILL.md").read_text()


def test_history_keeps_only_last_five(tmp_path):
    with _skill_dir(tmp_path):
        assert _create_skill("test-skill", _content(0))["success"]
        with _as_background_review():
            for v in range(1, 8):
                assert _edit_skill("test-skill", _content(v))["success"]
        backups = sorted((tmp_path / "test-skill" / ".history").glob("SKILL-*.md"))
        assert len(backups) == 5
        # the newest backup holds the immediately-previous version (6)
        assert "Version 6" in backups[-1].read_text()


def test_foreground_edit_makes_no_backup(tmp_path):
    with _skill_dir(tmp_path):
        assert _create_skill("test-skill", _content(1))["success"]
        assert _edit_skill("test-skill", _content(2))["success"]
        assert not (tmp_path / "test-skill" / ".history").exists()


def test_backup_failure_refuses_the_rewrite(tmp_path, monkeypatch):
    with _skill_dir(tmp_path):
        assert _create_skill("test-skill", _content(1))["success"]
        import tools.skill_manager_tool as smt

        def _boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(smt, "_backup_skill_history", _boom)
        with _as_background_review():
            result = _edit_skill("test-skill", _content(2))
        assert not result["success"]
        assert "backup" in result["error"].lower()
        # original content untouched
        assert "Version 1" in (tmp_path / "test-skill" / "SKILL.md").read_text()
