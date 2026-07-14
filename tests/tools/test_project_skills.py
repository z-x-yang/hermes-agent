from __future__ import annotations

import json


def test_project_skill_is_consistent_across_prompt_list_and_view(tmp_path, monkeypatch):
    from agent import prompt_builder, skill_commands, skill_utils
    from tools import skills_tool

    home = tmp_path / "home"
    local_skills = home / "skills"
    local_skills.mkdir(parents=True)
    (home / "config.yaml").write_text("skills: {}\n", encoding="utf-8")

    repo = tmp_path / "repo"
    nested = repo / "packages" / "worker"
    (repo / ".git").mkdir(parents=True)
    nested.mkdir(parents=True)
    project_skill = repo / ".evelyn" / "skills" / "project-owner"
    project_skill.mkdir(parents=True)
    (project_skill / "SKILL.md").write_text(
        "---\nname: project-owner\ndescription: Project-only owner.\n---\n\n# Project Owner\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("EVELYN_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(nested))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", local_skills)
    monkeypatch.setattr(skills_tool, "HERMES_HOME", home)
    skill_utils._external_dirs_cache_clear()
    prompt_builder.clear_skills_system_prompt_cache()

    prompt = prompt_builder.build_skills_system_prompt()
    listed = json.loads(skills_tool.skills_list_readonly())
    viewed = json.loads(skills_tool.skill_view_readonly("project-owner"))
    commands = skill_commands.scan_skill_commands()
    invocation = skill_commands.build_skill_invocation_message(
        "/project-owner", "run project workflow"
    )

    assert "project-owner: Project-only owner." in prompt
    assert [skill["name"] for skill in listed["skills"]] == ["project-owner"]
    assert viewed["success"] is True
    assert viewed["name"] == "project-owner"
    assert viewed["skill_dir"] == str(project_skill)
    assert commands["/project-owner"]["skill_md_path"] == str(project_skill / "SKILL.md")
    assert invocation is not None
    assert "run project workflow" in invocation


def test_project_skill_lists_without_profile_skills_directory(tmp_path, monkeypatch):
    from agent import prompt_builder, skill_utils
    from tools import skills_tool

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("skills: {}\n", encoding="utf-8")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    project_skill = repo / ".evelyn" / "skills" / "project-only"
    project_skill.mkdir(parents=True)
    (project_skill / "SKILL.md").write_text(
        "---\nname: project-only\ndescription: Project only.\n---\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("EVELYN_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(repo))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(skills_tool, "HERMES_HOME", home)
    skill_utils._external_dirs_cache_clear()
    prompt_builder.clear_skills_system_prompt_cache()

    listed = json.loads(skills_tool.skills_list_readonly())

    assert [skill["name"] for skill in listed["skills"]] == ["project-only"]


def test_skill_collision_is_explicit_across_prompt_list_and_view(tmp_path, monkeypatch):
    from agent import prompt_builder, skill_commands, skill_utils
    from tools import skills_tool

    home = tmp_path / "home"
    local_skill = home / "skills" / "local-owner"
    local_skill.mkdir(parents=True)
    (home / "config.yaml").write_text("skills: {}\n", encoding="utf-8")
    (local_skill / "SKILL.md").write_text(
        "---\nname: collision-owner\ndescription: Local owner.\n---\n",
        encoding="utf-8",
    )

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    project_skill = repo / ".evelyn" / "skills" / "project-owner"
    project_skill.mkdir(parents=True)
    (project_skill / "SKILL.md").write_text(
        "---\nname: collision-owner\ndescription: Project owner.\n---\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("EVELYN_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(repo))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(skills_tool, "HERMES_HOME", home)
    skill_utils._external_dirs_cache_clear()
    prompt_builder.clear_skills_system_prompt_cache()

    prompt = prompt_builder.build_skills_system_prompt()
    listed = json.loads(skills_tool.skills_list_readonly())
    viewed = json.loads(skills_tool.skill_view_readonly("collision-owner"))
    commands = skill_commands.scan_skill_commands()

    assert "collision-owner: [ambiguous" in prompt
    assert "collision-owner: Local owner." not in prompt
    assert "collision-owner: Project owner." not in prompt
    assert listed["skills"] == [
        {
            "name": "collision-owner",
            "description": "Ambiguous skill name; rename one source before loading.",
            "category": "collisions",
            "ambiguous": True,
            "matches": sorted(
                [str(local_skill / "SKILL.md"), str(project_skill / "SKILL.md")]
            ),
        }
    ]
    assert viewed["success"] is False
    assert "Ambiguous skill name" in viewed["error"]
    assert sorted(viewed["matches"]) == listed["skills"][0]["matches"]
    assert "/collision-owner" not in commands


def test_restrictive_import_is_hidden_and_fails_closed(tmp_path, monkeypatch):
    from agent import prompt_builder, skill_commands, skill_utils
    from tools import skills_tool

    home = tmp_path / "home"
    (home / "skills").mkdir(parents=True)
    (home / "config.yaml").write_text("skills: {}\n", encoding="utf-8")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    imported = repo / ".claude" / "skills" / "restricted-owner"
    imported.mkdir(parents=True)
    (imported / "SKILL.md").write_text(
        """---
name: restricted-owner
description: Must not load without its restrictions.
disable-model-invocation: true
disallowed-tools: [terminal]
context: fork
paths: [src/**]
---

# Restricted Owner
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("EVELYN_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(repo))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(skills_tool, "HERMES_HOME", home)
    skill_utils._external_dirs_cache_clear()
    prompt_builder.clear_skills_system_prompt_cache()

    prompt = prompt_builder.build_skills_system_prompt()
    listed = json.loads(skills_tool.skills_list_readonly())
    viewed = json.loads(skills_tool.skill_view_readonly("restricted-owner"))
    commands = skill_commands.scan_skill_commands()

    assert "restricted-owner" not in prompt
    assert listed["skills"] == []
    assert viewed["success"] is False
    assert viewed["compatibility"] == "rejected"
    assert viewed["unsupported_restrictive_fields"] == [
        "context",
        "disable-model-invocation",
        "disallowed-tools",
        "paths",
    ]
    assert "/restricted-owner" not in commands


def test_ineligible_duplicate_does_not_block_visible_skill_view(tmp_path, monkeypatch):
    from agent import prompt_builder, skill_commands, skill_utils
    from tools import skills_tool

    home = tmp_path / "home"
    local_skill = home / "skills" / "local-owner"
    local_skill.mkdir(parents=True)
    (home / "config.yaml").write_text("skills: {}\n", encoding="utf-8")
    (local_skill / "SKILL.md").write_text(
        "---\nname: shared-owner\ndescription: Valid profile owner.\n---\n",
        encoding="utf-8",
    )

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    restricted = repo / ".claude" / "skills" / "restricted-owner"
    restricted.mkdir(parents=True)
    (restricted / "SKILL.md").write_text(
        """---
name: shared-owner
description: Must remain hidden.
allowed-tools: [terminal]
---
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("EVELYN_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(repo))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(skills_tool, "HERMES_HOME", home)
    skill_utils._external_dirs_cache_clear()
    prompt_builder.clear_skills_system_prompt_cache()

    prompt = prompt_builder.build_skills_system_prompt()
    listed = json.loads(skills_tool.skills_list_readonly())
    viewed = json.loads(skills_tool.skill_view_readonly("shared-owner"))
    commands = skill_commands.scan_skill_commands()

    assert "shared-owner: Valid profile owner." in prompt
    assert [skill["name"] for skill in listed["skills"]] == ["shared-owner"]
    assert viewed["success"] is True
    assert viewed["skill_dir"] == str(local_skill)
    assert commands["/shared-owner"]["skill_dir"] == str(local_skill)


def test_slash_command_cache_is_scoped_to_runtime_project_cwd(tmp_path, monkeypatch):
    from agent import runtime_cwd, skill_commands, skill_utils
    from tools import skills_tool

    home = tmp_path / "home"
    (home / "skills").mkdir(parents=True)
    (home / "config.yaml").write_text("skills: {}\n", encoding="utf-8")

    repos = []
    for suffix in ("a", "b"):
        repo = tmp_path / f"repo-{suffix}"
        (repo / ".git").mkdir(parents=True)
        skill = repo / ".evelyn" / "skills" / f"project-{suffix}"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: project-{suffix}\ndescription: Project {suffix}.\n---\n",
            encoding="utf-8",
        )
        repos.append(repo)

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("EVELYN_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(skills_tool, "HERMES_HOME", home)
    skill_utils._external_dirs_cache_clear()
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)

    token_a = runtime_cwd.set_session_cwd(str(repos[0]))
    try:
        commands_a = skill_commands.get_skill_commands()
    finally:
        token_a.var.reset(token_a)

    token_b = runtime_cwd.set_session_cwd(str(repos[1]))
    try:
        commands_b = skill_commands.get_skill_commands()
    finally:
        token_b.var.reset(token_b)

    assert "/project-a" in commands_a
    assert "/project-b" not in commands_a
    assert "/project-b" in commands_b
    assert "/project-a" not in commands_b
