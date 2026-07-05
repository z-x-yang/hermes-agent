from plugins.platforms.discord.notion_tasks.threading import generate_thread_title, read_thread_binding


def _page(title="Reply to Tianxi about manuscript plan", project="CRAFT NM"):
    return {"properties": {
        "Name": {"type": "title", "title": [{"plain_text": title}]},
        "Project": {"type": "relation", "relation": [{"id": "p1"}], "has_more": False},
        "Discord Thread ID": {"type": "rich_text", "rich_text": [{"plain_text": "1523"}]},
        "Discord Thread URL": {"type": "url", "url": "https://discord.com/channels/g/c/1523"},
        "Thread Title Mode": {"type": "select", "select": {"name": "auto"}},
    }, "project_name_for_test": project}


def test_generate_thread_title_uses_project_source_and_truncates():
    title = generate_thread_title(_page(), source_hint="Email")
    assert title.startswith("CRAFT NM · ")
    assert "Reply to Tianxi" in title
    assert len(title) <= 80
    assert "loot" not in title.lower()


def test_child_title_uses_parent_context():
    title = generate_thread_title(_page("Baseline audit"), parent_title="Response package")
    assert "Response package › Baseline audit" in title


def test_read_thread_binding_extracts_id_url_mode():
    binding = read_thread_binding(_page())
    assert binding["thread_id"] == "1523"
    assert binding["thread_url"].endswith("/1523")
    assert binding["title_mode"] == "auto"
