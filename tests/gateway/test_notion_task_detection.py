from plugins.platforms.discord.notion_tasks import detection as d

TASKS = {"1f17a58d229e816f839bef72f6f2ec72"}


def _task_page(status_prop):
    return {
        "parent": {"type": "database_id", "database_id": "1f17a58d-229e-816f-839b-ef72f6f2ec72"},
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "Reply to Alice"}]},
            "Status": status_prop,
        },
    }


def test_extract_markdown_link():
    text = "Done your [Reply to Alice](https://www.notion.so/Reply-1f17a58d229e816f839bef72f6f2ec72) today"
    links = d.extract_notion_links(text)
    assert len(links) == 1
    assert links[0].page_id == "1f17a58d229e816f839bef72f6f2ec72"
    assert links[0].anchor == "Reply to Alice"


def test_extract_bare_url_and_hyphenated_id():
    text = "see https://notion.so/1f17a58d-229e-816f-839b-ef72f6f2ec72"
    links = d.extract_notion_links(text)
    assert links[0].page_id == "1f17a58d229e816f839bef72f6f2ec72"
    assert links[0].anchor is None


def test_non_notion_link_ignored():
    assert d.extract_notion_links("[x](https://example.com/abc)") == []


def test_rejects_spoofed_host():
    # Host is the authority: a real task id behind a non-notion host must NOT match.
    pid = "1f17a58d229e816f839bef72f6f2ec72"
    assert d.extract_notion_links(f"[x](https://evil.example/notion.so/{pid})") == []
    assert d.extract_notion_links(f"https://evil.example/notion.so/{pid}") == []


def test_id_only_in_query_string_ignored():
    # id must come from the path, not the query string
    pid = "1f17a58d229e816f839bef72f6f2ec72"
    assert d.extract_notion_links(f"https://www.notion.so/page?ref={pid}") == []


def test_is_task_page_matches_database_parent():
    assert d.is_task_page(_task_page({"type": "select", "select": {"name": "Todo"}}), TASKS) is True


def test_is_task_page_matches_data_source_parent():
    page = {"parent": {"type": "data_source_id", "data_source_id": "1f17a58d229e816f839bef72f6f2ec72"},
            "properties": {}}
    assert d.is_task_page(page, TASKS) is True


def test_is_task_page_rejects_other_db():
    page = {"parent": {"type": "database_id", "database_id": "deadbeef" * 4}, "properties": {}}
    assert d.is_task_page(page, TASKS) is False


def test_read_status_select_and_status_kinds():
    assert d.read_status(_task_page({"type": "select", "select": {"name": "Todo"}})) == ("Todo", "select")
    assert d.read_status(_task_page({"type": "status", "status": {"name": "In Progress"}})) == ("In Progress", "status")


def test_status_patch_roundtrip():
    assert d.status_patch("Done", "select") == {"Status": {"select": {"name": "Done"}}}
    assert d.status_patch("Done", "status") == {"Status": {"status": {"name": "Done"}}}


def test_page_title_reads_title_property():
    assert d.page_title(_task_page({"type": "select", "select": None})) == "Reply to Alice"
