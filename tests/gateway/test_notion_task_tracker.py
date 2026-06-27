import pytest

from plugins.platforms.discord.notion_tasks.tracker import NotionTaskTracker


@pytest.fixture
def home(tmp_path, monkeypatch):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def test_upsert_meta_persists_and_reloads(home):
    t = NotionTaskTracker()
    t.upsert_meta("pid1", title="T", status_kind="status", original_status="To Do", done=False)
    rec = NotionTaskTracker().get("pid1")  # new instance reads from disk
    assert rec["title"] == "T"
    assert rec["original_status"] == "To Do"
    assert rec["status_kind"] == "status"
    assert rec["done"] is False


def test_add_location_keeps_multiple_and_preserves_orig_content(home):
    t = NotionTaskTracker()
    t.add_location("pid1", message_id="m1", channel_id="c1", orig_content="hello m1")
    t.add_location("pid1", message_id="m2", channel_id="c2", orig_content="hello m2")
    # a later call with None must NOT clobber the stored original
    t.add_location("pid1", message_id="m1", channel_id="c1", orig_content=None)
    locs = {l["message_id"]: l for l in NotionTaskTracker().locations("pid1")}
    assert set(locs) == {"m1", "m2"}
    assert locs["m1"]["orig_content"] == "hello m1"
    assert locs["m1"]["channel_id"] == "c1"
    assert locs["m2"]["orig_content"] == "hello m2"


def test_upsert_meta_partial_does_not_wipe_fields(home):
    t = NotionTaskTracker()
    t.upsert_meta("pid1", title="T", status_kind="status", original_status="To Do", done=False)
    t.upsert_meta("pid1", done=True)  # only flip done
    rec = NotionTaskTracker().get("pid1")
    assert rec["done"] is True
    assert rec["original_status"] == "To Do"  # preserved
    assert rec["title"] == "T"


def test_get_missing_returns_none(home):
    assert NotionTaskTracker().get("nope") is None
    assert NotionTaskTracker().locations("nope") == []


def test_corrupt_state_is_backed_up_not_swallowed(home):
    (home / "discord_notion_tasks.json").write_text("{not valid json", encoding="utf-8")
    t = NotionTaskTracker()  # must not raise
    assert t.get("anything") is None
    assert (home / "discord_notion_tasks.corrupt").exists()  # backed up, visible
