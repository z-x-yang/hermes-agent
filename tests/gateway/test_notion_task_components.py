import re

from plugins.platforms.discord.notion_tasks import components as c

PID = "1f17a58d229e816f839bef72f6f2ec72"


def test_make_and_match_custom_id():
    cid = c.make_custom_id("done", PID)
    assert cid == f"ntask:done:{PID}"
    m = re.fullmatch(c.CUSTOM_ID_RE, cid)
    assert m and m.group("action") == "done" and m.group("page_id") == PID


def test_button_component_done_and_undo():
    done = c.button_component("done", PID)
    assert done == {"type": 2, "style": 3, "label": c.LABEL_DONE, "custom_id": f"ntask:done:{PID}"}
    undo = c.button_component("undo", PID)
    assert undo["style"] == 2 and undo["label"] == c.LABEL_UNDO and undo["custom_id"] == f"ntask:undo:{PID}"


def test_components_payload_empty():
    assert c.components_payload([]) == []


def test_components_payload_packs_rows_of_five():
    tasks = [("done", f"{i:032x}") for i in range(7)]
    rows = c.components_payload(tasks)
    assert len(rows) == 2
    assert len(rows[0]["components"]) == 5
    assert len(rows[1]["components"]) == 2
    assert all(r["type"] == 1 for r in rows)


def test_components_payload_caps_at_25():
    tasks = [("done", f"{i:032x}") for i in range(40)]
    rows = c.components_payload(tasks)
    assert sum(len(r["components"]) for r in rows) == 25
    assert len(rows) == 5


def test_strike_done_single_line_preserves_text():
    assert c.strike_done("Reply to Alice") == "✅ ~~Reply to Alice~~"


def test_strike_done_multiline_strikes_each_line():
    out = c.strike_done("Task title\nhttps://notion.so/x")
    assert out == "✅ ~~Task title~~\n~~https://notion.so/x~~"


def test_strike_done_empty():
    assert c.strike_done("") == "✅ 已完成"
    assert c.strike_done("   ") == "✅ 已完成"
