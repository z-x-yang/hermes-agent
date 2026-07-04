import re

from plugins.platforms.discord.notion_tasks import components as c

PID = "1f17a58d229e816f839bef72f6f2ec72"


def test_make_and_match_custom_id():
    cid = c.make_custom_id("done", PID)
    assert cid == f"ntask:done:{PID}"
    m = re.fullmatch(c.CUSTOM_ID_RE, cid)
    assert m and m.group("action") == "done" and m.group("page_id") == PID

    snooze = c.make_custom_id("snooze", PID)
    m2 = re.fullmatch(c.CUSTOM_ID_RE, snooze)
    assert m2 and m2.group("action") == "snooze" and m2.group("page_id") == PID


def test_button_component_done_and_undo():
    done = c.button_component("done", PID)
    assert done == {"type": 2, "style": 3, "label": c.LABEL_DONE, "custom_id": f"ntask:done:{PID}"}
    undo = c.button_component("undo", PID)
    assert undo["style"] == 2 and undo["label"] == c.LABEL_UNDO and undo["custom_id"] == f"ntask:undo:{PID}"
    snooze = c.button_component("snooze", PID)
    assert snooze["style"] == 2 and "稍后" in snooze["label"] and snooze["custom_id"] == f"ntask:snooze:{PID}"


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


def test_action_pairs_with_snooze_preserves_done_buttons_before_snooze_overflow():
    page_ids = [f"{i:032x}" for i in range(25)]
    pairs = c.action_pairs_with_snooze(page_ids)
    assert len(pairs) == 25
    assert pairs == [("done", pid) for pid in page_ids]

    thirteen = c.action_pairs_with_snooze(page_ids[:13])
    assert len(thirteen) == 25
    assert [p for action, p in thirteen if action == "done"] == page_ids[:13]
    assert [p for action, p in thirteen if action == "snooze"] == page_ids[:12]


# ===========================================================================
# numbered_label — 按钮编号标签
# ===========================================================================

class TestNumberedLabel:
    def test_actions_with_number(self):
        assert c.numbered_label("done", 1) == "✓ 1"
        assert c.numbered_label("snooze", 2) == "⏰ 2"
        assert c.numbered_label("undo", 3) == "↩ 3"

    def test_none_falls_back_to_legacy_constant(self):
        # from_custom_id 重建的按钮无编号信息,label 仅作路由占位、不展示
        assert c.numbered_label("done", None) == c.LABEL_DONE
        assert c.numbered_label("undo", None) == c.LABEL_UNDO
        assert c.numbered_label("snooze", None) == c.LABEL_SNOOZE

    def test_unknown_action_raises(self):
        import pytest
        with pytest.raises(ValueError):
            c.numbered_label("explode", 1)


# ===========================================================================
# task_card_embed — 任务卡(状态的纯函数)
# ===========================================================================

class TestTaskCardEmbed:
    def test_open_rows_numbered(self):
        e = c.task_card_embed([
            {"num": 1, "title": "BioRender access", "state": "open", "due_label": None},
            {"num": 2, "title": "Summer Camp lecture", "state": "open", "due_label": None},
        ])
        assert e["title"] == "📋 任务"
        assert "1️⃣ BioRender access" in e["description"]
        assert "2️⃣ Summer Camp lecture" in e["description"]
        assert e["footer"]["text"]

    def test_done_row_struck_and_counted(self):
        e = c.task_card_embed([
            {"num": 1, "title": "A", "state": "done", "due_label": None},
            {"num": 2, "title": "B", "state": "open", "due_label": None},
        ])
        assert e["title"] == "📋 任务 · 1/2 已完成"
        assert "✅ ~~A~~" in e["description"]
        assert "~~B~~" not in e["description"]

    def test_snoozed_row_shows_due(self):
        e = c.task_card_embed(
            [{"num": 1, "title": "A", "state": "snoozed", "due_label": "07/02 09:30"}])
        assert "⏰ 已延后·07/02 09:30 · A" in e["description"]

    def test_empty_rows_none(self):
        assert c.task_card_embed([]) is None

    def test_row_gt_ten_uses_plain_number(self):
        rows = [{"num": i, "title": f"t{i}", "state": "open", "due_label": None}
                for i in range(1, 12)]
        e = c.task_card_embed(rows)
        assert "11. t11" in e["description"]

    def test_long_title_hard_truncated(self):
        e = c.task_card_embed(
            [{"num": 1, "title": "x" * 500, "state": "open", "due_label": None}])
        line = e["description"].splitlines()[0]
        assert len(line) < 200 and line.endswith("…")


# ===========================================================================
# task_card_embed — 行标题超链接（带 page_id 的行渲染为 masked link）
# ===========================================================================

class TestTaskCardEmbedLinks:
    PID = "a" * 32
    PID2 = "b" * 32

    def test_open_row_title_is_masked_link(self):
        e = c.task_card_embed([
            {"num": 1, "title": "BioRender access", "state": "open",
             "due_label": None, "page_id": self.PID}])
        assert f"1️⃣ [BioRender access](https://www.notion.so/{self.PID})" in e["description"]

    def test_done_and_snoozed_rows_also_link(self):
        e = c.task_card_embed([
            {"num": 1, "title": "A", "state": "done", "due_label": None,
             "page_id": self.PID},
            {"num": 2, "title": "B", "state": "snoozed", "due_label": "07/02 09:30",
             "page_id": self.PID2},
        ])
        assert f"✅ ~~[A](https://www.notion.so/{self.PID})~~" in e["description"]
        assert f"⏰ 已延后·07/02 09:30 · [B](https://www.notion.so/{self.PID2})" in e["description"]

    def test_row_without_page_id_stays_plain(self):
        e = c.task_card_embed(
            [{"num": 1, "title": "A", "state": "open", "due_label": None}])
        assert "1️⃣ A" in e["description"]
        assert "](https://" not in e["description"]

    def test_brackets_in_title_use_fullwidth_inside_link(self):
        # A raw ] would end the masked-link text early; Discord also renders
        # backslash-escapes literally inside link text (shows "\["), so ASCII
        # [] are swapped for fullwidth ［］ (visually equivalent, markdown-inert).
        e = c.task_card_embed([
            {"num": 1, "title": "[邮件][HMS] 检查", "state": "open",
             "due_label": None, "page_id": self.PID}])
        assert (f"1️⃣ [［邮件］［HMS］ 检查](https://www.notion.so/{self.PID})"
                in e["description"])
        assert "\\[" not in e["description"]      # no literal backslash escapes

    def test_link_title_still_truncated(self):
        e = c.task_card_embed(
            [{"num": 1, "title": "x" * 500, "state": "open", "due_label": None,
              "page_id": self.PID}])
        line = e["description"].splitlines()[0]
        assert line.endswith(f"…](https://www.notion.so/{self.PID})")
        assert len(line) < 250


# ===========================================================================
# 编号按钮 payload
# ===========================================================================

class TestNumberedComponents:
    def test_button_component_numbered(self):
        b = c.button_component("done", "a" * 32, 2)
        assert b["label"] == "✓ 2"
        assert b["custom_id"] == f"ntask:done:{'a' * 32}"

    def test_payload_numbers_by_first_occurrence(self):
        p1, p2 = "a" * 32, "b" * 32
        rows = c.components_payload([("done", p1), ("snooze", p1), ("done", p2), ("snooze", p2)])
        labels = [b["label"] for row in rows for b in row["components"]]
        assert labels == ["✓ 1", "⏰ 1", "✓ 2", "⏰ 2"]


# ===========================================================================
# 按钮行打包 — 整任务组装箱到尽量少的行（组不拆分、≤5 按钮/行）
# ===========================================================================

class TestButtonRowPacking:
    def test_two_tasks_share_one_row(self):
        p1, p2 = "a" * 32, "b" * 32
        rows = c.components_payload(
            [("done", p1), ("snooze", p1), ("done", p2), ("snooze", p2)])
        assert len(rows) == 1                       # both tasks' 4 buttons on one row
        assert [b["label"] for b in rows[0]["components"]] == [
            "✓ 1", "⏰ 1", "✓ 2", "⏰ 2"]

    def test_third_task_group_wraps_whole_to_next_row(self):
        pids = ["a" * 32, "b" * 32, "c" * 32]
        tasks = []
        for pid in pids:
            tasks += [("done", pid), ("snooze", pid)]
        rows = c.components_payload(tasks)           # 3×2 = 6 buttons > 5/row
        assert len(rows) == 2
        assert len(rows[0]["components"]) == 4       # tasks 1,2 (4 ≤ 5)
        assert [b["label"] for b in rows[1]["components"]] == [
            "✓ 3", "⏰ 3"]                            # task 3 pair kept intact

    def test_pack_group_rows_first_fit(self):
        assert c.pack_group_rows([2, 2]) == [0, 0]
        assert c.pack_group_rows([2, 2, 2]) == [0, 0, 1]
        assert c.pack_group_rows([1] * 7) == [0, 0, 0, 0, 0, 1, 1]

    def test_pack_group_rows_rejects_oversized_or_overflow(self):
        assert c.pack_group_rows([6]) is None        # one group > 5 per row
        assert c.pack_group_rows([1] * 26) is None    # can't fit in 5 rows
