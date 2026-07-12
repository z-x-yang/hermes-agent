import re

from plugins.platforms.discord.notion_tasks import components as c

PID = "1f17a58d229e816f839bef72f6f2ec72"


def test_make_and_match_custom_id():
    cid = c.make_custom_id("done", PID)
    assert cid == f"ntask:v1:done:{PID}"
    m = re.fullmatch(c.CUSTOM_ID_RE, cid)
    assert m and m.group("action") == "done" and m.group("page_id") == PID

    snooze = c.make_custom_id("snooze", PID)
    m2 = re.fullmatch(c.CUSTOM_ID_RE, snooze)
    assert m2 and m2.group("action") == "snooze" and m2.group("page_id") == PID

    legacy = re.fullmatch(c.CUSTOM_ID_RE, f"ntask:done:{PID}")
    assert legacy and legacy.group("action") == "done"


def test_workbench_custom_ids_include_v1_and_legacy_regex():
    assert c.make_custom_id("open_thread", PID) == f"ntask:v1:open_thread:{PID}"
    assert re.fullmatch(c.CUSTOM_ID_RE, f"ntask:v1:drop:{PID}")
    assert re.fullmatch(c.CUSTOM_ID_RE, f"ntask:done:{PID}")


def test_workbench_action_labels_are_neutral():
    assert c.numbered_label("open_thread", 2) == "🧵2"
    assert c.numbered_label("drop", 2) == "🗑2"
    assert "loot" not in c.numbered_label("drop", 2).lower()


def test_button_component_done_and_undo():
    done = c.button_component("done", PID)
    assert done == {"type": 2, "style": 3, "label": c.LABEL_DONE, "custom_id": f"ntask:v1:done:{PID}"}
    undo = c.button_component("undo", PID)
    assert undo["style"] == 2 and undo["label"] == c.LABEL_UNDO and undo["custom_id"] == f"ntask:v1:undo:{PID}"
    snooze = c.button_component("snooze", PID)
    assert snooze["style"] == 2 and "稍后" in snooze["label"] and snooze["custom_id"] == f"ntask:v1:snooze:{PID}"
    open_thread = c.button_component("open_thread", PID)
    assert open_thread["style"] == 1 and open_thread["custom_id"] == f"ntask:v1:open_thread:{PID}"
    dropped = c.button_component("drop", PID)
    assert dropped["style"] == 4 and dropped["custom_id"] == f"ntask:v1:drop:{PID}"


def test_open_thread_button_uses_link_when_thread_url_known():
    url = "https://discord.com/channels/147/777"

    button = c.button_component("open_thread", PID, num=1, link_url=url)

    assert button == {"type": 2, "style": 5, "label": "🧵1", "url": url}
    assert "custom_id" not in button


def test_components_payload_uses_thread_link_only_for_open_thread():
    url = "https://discord.com/channels/147/777"

    rows = c.components_payload(
        [("open_thread", PID), ("done", PID)],
        link_url_by_page={PID: url},
    )

    open_thread, done = rows[0]["components"]
    assert open_thread == {"type": 2, "style": 5, "label": "🧵1", "url": url}
    assert done["custom_id"] == f"ntask:v1:done:{PID}"


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


def test_action_pairs_for_task_card_caps_full_workbench_actions():
    page_ids = [f"{i:032x}" for i in range(25)]
    pairs = c.action_pairs_with_snooze(page_ids)
    assert len(pairs) == 25
    assert pairs == [
        (action, page_ids[i])
        for i in range(5)
        for action in ("open_thread", "done", "hold", "drop", "snooze")
    ]

    single = c.action_pairs_for_task_card(page_ids[:1])
    assert single == [(action, page_ids[0]) for action in ("open_thread", "done", "hold", "drop", "snooze")]


# ===========================================================================
# numbered_label — 按钮编号标签
# ===========================================================================

class TestNumberedLabel:
    def test_actions_with_number(self):
        assert c.numbered_label("done", 1) == "✓1"
        assert c.numbered_label("snooze", 2) == "⏰2"
        assert c.numbered_label("undo", 3) == "↩ 3"
        assert c.numbered_label("hold", 4) == "⏸4"
        assert c.numbered_label("drop", 5) == "🗑5"

    def test_none_falls_back_to_legacy_constant(self):
        # from_custom_id 重建的按钮无编号信息,label 仅作路由占位、不展示
        assert c.numbered_label("done", None) == c.LABEL_DONE
        assert c.numbered_label("undo", None) == c.LABEL_UNDO
        assert c.numbered_label("snooze", None) == c.LABEL_SNOOZE
        assert c.numbered_label("open_thread", None) == "打开子区"

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
        assert b["label"] == "✓2"
        assert b["custom_id"] == f"ntask:v1:done:{'a' * 32}"

    def test_payload_numbers_by_first_occurrence(self):
        p1, p2 = "a" * 32, "b" * 32
        rows = c.components_payload([("done", p1), ("snooze", p1), ("done", p2), ("snooze", p2)])
        labels = [b["label"] for row in rows for b in row["components"]]
        assert labels == ["✓1", "⏰1", "✓2", "⏰2"]


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
            "✓1", "⏰1", "✓2", "⏰2"]

    def test_third_task_group_wraps_whole_to_next_row(self):
        pids = ["a" * 32, "b" * 32, "c" * 32]
        tasks = []
        for pid in pids:
            tasks += [("done", pid), ("snooze", pid)]
        rows = c.components_payload(tasks)           # 3×2 = 6 buttons > 5/row
        assert len(rows) == 2
        assert len(rows[0]["components"]) == 4       # tasks 1,2 (4 ≤ 5)
        assert [b["label"] for b in rows[1]["components"]] == [
            "✓3", "⏰3"]                            # task 3 pair kept intact

    def test_pack_group_rows_first_fit(self):
        assert c.pack_group_rows([2, 2]) == [0, 0]
        assert c.pack_group_rows([2, 2, 2]) == [0, 0, 1]
        assert c.pack_group_rows([1] * 7) == [0, 0, 0, 0, 0, 1, 1]

    def test_pack_group_rows_rejects_oversized_or_overflow(self):
        assert c.pack_group_rows([6]) is None        # one group > 5 per row
        assert c.pack_group_rows([1] * 26) is None    # can't fit in 5 rows


# ===========================================================================
# Task Clarify Card — 1/2/3 are intelligent choices, routine controls separate
# ===========================================================================

class TestTaskClarifyCard:
    PID = "a" * 32

    def card(self):
        return {
            "notionTaskId": self.PID,
            "notionTaskUrl": f"https://www.notion.so/{self.PID}",
            "notionTaskTitle": "Paper reply task",
            "body": {
                "context": "**这是什么**：合作者回了论文修改意见\n**为什么现在推**：邮件生成了 task\n**不会发生什么**：不会自动发邮件"
            },
            "primaryChoices": [
                {"label": "推荐：先开子区整理上下文", "description": "整理邮件背景、Notion task、关键链接和第一步。"},
                {"label": "先起草回复/材料", "description": "先在子区里起草，不直接发送。"},
                {"label": "先梳理执行图", "description": "整理父子任务、依赖和下一步顺序。"},
            ],
            "otherChoice": {"enabled": True, "label": "Other / 我自己说"},
            "secondaryActions": [
                {"action": "open_thread", "label": "打开/继续子区"},
                {"action": "snooze", "label": "稍后提醒"},
                {"action": "hold", "label": "暂挂"},
                {"action": "drop", "label": "弃置"},
                {"action": "done", "label": "已完成"},
            ],
        }

    def test_embed_puts_long_choice_text_in_body(self):
        embed = c.task_clarify_embed(self.card())
        assert embed["title"] == "🧭 Task Clarify · Paper reply task"
        assert "**这是什么**" in embed["description"]
        assert "1. **推荐：先开子区整理上下文** — 整理邮件背景" in embed["description"]
        assert "2. **先起草回复/材料**" in embed["description"]
        assert "3. **先梳理执行图**" in embed["description"]
        assert "Snooze" not in embed["description"]

    def test_components_keep_primary_buttons_numeric_and_secondary_separate(self):
        rows = c.task_clarify_components(self.card())
        labels = [b["label"] for row in rows for b in row["components"]]
        assert labels[:5] == ["1.", "2.", "3.", "Other", "已接手"]
        assert labels[5:] == ["🧵", "⏰", "⏸", "🗑", "✓"]
        custom_ids = [b["custom_id"] for row in rows for b in row["components"]]
        assert custom_ids[0] == f"ntask:v1:choice1:{self.PID}"
        assert custom_ids[1] == f"ntask:v1:choice2:{self.PID}"
        assert custom_ids[2] == f"ntask:v1:choice3:{self.PID}"
        assert custom_ids[3] == f"ntask:v1:other:{self.PID}"
        assert custom_ids[4] == f"ntask:v1:ack:{self.PID}"
        assert custom_ids[5] == f"ntask:v1:open_thread:{self.PID}"

    def test_components_render_existing_thread_as_same_subthread_link_button(self):
        card = self.card()
        card["threadUrl"] = "https://discord.com/channels/147/999"

        rows = c.task_clarify_components(card)

        thread_button = rows[1]["components"][0]
        assert thread_button["label"] == "🧵"
        assert thread_button["style"] == 5
        assert thread_button["url"] == "https://discord.com/channels/147/999"
        assert "custom_id" not in thread_button
        assert [b["label"] for b in rows[1]["components"]] == ["🧵", "⏰", "⏸", "🗑", "✓"]

    def test_selected_card_removes_primary_buttons_and_shows_choice_state(self):
        card = self.card()
        card["threadUrl"] = "https://discord.com/channels/147/999"
        card["selectedChoice"] = {
            "kind": "choice2",
            "text": "先起草回复/材料 — 先在子区里起草，不直接发送。",
        }
        card["followthroughState"] = "continued"

        embed = c.task_clarify_embed(card)
        rows = c.task_clarify_components(card)

        assert "已选择：先起草回复/材料" in embed["description"]
        assert "状态：已在子区继续" in embed["description"]
        labels = [b["label"] for row in rows for b in row["components"]]
        assert labels == ["🧵", "⏰", "⏸", "🗑", "✓"]
        assert all(not str(b.get("custom_id", "")).startswith(f"ntask:v1:choice")
                   for row in rows for b in row["components"])
        assert all(b.get("custom_id") != f"ntask:v1:other:{self.PID}"
                   for row in rows for b in row["components"])

    def test_terminal_done_drop_and_snooze_cards_collapse_body_and_recolor(self):
        cases = [
            ("done", "完成", 0x4E8CD8, "已完成", "正文已收起，需要恢复可点 ↩。"),
            ("dropped", "弃置", 0x8E9297, "已弃置", "正文已收起，需要恢复可点 ↩。"),
            ("snoozed", "暂挂 / 延后提醒", 0x8E9297, "已暂挂 / 延后提醒", "正文已收起，稍后再看。"),
        ]
        for state, selected, color, status, hint in cases:
            card = self.card()
            card["selectedChoice"] = {"text": selected}
            card["followthroughState"] = state

            embed = c.task_clarify_embed(card)

            assert embed["color"] == color
            assert f"已选择：{selected}" in embed["description"]
            assert f"状态：{status}" in embed["description"]
            assert hint in embed["description"]
            assert "合作者回了论文修改意见" not in embed["description"]
            assert "1. **推荐：先开子区整理上下文**" not in embed["description"]
            assert embed["footer"]["text"] == "已处理；正文已收起"
