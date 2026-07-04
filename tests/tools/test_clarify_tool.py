"""Tests for tools/clarify_tool.py - Interactive clarifying questions."""

import json
from typing import Dict, List, Optional


from tools.clarify_tool import (
    clarify_tool,
    check_clarify_requirements,
    MAX_CHOICES,
    CLARIFY_SCHEMA,
    _normalize_choice,
)


class TestClarifyToolBasics:

    def test_simple_question_with_callback(self):
        def mock_callback(question, choices, context):
            assert question == "What color?"
            assert choices is None
            assert context is None
            return "blue"

        result = json.loads(clarify_tool("What color?", callback=mock_callback))
        assert result["question"] == "What color?"
        assert result["context"] is None
        assert result["choices_offered"] is None
        assert result["user_response"] == "blue"

    def test_question_with_dict_choices(self):
        def mock_callback(question, choices, context):
            assert choices == [
                {"label": "staging", "description": "Deploy to the test cluster"},
                {"label": "prod", "description": "Deploy to production"},
            ]
            return "staging"

        result = json.loads(clarify_tool(
            "Which target?",
            choices=[
                {"label": "staging", "description": "Deploy to the test cluster"},
                {"label": "prod", "description": "Deploy to production"},
            ],
            callback=mock_callback,
        ))
        assert result["choices_offered"] == ["staging", "prod"]
        assert result["user_response"] == "staging"

    def test_context_passed_through(self):
        def mock_callback(question, choices, context):
            assert context == "Two clusters exist; prod deploy is irreversible."
            return "ok"

        result = json.loads(clarify_tool(
            "Which target?",
            context="  Two clusters exist; prod deploy is irreversible.  ",
            callback=mock_callback,
        ))
        assert result["context"] == "Two clusters exist; prod deploy is irreversible."

    def test_blank_context_becomes_none(self):
        def mock_callback(question, choices, context):
            assert context is None
            return "ok"

        result = json.loads(clarify_tool("Q?", context="   ", callback=mock_callback))
        assert result["context"] is None

    def test_empty_question_returns_error(self):
        result = json.loads(clarify_tool("", callback=lambda q, c, x: "ignored"))
        assert "error" in result

    def test_no_callback_returns_error(self):
        result = json.loads(clarify_tool("What do you want?"))
        assert "error" in result
        assert "not available" in result["error"].lower()


class TestClarifyToolChoicesValidation:

    def test_bare_string_choice_becomes_label(self):
        def mock_callback(question, choices, context):
            assert choices == [{"label": "yes", "description": ""}]
            return "yes"

        result = json.loads(clarify_tool("Go?", choices=["yes"], callback=mock_callback))
        assert result["choices_offered"] == ["yes"]

    def test_invalid_choice_shape_fails_fast(self):
        # dict without label / number / nested list → explicit tool_error, no silent drop
        for bad in ([{"description": "no label"}], [42], [["a", "b"]], [None]):
            result = json.loads(clarify_tool("Q?", choices=bad, callback=lambda q, c, x: "r"))
            assert "error" in result, f"expected error for {bad!r}"
            assert "choices[0]" in result["error"]
            assert "label" in result["error"]

    def test_whitespace_only_choice_fails_fast(self):
        result = json.loads(clarify_tool("Q?", choices=["   "], callback=lambda q, c, x: "r"))
        assert "error" in result

    def test_choices_trimmed_to_max(self):
        received = []

        def mock_callback(question, choices, context):
            received.extend(choices or [])
            return "picked"

        clarify_tool("Pick", choices=[f"c{i}" for i in range(7)], callback=mock_callback)
        assert len(received) == MAX_CHOICES

    def test_empty_choices_become_none(self):
        def mock_callback(question, choices, context):
            assert choices is None
            return "answer"

        clarify_tool("Open question?", choices=[], callback=mock_callback)

    def test_choices_not_a_list_returns_error(self):
        result = json.loads(clarify_tool("Q?", choices="not-a-list", callback=lambda q, c, x: "r"))
        assert "error" in result


class TestNormalizeChoice:

    def test_string_becomes_label(self):
        assert _normalize_choice(" go ") == {"label": "go", "description": ""}

    def test_dict_label_description(self):
        assert _normalize_choice({"label": " a ", "description": " b "}) == {
            "label": "a", "description": "b",
        }

    def test_dict_missing_description_defaults_empty(self):
        assert _normalize_choice({"label": "a"}) == {"label": "a", "description": ""}

    def test_dict_without_label_is_invalid(self):
        assert _normalize_choice({"description": "only desc"}) is None

    def test_garbage_shapes_invalid(self):
        for bad in (None, 42, ["a"], {"name": "x", "value": "y"}, "", "   "):
            assert _normalize_choice(bad) is None, f"{bad!r} should be invalid"


class TestClarifySchema:

    def test_schema_name(self):
        assert CLARIFY_SCHEMA["name"] == "clarify"

    def test_schema_has_context_property(self):
        assert "context" in CLARIFY_SCHEMA["parameters"]["properties"]
        assert CLARIFY_SCHEMA["parameters"]["properties"]["context"]["type"] == "string"

    def test_schema_question_required_context_not(self):
        assert CLARIFY_SCHEMA["parameters"]["required"] == ["question"]

    def test_schema_choices_are_objects(self):
        items = CLARIFY_SCHEMA["parameters"]["properties"]["choices"]["items"]
        assert items["type"] == "object"
        assert set(items["required"]) == {"label", "description"}
        assert "label" in items["properties"]
        assert "description" in items["properties"]

    def test_schema_choices_max_items(self):
        assert CLARIFY_SCHEMA["parameters"]["properties"]["choices"]["maxItems"] == MAX_CHOICES

    def test_description_teaches_quality(self):
        desc = CLARIFY_SCHEMA["description"]
        assert "context" in desc
        assert "description" in desc
        # 推荐项前置的指引
        assert "FIRST" in desc or "推荐" in desc


class TestCheckRequirements:
    def test_always_returns_true(self):
        assert check_clarify_requirements() is True
