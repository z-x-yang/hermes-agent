"""Regression tests for the canonical tiktoken-backed estimator.

The estimator counts arbitrary conversation text. Literal special-token
strings such as ``<|endoftext|>`` are data here, not control tokens: a
transcript that *discusses* tokenizers must never crash the estimate.
(2026-07-15: every turn of two production sessions failed with
``ValueError: Encountered text corresponding to disallowed special token``
raised from tiktoken's default ``disallowed_special="all"``.)
"""

import pytest

from agent.token_estimator import (
    count_json_tokens,
    count_text_tokens,
    split_text_for_token_budget,
)
from agent.model_metadata import estimate_messages_tokens_rough

SPECIAL_TOKEN_TEXTS = [
    "<|endoftext|>",
    "prefix <|endoftext|> suffix",
    "<|fim_prefix|>code<|fim_suffix|>",
    "<|endofprompt|>",
]


@pytest.mark.parametrize("text", SPECIAL_TOKEN_TEXTS)
def test_count_text_tokens_special_token_literals(text):
    assert count_text_tokens(text) > 0


def test_count_json_tokens_special_token_literals():
    payload = {"content": "the tokenizer chokes on <|endoftext|> literals"}
    assert count_json_tokens(payload) > 0


def test_split_text_for_token_budget_special_token_literals():
    text = "discussing <|endoftext|> handling " * 500
    result = split_text_for_token_budget(
        text, 100, head_ratio=0.5, tail_ratio=0.5
    )
    assert result is not None
    head, tail, total, head_tokens, tail_tokens = result
    assert total > 100


def test_split_text_for_token_budget_never_decodes_partial_unicode_tokens():
    text = "👩🏽‍💻" * 1_000

    result = split_text_for_token_budget(
        text, 4, head_ratio=0.7, tail_ratio=0.3
    )

    assert result is not None
    head, tail, *_ = result
    assert "�" not in head
    assert "�" not in tail
    assert text.startswith(head)
    assert text.endswith(tail)


def test_estimate_messages_tokens_rough_special_token_literals():
    # The production crash path: a transcript discussing tokenizers.
    messages = [
        {"role": "user", "content": "tiktoken fails on <|endoftext|> — why?"},
        {"role": "assistant", "content": "because <|endoftext|> is special"},
    ]
    assert estimate_messages_tokens_rough(messages) > 0
