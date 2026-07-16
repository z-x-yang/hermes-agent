"""Canonical tokenizer-backed counts for runtime size estimates."""

from __future__ import annotations

from functools import lru_cache
import json
from typing import Any

import tiktoken

TOKEN_ESTIMATE_ENCODING = "o200k_base"
IMAGE_TOKEN_ESTIMATE = 1_500
_IMAGE_PART_TYPES = frozenset({"image", "image_url", "input_image"})


@lru_cache(maxsize=1)
def _encoding():
    """Return the process-wide encoding used for deterministic estimates."""
    return tiktoken.get_encoding(TOKEN_ESTIMATE_ENCODING)


def count_text_tokens(text: object) -> int:
    """Count text with the canonical runtime estimate encoding.

    Special-token literals (e.g. ``<|endoftext|>`` appearing in a transcript
    that discusses tokenizers) are data to be counted, not control tokens, so
    the disallowed-special check is disabled — an estimator must be total over
    arbitrary text.
    """
    if text is None or text == "":
        return 0
    return len(_encoding().encode(str(text), disallowed_special=()))


def count_json_tokens(value: Any) -> int:
    """Count a stable compact JSON representation with the canonical encoding."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return count_text_tokens(payload)


def strip_image_payloads_for_token_estimate(value: Any) -> tuple[Any, int]:
    """Return a JSON shadow with image transport data removed plus image count.

    Chat Completions and Responses API image parts use different nesting, but
    both identify the image object with a ``type`` value in
    ``image|image_url|input_image``. Replacing the whole object prevents raw
    base64 from entering the tokenizer while retaining stable structural cost.
    """
    if isinstance(value, dict):
        image_type = str(value.get("type") or "").strip().lower()
        if image_type in _IMAGE_PART_TYPES:
            shadow: dict[str, Any] = {
                "type": value.get("type"),
                "image": "[stripped]",
            }
            detail = value.get("detail")
            if detail not in (None, ""):
                shadow["detail"] = detail
            return shadow, 1

        cleaned: dict[Any, Any] = {}
        image_count = 0
        for key, item in value.items():
            cleaned_item, item_images = strip_image_payloads_for_token_estimate(item)
            cleaned[key] = cleaned_item
            image_count += item_images
        return cleaned, image_count

    if isinstance(value, (list, tuple)):
        cleaned_items: list[Any] = []
        image_count = 0
        for item in value:
            cleaned_item, item_images = strip_image_payloads_for_token_estimate(item)
            cleaned_items.append(cleaned_item)
            image_count += item_images
        return cleaned_items, image_count

    return value, 0


def count_json_tokens_with_images(
    value: Any,
    *,
    image_token_cost: int = IMAGE_TOKEN_ESTIMATE,
) -> int:
    """Count arbitrary request JSON with a fixed charge per image part."""
    cleaned, image_count = strip_image_payloads_for_token_estimate(value)
    return count_json_tokens(cleaned) + image_count * image_token_cost


def split_text_for_token_budget(
    text: str,
    max_tokens: int,
    *,
    head_ratio: float,
    tail_ratio: float,
) -> tuple[str, str, int, int, int] | None:
    """Split oversized text into token-aligned head/tail fragments.

    Returns ``None`` when the text already fits. Otherwise returns
    ``(head, tail, total_tokens, head_tokens, tail_tokens)``.
    """
    tokens = _encoding().encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return None
    head_tokens = int(max_tokens * head_ratio)
    tail_tokens = int(max_tokens * tail_ratio)
    head = _encoding().decode(tokens[:head_tokens])
    tail = _encoding().decode(tokens[-tail_tokens:]) if tail_tokens else ""
    return head, tail, len(tokens), head_tokens, tail_tokens
