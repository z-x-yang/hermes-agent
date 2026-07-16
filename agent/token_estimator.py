"""Canonical tokenizer-backed counts for runtime size estimates."""

from __future__ import annotations

from functools import lru_cache
import json
from typing import Any

import tiktoken

TOKEN_ESTIMATE_ENCODING = "o200k_base"
IMAGE_TOKEN_ESTIMATE = 1_500
_IMAGE_PART_TYPES = frozenset({"image", "image_url", "input_image"})
_OPAQUE_TRANSPORT_CONTENT_KEYS = frozenset({"encrypted_content"})


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


def strip_opaque_transport_payloads_for_token_estimate(value: Any) -> Any:
    """Return a semantic shadow with opaque transport blobs zeroed.

    Encrypted reasoning is replayed verbatim for provider continuity, but its
    ciphertext bytes are not model-visible text. Tokenizing random ciphertext
    as prose makes context-pressure estimates grow with transport size instead
    of semantic input size. Keep the key/item shape for small structural cost
    while replacing the opaque value with an empty sentinel.
    """
    if isinstance(value, dict):
        return {
            key: (
                ""
                if str(key) in _OPAQUE_TRANSPORT_CONTENT_KEYS
                else strip_opaque_transport_payloads_for_token_estimate(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [strip_opaque_transport_payloads_for_token_estimate(item) for item in value]
    return value


def count_json_tokens_with_images(
    value: Any,
    *,
    image_token_cost: int = IMAGE_TOKEN_ESTIMATE,
) -> int:
    """Count semantic request JSON with fixed image and opaque-blob handling."""
    cleaned, image_count = strip_image_payloads_for_token_estimate(value)
    semantic = strip_opaque_transport_payloads_for_token_estimate(cleaned)
    return count_json_tokens(semantic) + image_count * image_token_cost


def _longest_prefix_within_token_budget(text: str, budget: int) -> str:
    """Return the longest character-aligned prefix whose estimate fits."""
    if budget <= 0 or not text:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count_text_tokens(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _longest_suffix_within_token_budget(text: str, budget: int) -> str:
    """Return the longest character-aligned suffix whose estimate fits."""
    if budget <= 0 or not text:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[len(text) - middle :]
        if count_text_tokens(candidate) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[len(text) - low :] if low else ""


def split_text_for_token_budget(
    text: str,
    max_tokens: int,
    *,
    head_ratio: float,
    tail_ratio: float,
) -> tuple[str, str, int, int, int] | None:
    """Split oversized text into character-aligned head/tail fragments.

    Returns ``None`` when the text already fits. Otherwise returns
    ``(head, tail, total_tokens, head_tokens, tail_tokens)``. Token budgets
    are located on Python string boundaries rather than by decoding arbitrary
    token slices, which can otherwise synthesize U+FFFD replacement text.
    """
    total_tokens = count_text_tokens(text)
    if total_tokens <= max_tokens:
        return None
    head_budget = int(max_tokens * head_ratio)
    tail_budget = int(max_tokens * tail_ratio)
    head = _longest_prefix_within_token_budget(text, head_budget)
    remaining = text[len(head) :]
    tail = _longest_suffix_within_token_budget(remaining, tail_budget)
    head_tokens = count_text_tokens(head)
    tail_tokens = count_text_tokens(tail)
    return head, tail, total_tokens, head_tokens, tail_tokens
