"""Small pytest-only evaluator for Hermes nine-section compaction summaries.

This is deliberately not runtime code: it is a lightweight guardrail for tests
and future captured fixtures.  It checks representation invariants that should
hold regardless of which strong LLM produced the summary.
"""

from __future__ import annotations

import re

from collections.abc import Sequence


NINE_SECTION_HEADINGS = [
    "## Primary Request and Intent",
    "## Key Technical Concepts",
    "## Files and Code Sections",
    "## Errors and Fixes",
    "## Problem Solving",
    "## All User Messages",
    "## Pending Tasks",
    "## Current Work",
    "## Optional Next Step",
]

SUMMARY_PREFIX = "[CONTEXT COMPACTION]"
SUMMARY_END_MARKER = "--- END OF COMPACTED CONTEXT ---"
ASSISTANT_TAIL_MARKER = "[RETAINED ASSISTANT CONTINUATION — not user-provided text]"
USER_TAIL_MARKER_PREFIX = "[RETAINED USER CONTINUATION"


def _content_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _line_heading_positions(summary: str, heading: str) -> list[int]:
    return [
        match.start()
        for match in re.finditer(rf"(?m)^{re.escape(heading)}$", summary)
        if not _is_inside_fenced_code_block(summary, match.start())
    ]


def _section(summary: str, heading: str) -> str:
    positions = _line_heading_positions(summary, heading)
    if not positions:
        return ""
    start = positions[0] + len(heading)
    next_starts = [
        pos
        for next_heading in NINE_SECTION_HEADINGS
        if next_heading != heading
        for pos in _line_heading_positions(summary, next_heading)
        if pos > start
    ]
    end = min(next_starts) if next_starts else len(summary)
    return summary[start:end].strip()


def _is_inside_fenced_code_block(text: str, position: int) -> bool:
    fence_char: str | None = None
    fence_len = 0
    for line in text[:position].splitlines():
        match = re.match(r"^[ \t]*(`{3,}|~{3,})", line)
        if not match:
            continue
        fence = match.group(1)
        char = fence[0]
        length = len(fence)
        if fence_char is None:
            fence_char = char
            fence_len = length
        elif char == fence_char and length >= fence_len:
            fence_char = None
            fence_len = 0
    return fence_char is not None


def _looks_like_summary_tail_boundary_remainder(text: str) -> bool:
    remainder = (text or "").lstrip()
    return (
        not remainder
        or remainder.startswith(ASSISTANT_TAIL_MARKER)
        or remainder.startswith(USER_TAIL_MARKER_PREFIX)
    )


def _find_summary_end_marker(text: str) -> int:
    outside_fence: list[int] = []
    boundary_fallbacks: list[int] = []
    start = 0
    while True:
        pos = text.find(SUMMARY_END_MARKER, start)
        if pos < 0:
            break
        if pos == 0 or text[pos - 1] == "\n":
            if not _is_inside_fenced_code_block(text, pos):
                outside_fence.append(pos)
            elif _looks_like_summary_tail_boundary_remainder(
                text[pos + len(SUMMARY_END_MARKER):]
            ) or text[pos + len(SUMMARY_END_MARKER):].lstrip():
                boundary_fallbacks.append(pos)
        start = pos + len(SUMMARY_END_MARKER)
    if outside_fence:
        return min(outside_fence)
    if boundary_fallbacks:
        return max(boundary_fallbacks)
    return -1


def evaluate_nine_section_summary(
    summary: str,
    *,
    user_messages: Sequence[str],
    forbidden_user_attribution_texts: Sequence[str] = (),
) -> list[str]:
    """Return human-readable quality failures for a nine-section summary.

    The checks stay intentionally small: they do not judge prose quality or
    whether a strong model picked the perfect details.  They only protect the
    boundary invariants that caused real compaction bugs.
    """
    failures: list[str] = []

    positions: list[int] = []
    for heading in NINE_SECTION_HEADINGS:
        heading_positions = _line_heading_positions(summary, heading)
        count = len(heading_positions)
        if count != 1:
            failures.append(f"heading {heading!r} appears {count} time(s), expected exactly once")
            continue
        positions.append(heading_positions[0])
    if len(positions) == len(NINE_SECTION_HEADINGS) and positions != sorted(positions):
        failures.append("nine-section headings are not in canonical order")

    all_user = _section(summary, "## All User Messages")
    if user_messages:
        latest = user_messages[-1]
        if "Latest/last user message in compacted range:" not in all_user:
            failures.append("All User Messages must explicitly label the latest/last user message")
        if latest and latest not in all_user:
            failures.append("All User Messages must quote the latest/last user message verbatim")

    for forbidden in forbidden_user_attribution_texts:
        if forbidden and forbidden in all_user:
            failures.append(
                "All User Messages contains forbidden non-user text: "
                f"{forbidden[:80]!r}"
            )

    return failures


def evaluate_compacted_messages(messages: Sequence[dict]) -> list[str]:
    """Check compacted transcript-level attribution boundaries."""
    failures: list[str] = []
    for index, msg in enumerate(messages):
        content = _content_text(msg.get("content"))
        marker_pos = _find_summary_end_marker(content)
        if marker_pos < 0:
            continue
        remainder = content[marker_pos + len(SUMMARY_END_MARKER):].strip()
        if not remainder:
            continue
        role = msg.get("role")
        if role == "assistant" and not remainder.startswith(ASSISTANT_TAIL_MARKER):
            failures.append(
                f"message {index} has retained assistant continuation without role marker"
            )
        elif role == "user" and not remainder.startswith(USER_TAIL_MARKER_PREFIX):
            failures.append(
                f"message {index} has retained user continuation without role marker"
            )

        if role == "assistant" and remainder.startswith(ASSISTANT_TAIL_MARKER):
            tail_text = remainder[len(ASSISTANT_TAIL_MARKER):].lstrip()
        elif role == "user" and remainder.startswith(USER_TAIL_MARKER_PREFIX):
            marker_end = remainder.find("]")
            tail_text = remainder[marker_end + 1:].lstrip() if marker_end >= 0 else remainder
        else:
            tail_text = remainder
        if tail_text.startswith(SUMMARY_PREFIX):
            failures.append(f"message {index} has nested compacted summary inside retained tail")
    return failures
