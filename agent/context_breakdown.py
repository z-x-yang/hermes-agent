"""Live session context-window breakdown for UI surfaces.

Estimates how the next provider request is composed: system prompt tiers,
tool schemas, and conversation history. Uses the canonical ``o200k_base``
runtime estimator so UI categories align with compression thresholds.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)

_SUBAGENT_TOOL_NAMES = frozenset({"delegate_task"})

_CATEGORY_COLORS = {
    "system_prompt": "var(--context-usage-system)",
    "tool_definitions": "var(--context-usage-tools)",
    "rules": "var(--context-usage-rules)",
    "skills": "var(--context-usage-skills)",
    "mcp": "var(--context-usage-mcp)",
    "subagent_definitions": "var(--context-usage-subagents)",
    "memory": "var(--context-usage-memory)",
    "conversation": "var(--context-usage-conversation)",
}


def _chars_to_tokens(text: str) -> int:
    from agent.token_estimator import count_text_tokens

    return count_text_tokens(text)


def _json_tokens(value: Any) -> int:
    if not value:
        return 0
    from agent.token_estimator import count_json_tokens

    return count_json_tokens(value)


def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(tool.get("name") or "")


def _split_tools(tools: Sequence[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    builtin: List[dict] = []
    mcp: List[dict] = []
    subagent: List[dict] = []
    for tool in tools:
        name = _tool_name(tool)
        if name.startswith("mcp_"):
            mcp.append(tool)
        elif name in _SUBAGENT_TOOL_NAMES:
            subagent.append(tool)
        else:
            builtin.append(tool)
    return builtin, mcp, subagent


def _memory_blocks(agent: Any) -> Tuple[str, str]:
    memory_block = ""
    user_block = ""
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return memory_block, user_block
    try:
        if getattr(agent, "_memory_enabled", True):
            memory_block = store.format_for_system_prompt("memory") or ""
        if getattr(agent, "_user_profile_enabled", True):
            user_block = store.format_for_system_prompt("user") or ""
    except Exception:
        pass
    return memory_block, user_block


def _strip_blocks(text: str, *blocks: str) -> str:
    out = text
    for block in blocks:
        if block:
            out = out.replace(block, "")
    return out.strip()


def compute_session_context_breakdown(
    agent: Any,
    messages: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Return a Cursor-style context usage breakdown for one live agent."""
    from agent.model_metadata import estimate_messages_tokens_rough
    from agent.system_prompt import build_system_prompt_parts

    parts = build_system_prompt_parts(agent)
    stable = parts.get("stable", "") or ""
    context = parts.get("context", "") or ""
    volatile = parts.get("volatile", "") or ""

    skills_match = _SKILLS_BLOCK_RE.search(stable)
    skills_index = skills_match.group(0) if skills_match else ""

    memory_block, user_block = _memory_blocks(agent)
    memory_text = "\n\n".join(part for part in (memory_block, user_block) if part).strip()

    system_core = _strip_blocks(stable, skills_index)
    system_tail = _strip_blocks(volatile, memory_block, user_block)
    system_prompt_text = "\n\n".join(part for part in (system_core, system_tail) if part).strip()

    tools = list(getattr(agent, "tools", None) or [])
    builtin_tools, mcp_tools, subagent_tools = _split_tools(tools)

    comp = getattr(agent, "context_compressor", None)
    provider_estimator = getattr(comp, "estimate_provider_messages_tokens", None)
    estimated_conversation_tokens: int
    if callable(provider_estimator):
        try:
            provider_estimate = provider_estimator(messages or [])
        except Exception:
            provider_estimate = None
        if isinstance(provider_estimate, (int, float)) and not isinstance(provider_estimate, bool):
            estimated_conversation_tokens = max(0, int(provider_estimate))
        else:
            estimated_conversation_tokens = estimate_messages_tokens_rough(messages or [])
    else:
        estimated_conversation_tokens = estimate_messages_tokens_rough(messages or [])

    categories = [
        ("system_prompt", "System prompt", _chars_to_tokens(system_prompt_text)),
        ("tool_definitions", "Tool definitions", _json_tokens(builtin_tools)),
        ("rules", "Rules", _chars_to_tokens(context)),
        ("skills", "Skills", _chars_to_tokens(skills_index)),
        ("mcp", "MCP", _json_tokens(mcp_tools)),
        ("subagent_definitions", "Subagent definitions", _json_tokens(subagent_tools)),
        ("memory", "Memory", _chars_to_tokens(memory_text)),
        ("conversation", "Conversation", estimated_conversation_tokens),
    ]

    context_max = int(
        getattr(
            comp,
            "compression_context_length",
            getattr(comp, "context_length", 0),
        ) or 0
    ) if comp else 0
    measured_used = int(getattr(comp, "last_prompt_tokens", 0) or 0) if comp else 0
    if measured_used > 0:
        # ``last_prompt_tokens`` is the provider-observed request size that
        # /usage shows as the current context. Category buckets are independent
        # ``o200k_base`` estimates and can still overshoot provider accounting,
        # especially after compression. Keep the user-facing breakdown on the
        # same basis by treating conversation as the measured residual after
        # categories, rather than showing a conversation slice larger than the
        # displayed total context.
        non_conversation_total = sum(
            tokens for category_id, _label, tokens in categories
            if category_id != "conversation"
        )
        if non_conversation_total <= measured_used:
            conversation_tokens = measured_used - non_conversation_total
            categories = [
                (category_id, label, conversation_tokens if category_id == "conversation" else tokens)
                for category_id, label, tokens in categories
            ]
        else:
            scaled_tokens: list[int] = []
            fractions: list[tuple[float, int]] = []
            for index, (category_id, _label, tokens) in enumerate(categories):
                if category_id == "conversation" or tokens <= 0:
                    scaled_tokens.append(0)
                    continue
                exact = tokens * measured_used / non_conversation_total
                whole = int(exact)
                scaled_tokens.append(whole)
                fractions.append((exact - whole, index))

            remainder = measured_used - sum(scaled_tokens)
            for _fraction, index in sorted(fractions, reverse=True)[:remainder]:
                scaled_tokens[index] += 1

            categories = [
                (category_id, label, scaled_tokens[index])
                for index, (category_id, label, _tokens) in enumerate(categories)
            ]

    estimated_total = sum(tokens for _, _, tokens in categories)
    context_used = measured_used if measured_used > 0 else estimated_total
    context_percent = (
        max(0, min(100, round(context_used / context_max * 100)))
        if context_max
        else 0
    )

    return {
        "categories": [
            {
                "color": _CATEGORY_COLORS.get(category_id, "var(--ui-text-tertiary)"),
                "id": category_id,
                "label": label,
                "tokens": tokens,
            }
            for category_id, label, tokens in categories
            if tokens > 0
        ],
        "context_max": context_max,
        "context_percent": context_percent,
        "context_used": context_used,
        "estimated_total": estimated_total,
        "model": getattr(agent, "model", "") or "",
    }
