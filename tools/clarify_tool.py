#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions or open-ended
prompts to the user. In CLI mode, choices are navigable with arrow keys. On
messaging platforms, choices are rendered as a numbered list.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.
"""

import json
from typing import Callable, Dict, List, Optional


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4


def _normalize_choice(c) -> Optional[Dict[str, str]]:
    """Coerce one choice into the canonical ``{"label", "description"}`` dict.

    The schema declares choices as objects with a required short ``label``
    (the pickable answer, returned verbatim as the user's response) and a
    one-sentence ``description``. Accepted inputs:

      * dict with a non-empty string ``label`` — canonical form; optional
        ``description`` defaults to "".
      * bare string — degenerate but unambiguous: the string IS the label.

    Anything else (dict without label, number, list, None, blank string)
    returns None so the caller can fail fast with a tool_error naming the
    bad element — the model sees the error and retries with a fixed shape.
    Silently dropping or guessing a label hides the defect from both the
    model and the user.
    """
    if isinstance(c, str):
        label = c.strip()
        return {"label": label, "description": ""} if label else None
    if isinstance(c, dict):
        label = c.get("label")
        if isinstance(label, str) and label.strip():
            desc = c.get("description")
            return {
                "label": label.strip(),
                "description": desc.strip() if isinstance(desc, str) else "",
            }
    return None


def clarify_tool(
    question: str,
    choices: Optional[List] = None,
    context: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> str:
    """
    Ask the user a question, optionally with multiple-choice options.

    Args:
        question: The question text to present (one short sentence).
        choices:  Up to 4 predefined answer choices, each
                  ``{"label": str, "description": str}``. Bare strings are
                  accepted as label-only choices. When omitted the question
                  is purely open-ended.
        context:  1-3 sentences of background shown above the question so
                  the user can answer without having read the agent's
                  reasoning. Optional.
        callback: Platform-provided function that handles the actual UI
                  interaction. Signature:
                  callback(question, choices, context) -> str.
                  Injected by the agent runner (cli.py / gateway).

    Returns:
        JSON string with the user's response.
    """
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()
    context = context.strip() if isinstance(context, str) and context.strip() else None

    # Normalize and validate choices at this single platform-agnostic entry
    # point; every downstream surface (CLI panel, Discord, Telegram,
    # WhatsApp) trusts the {"label", "description"} shape produced here.
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error(
                "choices must be a list of {label, description} objects."
            )
        normalized = []
        for i, c in enumerate(choices):
            nc = _normalize_choice(c)
            if nc is None:
                return tool_error(
                    f"choices[{i}] is invalid: each choice must be an object "
                    f"with a non-empty string 'label' (the short pickable "
                    f"answer) and a one-sentence 'description'. Got: {c!r}"
                )
            normalized.append(nc)
        if len(normalized) > MAX_CHOICES:
            normalized = normalized[:MAX_CHOICES]
        choices = normalized or None  # empty list → open-ended

    if callback is None:
        return json.dumps(
            {"error": "Clarify tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        user_response = callback(question, choices, context)
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    return json.dumps({
        "question": question,
        "context": context,
        "choices_offered": [c["label"] for c in choices] if choices else None,
        "user_response": str(user_response).strip(),
    }, ensure_ascii=False)


def check_clarify_requirements() -> bool:
    """Clarify tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

CLARIFY_SCHEMA = {
    "name": "clarify",
    "description": (
        "Ask the user a question when you need clarification, feedback, or a "
        "decision before proceeding.\n\n"
        "Write for a user who has NOT read your reasoning or tool outputs. "
        "Put the background they need into `context` (1-3 sentences: what "
        "you found or did, why this decision arises now, what actually "
        "differs between the paths). Keep `question` to one short sentence. "
        "Give every choice a short pickable `label` plus a one-sentence "
        "`description` of what it means, what happens if chosen, or its "
        "trade-off — a bare method name or jargon label leaves the user "
        "unable to decide.\n\n"
        "Supports two modes:\n\n"
        "1. **Multiple choice** — provide up to 4 choices. The user picks "
        "one or types their own answer via an auto-appended 'Other' option.\n"
        "2. **Open-ended** — omit choices entirely. The user types a "
        "free-form response.\n\n"
        "If you recommend one option, put it FIRST and start its label with "
        "'推荐: ' / 'Recommended: '.\n\n"
        "CRITICAL: never enumerate the options inside the `question` text — "
        "the UI renders `choices` as selectable rows; options written into "
        "the question string render as dead prose the user can't pick.\n\n"
        "Use this tool when:\n"
        "- The task is ambiguous and you need the user to choose an approach\n"
        "- You want post-task feedback ('How did that work out?')\n"
        "- You want to offer to save a skill or update memory\n"
        "- A decision has meaningful trade-offs the user should weigh in on\n\n"
        "Do NOT use this tool for simple yes/no confirmation of dangerous "
        "commands (the terminal tool handles that). Reserve mid-task "
        "questions for decisions "
        "where the user's answer changes what you do next. For choices with "
        "a conventional default, or facts you can verify yourself, do not "
        "ask: pick the obvious option, state it in your reply, and proceed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "context": {
                "type": "string",
                "description": (
                    "1-3 sentences of background the user needs to answer "
                    "confidently, written for someone who has not seen your "
                    "reasoning: what you found/did, why the decision arises, "
                    "what actually differs between the options. Omit ONLY "
                    "when the question is fully self-explanatory."
                ),
            },
            "question": {
                "type": "string",
                "description": (
                    "The question itself, one short sentence, and ONLY the "
                    "question (e.g. 'Which deployment target?'). Background "
                    "goes in `context`; options go in `choices`."
                ),
            },
            "choices": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": (
                                "Short pickable answer (aim for ≤10 words). "
                                "This exact text is returned as the user's "
                                "answer when picked."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "One sentence: what this option means, what "
                                "happens if chosen, or its trade-off. Do not "
                                "repeat the label."
                            ),
                        },
                    },
                    "required": ["label", "description"],
                },
                "maxItems": MAX_CHOICES,
                "description": (
                    "REQUIRED whenever you are presenting selectable options: "
                    "each option is one {label, description} object (up to "
                    "4). The UI auto-appends an 'Other (type your answer)' "
                    "option. Omit this parameter entirely ONLY for a "
                    "genuinely open-ended free-text question."
                ),
            },
        },
        "required": ["question"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="clarify",
    toolset="clarify",
    schema=CLARIFY_SCHEMA,
    handler=lambda args, **kw: clarify_tool(
        question=args.get("question", ""),
        choices=args.get("choices"),
        context=args.get("context"),
        callback=kw.get("callback")),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
