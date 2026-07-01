"""Tests for Discord clarify button rendering and resolution.

Mirrors test_telegram_clarify_buttons.py for the Discord ``send_clarify``
override and the ``ClarifyChoiceView`` callbacks. Discord uses ``discord.ui.View``
button callbacks (closures) rather than a string-prefixed callback_query
dispatcher like Telegram — the auth + resolution path is the same:

  · numeric choice → resolve_gateway_clarify(clarify_id, choice_text)
  · "Other" button → mark_awaiting_text(clarify_id) so the text-intercept
    captures the next user message in this session
  · already-resolved or unauthorized → ephemeral "this prompt..." reply
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Repo root importable
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# Triggers the shared discord mock from tests/gateway/conftest.py before
# importing the production module.
from plugins.platforms.discord.adapter import (  # noqa: E402
    ClarifyChoiceView,
    DiscordAdapter,
)
from gateway.config import PlatformConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(*, allowed_users=None, allowed_roles=None):
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = DiscordAdapter(config)
    adapter._client = MagicMock()
    adapter._allowed_user_ids = set(allowed_users or [])
    adapter._allowed_role_ids = set(allowed_roles or [])
    return adapter


def _clear_clarify_state():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _make_interaction(*, user_id="42", display_name="Tester", roles=None,
                      include_message=True):
    """Build a mock discord.Interaction with response.edit_message /
    send_message / defer all coroutine-callable."""
    user = SimpleNamespace(
        id=user_id,
        display_name=display_name,
        roles=[SimpleNamespace(id=r) for r in (roles or [])],
    )
    response = SimpleNamespace(
        edit_message=AsyncMock(),
        send_message=AsyncMock(),
        defer=AsyncMock(),
    )
    if include_message:
        embed = MagicMock()
        embed.color = None
        embed.set_footer = MagicMock()
        message = SimpleNamespace(embeds=[embed])
    else:
        message = None
    return SimpleNamespace(user=user, response=response, message=message)


def _embed_text(embed) -> str:
    """Flatten a _FakeEmbed (description + every field name/value + footer)
    into one searchable string.

    Full option text now lives in the embed body, not on button labels, so
    choice-text assertions target this instead of ``button.label``.
    """
    parts = [getattr(embed, "description", "") or ""]
    for f in getattr(embed, "fields", []) or []:
        parts.append(f.get("name") or "")
        parts.append(f.get("value") or "")
    footer = getattr(embed, "footer", None)
    if footer:
        parts.append(footer.get("text") or "")
    return "\n".join(parts)


# ===========================================================================
# ClarifyChoiceView construction
# ===========================================================================

class TestClarifyChoiceViewConstruction:
    """The view builds one numeric button per choice plus an Other button.

    Full option text no longer rides on button labels -- Discord caps labels
    at 80 chars and mobile truncates far shorter, which is exactly the bug
    this addresses. Buttons carry only the choice number; send_clarify puts
    the full text in the embed body.
    """

    def test_renders_numeric_buttons_plus_other(self):
        view = ClarifyChoiceView(
            choices=["apple", "banana", "cherry"],
            clarify_id="cidX",
            allowed_user_ids={"42"},
        )
        # 3 numeric + 1 "Other"
        assert len(view.children) == 4
        labels = [b.label for b in view.children]
        assert labels[0] == "1."
        assert labels[1] == "2."
        assert labels[2] == "3."
        assert "Other" in labels[3]
        # custom_ids encode clarify_id + index/other
        ids = [b.custom_id for b in view.children]
        assert ids[0] == "clarify:cidX:0"
        assert ids[1] == "clarify:cidX:1"
        assert ids[2] == "clarify:cidX:2"
        assert ids[3] == "clarify:cidX:other"

    def test_caps_at_24_choices_plus_other(self):
        choices = [f"choice-{i}" for i in range(50)]
        view = ClarifyChoiceView(
            choices=choices,
            clarify_id="cidY",
            allowed_user_ids=set(),
        )
        # Discord limit is 25 components; we cap choices at 24 + 1 Other = 25
        assert len(view.children) == 25
        assert "Other" in view.children[-1].label
        # The 24th numeric button is labelled "24." (still just a number).
        assert view.children[23].label == "24."

    def test_buttons_never_carry_choice_text(self):
        # A long choice must NOT leak onto the button label -- moving full
        # text into the embed is the whole point. The button stays a bare
        # number with no truncation artifact.
        long_choice = "Tight, well-illustrated, covers all 3 audiences"
        view = ClarifyChoiceView(
            choices=[long_choice],
            clarify_id="cidNo",
            allowed_user_ids=set(),
        )
        first_label = view.children[0].label
        assert first_label == "1."
        assert "Tight" not in first_label
        assert "\u2026" not in first_label


# ===========================================================================
# Choice callback → resolve_gateway_clarify
# ===========================================================================

class TestClarifyChoiceResolve:
    """Clicking a numeric button should resolve the clarify entry."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_choice_resolves_with_canonical_choice_text(self):
        from tools import clarify_gateway as cm
        cm.register("cidA", "sk-A", "Pick", ["red", "green", "blue"])

        view = ClarifyChoiceView(
            choices=["red", "green", "blue"],
            clarify_id="cidA",
            allowed_user_ids={"42"},
        )

        interaction = _make_interaction(user_id="42")
        await view._resolve_choice(interaction, index=1, choice="green")

        # Resolved through clarify primitive
        with cm._lock:
            entry = cm._entries.get("cidA")
        assert entry is not None
        assert entry.response == "green"
        assert entry.event.is_set()
        # Buttons disabled
        assert all(b.disabled for b in view.children)
        # Embed updated + edit_message called
        interaction.response.edit_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_choice_falls_back_to_label_text_when_entry_missing(self):
        """If the gateway entry vanished (race / stale view), the button's
        own choice text is used as the response."""
        # Note: no cm.register() — entry intentionally absent

        view = ClarifyChoiceView(
            choices=["alpha"],
            clarify_id="cidGone",
            allowed_user_ids={"42"},  # matches _make_interaction's user; empty = fail-closed
        )
        interaction = _make_interaction()
        # Doesn't raise; resolve_gateway_clarify returns False quietly
        await view._resolve_choice(interaction, index=0, choice="alpha")
        # Still marks the view resolved + disables buttons
        assert view.resolved is True
        assert all(b.disabled for b in view.children)

    @pytest.mark.asyncio
    async def test_already_resolved_sends_ephemeral_reply(self):
        view = ClarifyChoiceView(
            choices=["a", "b"],
            clarify_id="cidB",
            allowed_user_ids=set(),
        )
        view.resolved = True

        interaction = _make_interaction()
        await view._resolve_choice(interaction, index=0, choice="a")

        interaction.response.send_message.assert_called_once()
        kwargs = interaction.response.send_message.call_args.kwargs
        assert kwargs.get("ephemeral") is True
        # No resolve was called
        interaction.response.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected(self):
        from tools import clarify_gateway as cm
        cm.register("cidC", "sk-C", "Pick", ["x"])

        # Allowlist set, user not in it
        view = ClarifyChoiceView(
            choices=["x"],
            clarify_id="cidC",
            allowed_user_ids={"99999"},  # not 42
        )

        interaction = _make_interaction(user_id="42")
        await view._resolve_choice(interaction, index=0, choice="x")

        # Ephemeral rejection, no resolution, no edit
        interaction.response.send_message.assert_called_once()
        kwargs = interaction.response.send_message.call_args.kwargs
        assert kwargs.get("ephemeral") is True
        interaction.response.edit_message.assert_not_called()
        with cm._lock:
            entry = cm._entries.get("cidC")
        assert entry is not None
        assert not entry.event.is_set()


# ===========================================================================
# "Other" button → mark_awaiting_text
# ===========================================================================

class TestClarifyOtherButton:
    """Clicking Other should flip the entry into text-capture mode."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_other_flips_entry_to_awaiting_text(self):
        from tools import clarify_gateway as cm
        cm.register("cidD", "sk-D", "Pick", ["x", "y"])

        view = ClarifyChoiceView(
            choices=["x", "y"],
            clarify_id="cidD",
            allowed_user_ids={"42"},  # matches _make_interaction's user; empty = fail-closed
        )

        interaction = _make_interaction()
        await view._on_other(interaction)

        # Entry awaiting_text now
        pending = cm.get_pending_for_session("sk-D")
        assert pending is not None
        assert pending.clarify_id == "cidD"
        assert pending.awaiting_text is True
        # Entry still pending (not resolved)
        with cm._lock:
            entry = cm._entries.get("cidD")
        assert entry is not None
        assert not entry.event.is_set()
        # View locked + buttons disabled
        assert view.resolved is True
        assert all(b.disabled for b in view.children)
        interaction.response.edit_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_other_unauthorized_user_rejected(self):
        from tools import clarify_gateway as cm
        cm.register("cidE", "sk-E", "Pick", ["x"])

        view = ClarifyChoiceView(
            choices=["x"],
            clarify_id="cidE",
            allowed_user_ids={"99999"},
        )

        interaction = _make_interaction(user_id="42")
        await view._on_other(interaction)

        # Rejected; entry NOT awaiting text
        interaction.response.send_message.assert_called_once()
        pending = cm.get_pending_for_session("sk-E")
        assert pending is None or pending.awaiting_text is False


# ===========================================================================
# DiscordAdapter.send_clarify integration
# ===========================================================================

class TestDiscordSendClarify:
    """Verify send_clarify renders an embed and (optionally) attaches the view."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_multi_choice_attaches_view(self):
        adapter = _make_adapter(allowed_users={"42"})
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 123456
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        result = await adapter.send_clarify(
            chat_id="9001",
            question="Pick a color",
            choices=["red", "green", "blue"],
            clarify_id="cidM",
            session_key="sk-M",
        )

        assert result.success is True
        assert result.message_id == "123456"
        # Verify channel.send was called with embed + view kwargs
        channel.send.assert_called_once()
        kwargs = channel.send.call_args.kwargs
        assert "embed" in kwargs
        assert "view" in kwargs
        assert isinstance(kwargs["view"], ClarifyChoiceView)
        # 3 choice buttons + 1 Other
        assert len(kwargs["view"].children) == 4

    @pytest.mark.asyncio
    async def test_open_ended_omits_view(self):
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 222
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        result = await adapter.send_clarify(
            chat_id="9001",
            question="What is your name?",
            choices=None,
            clarify_id="cidOE",
            session_key="sk-OE",
        )

        assert result.success is True
        channel.send.assert_called_once()
        kwargs = channel.send.call_args.kwargs
        # Open-ended path renders embed but no view (text-capture handles reply)
        assert "embed" in kwargs
        assert "view" not in kwargs

    @pytest.mark.asyncio
    async def test_routes_to_thread_when_metadata_thread_id_set(self):
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 333
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=["a"],
            clarify_id="cidT",
            session_key="sk-T",
            metadata={"thread_id": "7777"},
        )

        # Channel lookup should resolve to thread id, not chat_id
        adapter._client.get_channel.assert_called_once_with(7777)

    @pytest.mark.asyncio
    async def test_not_connected_returns_failure(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=["a"],
            clarify_id="cidNC",
            session_key="sk-NC",
        )
        assert result.success is False
        assert "Not connected" in (result.error or "")

    @pytest.mark.asyncio
    async def test_filters_empty_and_whitespace_choices(self):
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 444
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=["", "  ", "real-choice", None],
            clarify_id="cidF",
            session_key="sk-F",
        )
        kwargs = channel.send.call_args.kwargs
        view = kwargs["view"]
        # Only 1 real choice + 1 Other = 2 children
        assert len(view.children) == 2
        # Button carries the number; the choice text lives in the embed body.
        assert view.children[0].label == "1."
        assert "real-choice" in _embed_text(kwargs["embed"])

    @pytest.mark.asyncio
    async def test_unwraps_dict_choices_to_description(self):
        # LLMs sometimes emit [{"description": "..."}] instead of bare strings
        # — the renderer must unwrap common dict shapes, not str() the whole
        # dict into a Python repr on the button label.
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 555
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        malformed = [
            {"description": "Tight, well-illustrated"},
            {"label": "Use label key"},
            {"text": "Use text key"},
            "normal-string",  # strings still pass through
        ]
        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=malformed,
            clarify_id="cidU",
            session_key="sk-U",
        )
        kwargs = channel.send.call_args.kwargs
        embed_text = _embed_text(kwargs["embed"])
        # No raw Python repr should leak into the embed body.
        assert "{'" not in embed_text
        assert "':" not in embed_text
        # Each dict unwrapped to its inner string, rendered in the embed.
        assert "Tight, well-illustrated" in embed_text
        assert "Use label key" in embed_text
        assert "Use text key" in embed_text
        assert "normal-string" in embed_text
        # Buttons stay numeric (no text on labels).
        view = kwargs["view"]
        numeric = view.children[:-1]  # exclude Other
        assert [b.label for b in numeric] == [f"{k + 1}." for k in range(len(numeric))]

    @pytest.mark.asyncio
    async def test_unwrap_prefers_description_over_name_in_multi_key_dict(self):
        # When the LLM emits both 'name' (often a short identifier in
        # OpenAI-style tool calls) and 'description' (the user-facing text),
        # the renderer must surface 'description'. The user should never see
        # a 4-char model identifier on a button label.
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 666
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=[{"name": "tight", "description": "Tight, well-illustrated"}],
            clarify_id="cidN",
            session_key="sk-N",
        )
        kwargs = channel.send.call_args.kwargs
        embed_text = _embed_text(kwargs["embed"])
        assert "Tight, well-illustrated" in embed_text
        # The 'name' value (a short identifier) must NOT have leaked.
        assert "tight" not in embed_text, f"'name' leaked into embed: {embed_text!r}"

    @pytest.mark.asyncio
    async def test_unwrap_prefers_label_over_description(self):
        # When both 'label' and 'description' are present, 'label' wins.
        # 'label' is the canonical short user-facing text in most LLM tool
        # conventions; 'description' is the longer explanation.
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 777
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=[{"label": "Short", "description": "Long verbose explanation"}],
            clarify_id="cidL",
            session_key="sk-L",
        )
        kwargs = channel.send.call_args.kwargs
        embed_text = _embed_text(kwargs["embed"])
        assert "Short" in embed_text
        # The longer description must NOT have leaked.
        assert "Long verbose" not in embed_text, (
            f"'description' leaked over 'label': {embed_text!r}"
        )

    @pytest.mark.asyncio
    async def test_unwrap_does_not_pick_value_or_name_alone(self):
        # 'name' and 'value' are Discord-component-shaped fields that could
        # accidentally appear in dicts not intended as choices (e.g., a
        # developer-error in the gateway wiring). The renderer should not
        # surface them as button labels — only the well-known LLM tool-call
        # keys (label, description, text, title) should win.
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 888
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=[
                {"name": "only_name_here"},   # should be filtered out
                {"value": "only_value_here"},  # should be filtered out
                {"description": "real choice"},
            ],
            clarify_id="cidNV",
            session_key="sk-NV",
        )
        kwargs = channel.send.call_args.kwargs
        view = kwargs["view"]
        numeric = view.children[:-1]  # exclude Other
        # Only the well-formed dict survives -> 1 numeric button.
        assert len(numeric) == 1, (
            f"Expected 1 choice, got {len(numeric)}: {[b.label for b in numeric]!r}"
        )
        embed_text = _embed_text(kwargs["embed"])
        assert "real choice" in embed_text
        assert "only_name_here" not in embed_text, f"name leaked: {embed_text!r}"
        assert "only_value_here" not in embed_text, f"value leaked: {embed_text!r}"
    @pytest.mark.asyncio
    async def test_embed_lists_full_choice_text_untruncated(self):
        # The whole point of the redesign: a long option appears IN FULL in
        # the embed body, never cut with an ellipsis the way button labels were.
        adapter = _make_adapter(allowed_users={"42"})
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 999
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        long_choice = (
            "Tight, well-illustrated, covers all 3 audiences "
            "(patients, families, curious general readers) without dumbing it down"
        )
        await adapter.send_clarify(
            chat_id="9001",
            question="Which framing?",
            choices=[long_choice, "Short one"],
            clarify_id="cidFull",
            session_key="sk-Full",
        )
        embed_text = _embed_text(channel.send.call_args.kwargs["embed"])
        # Full text present verbatim, numbered, and NOT truncated.
        assert "1. " + long_choice in embed_text
        assert "2. Short one" in embed_text
        assert long_choice + "\u2026" not in embed_text


# ===========================================================================
# _chunk_numbered_choices -- embed field packing
# ===========================================================================

class TestChunkNumberedChoices:
    """Full choice text is packed into embed fields, each <= Discord's 1024
    per-field limit, splitting across fields when needed."""

    def test_single_field_when_under_limit(self):
        from plugins.platforms.discord.adapter import _chunk_numbered_choices
        assert _chunk_numbered_choices(["a", "b", "c"]) == ["1. a\n2. b\n3. c"]

    def test_splits_across_fields_when_exceeding_limit(self):
        from plugins.platforms.discord.adapter import _chunk_numbered_choices
        big = "x" * 600  # two "N. " + 600 lines can't share one 1024 field
        chunks = _chunk_numbered_choices([big, big])
        assert len(chunks) == 2
        assert chunks[0].startswith("1. ")
        assert chunks[1].startswith("2. ")
        assert all(len(c) <= 1024 for c in chunks)

    def test_hard_truncates_single_oversized_line(self):
        from plugins.platforms.discord.adapter import _chunk_numbered_choices
        huge = "y" * 2000  # a single option longer than a whole field
        chunks = _chunk_numbered_choices([huge])
        assert len(chunks) == 1
        assert len(chunks[0]) <= 1024
        assert chunks[0].endswith("\u2026")

