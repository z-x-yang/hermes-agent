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
    _clarify_asking_content,
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
# Plain Asking content / notification ping
# ===========================================================================

class TestClarifyAskingContent:
    """The real Discord notification lives in normal message content."""

    def test_mentions_numeric_allowed_user_in_asking_line(self):
        assert _clarify_asking_content({"42"}, has_choices=True) == (
            "Asking <@42> — tap a choice below, or ✏️ Other to type your own answer."
        )

    def test_open_ended_mentions_user_and_prompts_reply(self):
        assert _clarify_asking_content({"42"}, has_choices=False) == (
            "Asking <@42> — reply in this channel with your answer."
        )

    def test_non_numeric_allowlist_falls_back_without_raw_mention(self):
        assert _clarify_asking_content({"*", "alice"}, has_choices=True) == (
            "Asking for input — tap a choice below, or ✏️ Other to type your own answer."
        )


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

    def test_uses_shared_clarify_timeout_config(self, monkeypatch):
        from tools import clarify_gateway as cm

        monkeypatch.setattr(cm, "get_clarify_timeout", lambda: 123)
        view = ClarifyChoiceView(
            choices=["apple"],
            clarify_id="cidTimeout",
            allowed_user_ids=set(),
        )
        assert view.timeout == 123

    def test_nonpositive_shared_clarify_timeout_disables_view_timeout(self, monkeypatch):
        from tools import clarify_gateway as cm

        monkeypatch.setattr(cm, "get_clarify_timeout", lambda: 0)
        view = ClarifyChoiceView(
            choices=["apple"],
            clarify_id="cidNoTimeout",
            allowed_user_ids=set(),
        )
        assert view.timeout is None


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
        cm.register("cidA", "sk-A", "Pick", [
            {"label": "red", "description": ""},
            {"label": "green", "description": ""},
            {"label": "blue", "description": ""},
        ])

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
        # Must be the plain label string -- not a dict, and not a
        # stringified dict (the exact latent bug this test guards against:
        # entry.choices[index] is a {"label","description"} dict since
        # Task 2, and resolve_gateway_clarify str()s whatever it's given).
        assert entry.response == "green"
        assert isinstance(entry.response, str)
        assert entry.response != str({"label": "green", "description": ""})
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
        cm.register("cidC", "sk-C", "Pick", [{"label": "x", "description": ""}])

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
        cm.register("cidD", "sk-D", "Pick", [
            {"label": "x", "description": ""},
            {"label": "y", "description": ""},
        ])

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
        cm.register("cidE", "sk-E", "Pick", [{"label": "x", "description": ""}])

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
    async def test_embed_lists_label_and_description(self):
        adapter = _make_adapter(allowed_users={"42"})
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 9001
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="Which framing?",
            choices=[
                {"label": "Tight and short", "description": "Covers all 3 audiences without dumbing down"},
                {"label": "Long form", "description": "Full detail, slower read"},
            ],
            clarify_id="cidLD",
            session_key="sk-LD",
        )
        embed_text = _embed_text(channel.send.call_args.kwargs["embed"])
        assert "1. **Tight and short** — Covers all 3 audiences without dumbing down" in embed_text
        assert "2. **Long form** — Full detail, slower read" in embed_text
        assert channel.send.call_args.kwargs["content"].startswith("Asking <@42> —")
        assert "allowed_mentions" in channel.send.call_args.kwargs

    @pytest.mark.asyncio
    async def test_open_ended_content_pings_allowed_user(self):
        adapter = _make_adapter(allowed_users={"42"})
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 9004
        channel.send = AsyncMock(return_value=sent_msg)
        assert adapter._client is not None
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="What should I do next?",
            choices=None,
            clarify_id="cidPingOpen",
            session_key="sk-PingOpen",
        )

        assert channel.send.call_args.kwargs["content"] == (
            "Asking <@42> — reply in this channel with your answer."
        )

    @pytest.mark.asyncio
    async def test_context_renders_above_bold_question(self):
        adapter = _make_adapter(allowed_users={"42"})
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 9002
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="Which framing?",
            choices=[{"label": "A", "description": "aa"}],
            clarify_id="cidCtx",
            session_key="sk-Ctx",
            context="Two drafts exist; they differ only in length.",
        )
        desc = channel.send.call_args.kwargs["embed"].description
        assert desc.startswith("Two drafts exist; they differ only in length.")
        assert "**Which framing?**" in desc

    @pytest.mark.asyncio
    async def test_empty_description_renders_label_only(self):
        adapter = _make_adapter(allowed_users={"42"})
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 9003
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        await adapter.send_clarify(
            chat_id="9001",
            question="Go?",
            choices=[{"label": "yes", "description": ""}],
            clarify_id="cidNoDesc",
            session_key="sk-ND",
        )
        embed_text = _embed_text(channel.send.call_args.kwargs["embed"])
        assert "1. **yes**" in embed_text
        assert "1. **yes** —" not in embed_text

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
            choices=[
                {"label": "red", "description": ""},
                {"label": "green", "description": ""},
                {"label": "blue", "description": ""},
            ],
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
            choices=[{"label": "a", "description": ""}],
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
            choices=[{"label": "a", "description": ""}],
            clarify_id="cidNC",
            session_key="sk-NC",
        )
        assert result.success is False
        assert "Not connected" in (result.error or "")

    @pytest.mark.asyncio
    async def test_caps_at_24_real_choices(self):
        # Choices arrive pre-normalized from tools.clarify_tool -- send_clarify
        # no longer filters empty/whitespace entries itself, but it still caps
        # the button+field count at 24 (+ 1 "Other" = 25, Discord's component
        # limit).
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 444
        channel.send = AsyncMock(return_value=sent_msg)
        adapter._client.get_channel = MagicMock(return_value=channel)

        choices = [{"label": f"choice-{i}", "description": ""} for i in range(30)]
        await adapter.send_clarify(
            chat_id="9001",
            question="?",
            choices=choices,
            clarify_id="cidF",
            session_key="sk-F",
        )
        kwargs = channel.send.call_args.kwargs
        view = kwargs["view"]
        # 24 numeric + 1 Other = 25 children
        assert len(view.children) == 25
        assert view.children[0].label == "1."
        assert "choice-0" in _embed_text(kwargs["embed"])

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
            choices=[
                {"label": long_choice, "description": ""},
                {"label": "Short one", "description": ""},
            ],
            clarify_id="cidFull",
            session_key="sk-Full",
        )
        embed_text = _embed_text(channel.send.call_args.kwargs["embed"])
        # Full text present verbatim, numbered, and NOT truncated.
        assert f"1. **{long_choice}**" in embed_text
        assert "2. **Short one**" in embed_text
        assert long_choice + "\u2026" not in embed_text


# ===========================================================================
# _chunk_numbered_choices -- embed field packing
# ===========================================================================

class TestChunkNumberedChoices:
    """Full choice text is packed into embed fields, each <= Discord's 1024
    per-field limit, splitting across fields when needed.

    ``choices`` is a list of pre-normalized ``{"label", "description"}``
    dicts (Task 1's clarify_tool contract) — each line renders as
    ``N. **label** — description``.
    """

    def test_single_field_when_under_limit(self):
        from plugins.platforms.discord.adapter import _chunk_numbered_choices
        choices = [
            {"label": "a", "description": ""},
            {"label": "b", "description": ""},
            {"label": "c", "description": ""},
        ]
        assert _chunk_numbered_choices(choices) == [
            "1. **a**\n2. **b**\n3. **c**"
        ]

    def test_splits_across_fields_when_exceeding_limit(self):
        from plugins.platforms.discord.adapter import _chunk_numbered_choices
        big = "x" * 600  # two "N. **...**" lines can't share one 1024 field
        choices = [{"label": big, "description": ""}, {"label": big, "description": ""}]
        chunks = _chunk_numbered_choices(choices)
        assert len(chunks) == 2
        assert chunks[0].startswith("1. **")
        assert chunks[1].startswith("2. **")
        assert all(len(c) <= 1024 for c in chunks)

    def test_hard_truncates_single_oversized_line(self):
        from plugins.platforms.discord.adapter import _chunk_numbered_choices
        huge = "y" * 2000  # a single option longer than a whole field
        chunks = _chunk_numbered_choices([{"label": huge, "description": ""}])
        assert len(chunks) == 1
        assert len(chunks[0]) <= 1024
        assert chunks[0].endswith("\u2026")

