"""Tests for the gateway-side clarify primitive (tools/clarify_gateway.py).

The clarify tool needs to ask the user a question and block the agent
thread until they respond.  These tests cover the module-level state
machine: register, wait, resolve via button, resolve via text-fallback,
"Other"-button text-capture flip, timeout, session boundary cleanup.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor



def _clear_clarify_state():
    """Reset module-level state between tests."""
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


class TestClarifyPrimitive:
    """Core register/wait/resolve mechanics."""

    def setup_method(self):
        _clear_clarify_state()

    def test_button_choice_resolves_wait(self):
        """resolve_gateway_clarify unblocks wait_for_response with the chosen string."""
        from tools import clarify_gateway as cm

        cm.register("id1", "sk1", "Pick one", [{"label": "A", "description": ""}, {"label": "B", "description": ""}, {"label": "C", "description": ""}])

        def resolver():
            time.sleep(0.05)
            cm.resolve_gateway_clarify("id1", "B")

        threading.Thread(target=resolver).start()
        result = cm.wait_for_response("id1", timeout=2.0)
        assert result == "B"

    def test_open_ended_auto_awaits_text(self):
        """Clarify with no choices is in text-capture mode immediately."""
        from tools import clarify_gateway as cm

        entry = cm.register("id2", "sk2", "Free form?", None)
        assert entry.awaiting_text is True

        # get_pending_for_session returns the entry so the gateway
        # text-intercept can find it.
        pending = cm.get_pending_for_session("sk2")
        assert pending is not None
        assert pending.clarify_id == "id2"

    def test_button_choice_does_not_auto_await(self):
        """Multi-choice clarify should NOT be in text-capture mode initially."""
        from tools import clarify_gateway as cm

        entry = cm.register("id3", "sk3", "Pick", [{"label": "X", "description": ""}, {"label": "Y", "description": ""}])
        assert entry.awaiting_text is False
        assert cm.get_pending_for_session("sk3") is None

    def test_include_choice_prompts_returns_multi_choice_entry(self):
        """Gateway typed replies must see active choice prompts too."""
        from tools import clarify_gateway as cm

        cm.register("id3b", "sk3b", "Pick", [{"label": "X", "description": ""}, {"label": "Y", "description": ""}])
        pending = cm.get_pending_for_session("sk3b", include_choice_prompts=True)
        assert pending is not None
        assert pending.clarify_id == "id3b"

    def test_resolve_text_response_maps_numeric_choice(self):
        """Typed numbers should resolve to the canonical choice string."""
        from tools import clarify_gateway as cm

        cm.register("id3c", "sk3c", "Pick", [{"label": "X", "description": ""}, {"label": "Y", "description": ""}])
        assert cm.resolve_text_response_for_session("sk3c", "2") is True
        assert cm.wait_for_response("id3c", timeout=0.1) == "Y"

    def test_resolve_text_response_accepts_custom_other_text(self):
        """Arbitrary typed text should resolve as a custom Other answer."""
        from tools import clarify_gateway as cm

        cm.register("id3d", "sk3d", "Pick", [{"label": "X", "description": ""}, {"label": "Y", "description": ""}])
        custom = "None of those are valid options"
        assert cm.resolve_text_response_for_session("sk3d", custom) is True
        assert cm.wait_for_response("id3d", timeout=0.1) == custom

    def test_other_button_flips_to_text_mode(self):
        """mark_awaiting_text makes get_pending_for_session find the entry."""
        from tools import clarify_gateway as cm

        cm.register("id4", "sk4", "Pick", [{"label": "X", "description": ""}, {"label": "Y", "description": ""}])
        assert cm.get_pending_for_session("sk4") is None

        flipped = cm.mark_awaiting_text("id4")
        assert flipped is True

        pending = cm.get_pending_for_session("sk4")
        assert pending is not None
        assert pending.clarify_id == "id4"

    def test_mark_awaiting_text_unknown_id(self):
        """mark_awaiting_text on a non-existent id returns False."""
        from tools import clarify_gateway as cm

        assert cm.mark_awaiting_text("nope") is False

    def test_timeout_returns_none(self):
        """wait_for_response returns None when no resolve fires within the timeout."""
        from tools import clarify_gateway as cm

        cm.register("id5", "sk5", "Q?", [{"label": "A", "description": ""}])
        result = cm.wait_for_response("id5", timeout=0.2)
        assert result is None

    def test_nonpositive_timeout_waits_indefinitely_until_resolved(self):
        """timeout <= 0 disables the clarify timeout instead of expiring immediately."""
        from tools import clarify_gateway as cm

        cm.register("id5b", "sk5b", "Q?", [{"label": "A", "description": ""}])

        def resolver():
            time.sleep(0.05)
            cm.resolve_gateway_clarify("id5b", "A")

        threading.Thread(target=resolver).start()
        result = cm.wait_for_response("id5b", timeout=0)
        assert result == "A"

    def test_resolve_unknown_id_returns_false(self):
        """resolve_gateway_clarify is idempotent on unknown ids."""
        from tools import clarify_gateway as cm

        assert cm.resolve_gateway_clarify("nope", "anything") is False

    def test_resolve_after_wait_completes_is_noop(self):
        """A late resolve on a finished entry doesn't blow up."""
        from tools import clarify_gateway as cm

        cm.register("id6", "sk6", "Q?", [{"label": "A", "description": ""}])
        # Time out, entry gets cleaned up
        cm.wait_for_response("id6", timeout=0.1)
        # Late button click — should not raise
        result = cm.resolve_gateway_clarify("id6", "A")
        assert result is False

    def test_clear_session_cancels_pending_entries(self):
        """clear_session unblocks blocked threads with empty response."""
        from tools import clarify_gateway as cm

        cm.register("id7", "sk7", "Q?", [{"label": "A", "description": ""}])

        def waiter():
            return cm.wait_for_response("id7", timeout=10.0)

        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(waiter)
            time.sleep(0.05)
            cancelled = cm.clear_session("sk7")
            assert cancelled == 1
            result = fut.result(timeout=2.0)
            # clear_session sets response="" then the wait returns it
            assert result == ""

    def test_has_pending(self):
        from tools import clarify_gateway as cm

        cm.register("id8", "sk8", "Q?", [{"label": "A", "description": ""}])
        assert cm.has_pending("sk8") is True
        assert cm.has_pending("nonexistent") is False

    def test_notify_register_unregister_clears_pending(self):
        """unregister_notify cancels any pending clarify so threads unwind."""
        from tools import clarify_gateway as cm

        cm.register("id9", "sk9", "Q?", [{"label": "A", "description": ""}])

        def waiter():
            return cm.wait_for_response("id9", timeout=10.0)

        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(waiter)
            time.sleep(0.05)

            cm.register_notify("sk9", lambda entry: None)
            cm.unregister_notify("sk9")

            # unregister_notify calls clear_session; thread unwinds
            result = fut.result(timeout=2.0)
            assert result == ""

    def test_session_index_isolation(self):
        """Entries from different sessions don't leak across get_pending lookups."""
        from tools import clarify_gateway as cm

        cm.register("idA", "alpha", "Q?", None)  # auto-await text
        cm.register("idB", "beta", "Q?", None)   # auto-await text

        a = cm.get_pending_for_session("alpha")
        b = cm.get_pending_for_session("beta")
        assert a is not None and a.clarify_id == "idA"
        assert b is not None and b.clarify_id == "idB"

    def test_clarify_timeout_config_default(self):
        """get_clarify_timeout returns a positive int (default 3600)."""
        from tools import clarify_gateway as cm

        timeout = cm.get_clarify_timeout()
        # Default 3600s OR whatever is in the user's loaded config.
        # 0 is allowed and means "never time out".
        assert isinstance(timeout, int)
        assert timeout >= 0


class TestGatewayTextIntercept:
    """The gateway's _handle_message intercepts text replies to pending clarifies."""

    def setup_method(self):
        _clear_clarify_state()

    def test_get_pending_for_session_returns_oldest_text_awaiting(self):
        """When two clarifies are pending, get_pending_for_session returns the
        first that is awaiting_text (the older one if both)."""
        from tools import clarify_gateway as cm

        # Older multi-choice (not awaiting text)
        cm.register("first", "sk", "Q1?", [{"label": "A", "description": ""}])
        # Newer open-ended (awaiting text)
        cm.register("second", "sk", "Q2?", None)

        pending = cm.get_pending_for_session("sk")
        # The newer one is awaiting text; the older isn't.
        assert pending is not None
        assert pending.clarify_id == "second"

        # Now flip the first to text mode too.  Both are awaiting text,
        # FIFO returns the older one.
        cm.mark_awaiting_text("first")
        pending2 = cm.get_pending_for_session("sk")
        assert pending2 is not None
        assert pending2.clarify_id == "first"
    def test_text_fallback_enables_awaiting_text_for_multi_choice(self):
        """When base send_clarify renders choices as text, mark_awaiting_text
        is called so the gateway text-intercept can capture the reply."""
        from tools import clarify_gateway as cm

        entry = cm.register("id-tf", "sk-tf", "Pick one", [{"label": "A", "description": ""}, {"label": "B", "description": ""}, {"label": "C", "description": ""}])
        # Initially, multi-choice does NOT await text (button path)
        assert entry.awaiting_text is False

        # After the base send_clarify text fallback calls mark_awaiting_text:
        flipped = cm.mark_awaiting_text("id-tf")
        assert flipped is True

        # Now get_pending_for_session should find it
        pending = cm.get_pending_for_session("sk-tf")
        assert pending is not None
        assert pending.clarify_id == "id-tf"
        
        # Clean up
        cm.clear_session("sk-tf")


def test_entry_stores_context_and_dict_choices():
    from tools import clarify_gateway as cg
    entry = cg.register(
        clarify_id="t-ctx1",
        session_key="sess-ctx1",
        question="Q?",
        choices=[{"label": "a", "description": "aa"}],
        context="Some background.",
    )
    try:
        assert entry.context == "Some background."
        assert entry.choices == [{"label": "a", "description": "aa"}]
        assert entry.signature()["context"] == "Some background."
    finally:
        cg.clear_session("sess-ctx1")


def test_coerce_numeric_reply_returns_label():
    from tools import clarify_gateway as cg
    entry = cg.register(
        clarify_id="t-num1",
        session_key="sess-num1",
        question="Q?",
        choices=[
            {"label": "staging", "description": "test cluster"},
            {"label": "prod", "description": "production"},
        ],
    )
    try:
        assert cg._coerce_text_response(entry, "2") == "prod"
        assert cg._coerce_text_response(entry, "PROD") == "prod"
        assert cg._coerce_text_response(entry, "my own answer") == "my own answer"
    finally:
        cg.clear_session("sess-num1")
