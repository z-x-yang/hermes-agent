"""Codex Responses event-stream consumption against relay stream mangling.

Production incident 2026-07-03: gptcodex (relay) internally restarts long
requests (double ``response.created``) and then never emits
``response.output_item.done`` for the message item — the full text arrives
only via ``response.output_text.delta`` events, followed by a clean
``response.completed``. ``_consume_codex_event_stream`` kept only the
completed reasoning item in ``final.output``, so the auxiliary adapter
extracted no text and every compression through the relay failed as
"HTTP 200 with no text" — deadlocking sessions that needed compaction.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.codex_runtime import _consume_codex_event_stream
from agent.auxiliary_client import _CodexCompletionsAdapter


def _ev(type_, **fields):
    return SimpleNamespace(type=type_, **fields)


def _completed(status="completed", **resp_fields):
    return _ev(
        "response.completed",
        response=SimpleNamespace(
            id="resp_1", status=status,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
            **resp_fields,
        ),
    )


def _reasoning_item():
    return SimpleNamespace(type="reasoning", content=[])


def _message_item(text):
    return SimpleNamespace(
        type="message", role="assistant", status="completed",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _relay_mangled_stream(deltas):
    """The exact production event sequence from the gptcodex relay."""
    events = [
        _ev("response.created", response=SimpleNamespace(id="resp_0", status="in_progress")),
        _ev("response.created", response=SimpleNamespace(id="resp_1", status="in_progress")),
        _ev("response.output_item.added", item=_reasoning_item()),
        _ev("response.output_item.done", item=_reasoning_item()),
        _ev("response.output_item.added",
            item=SimpleNamespace(type="message", role="assistant", status="in_progress", content=[])),
        # NOTE: no output_item.done for the message item — relay drops it.
    ]
    events += [_ev("response.output_text.delta", delta=d) for d in deltas]
    events.append(_completed())
    return events


class TestConsumerRecoversDroppedMessageItem:
    def test_relay_dropped_message_done_synthesizes_text_item(self):
        deltas = ["## Primary Request", " and Intent\n", "summary body"]
        final = _consume_codex_event_stream(
            iter(_relay_mangled_stream(deltas)), model="gpt-5.5")

        assert final.status == "completed"
        texts = [
            part.text
            for item in final.output
            if getattr(item, "type", None) == "message"
            for part in (getattr(item, "content", None) or [])
            if getattr(part, "type", None) in {"output_text", "text"}
        ]
        assert "".join(texts) == "".join(deltas)

    def test_normal_done_message_item_not_duplicated(self):
        text = "normal summary text"
        events = [
            _ev("response.created", response=SimpleNamespace(id="r", status="in_progress")),
            _ev("response.output_item.added", item=_reasoning_item()),
            _ev("response.output_item.done", item=_reasoning_item()),
            _ev("response.output_text.delta", delta=text),
            _ev("response.output_item.done", item=_message_item(text)),
            _completed(),
        ]
        final = _consume_codex_event_stream(iter(events), model="gpt-5.5")

        texts = [
            part.text
            for item in final.output
            if getattr(item, "type", None) == "message"
            for part in (getattr(item, "content", None) or [])
            if getattr(part, "type", None) in {"output_text", "text"}
        ]
        assert "".join(texts) == text  # not doubled

    def test_tool_call_only_stream_unchanged(self):
        events = [
            _ev("response.created", response=SimpleNamespace(id="r", status="in_progress")),
            _ev("response.output_item.done",
                item=SimpleNamespace(type="function_call", call_id="c1",
                                     name="get_weather", arguments="{}")),
            _completed(),
        ]
        final = _consume_codex_event_stream(iter(events), model="gpt-5.5")
        types = [getattr(i, "type", None) for i in final.output]
        assert types == ["function_call"]


def _make_adapter(events):
    real_client = MagicMock()
    real_client.responses.create.return_value = iter(events)
    return _CodexCompletionsAdapter(real_client, "gpt-5.5")


class TestAuxAdapterTerminalHandling:
    def test_adapter_recovers_text_when_relay_drops_message_done(self):
        deltas = ["part one ", "part two"]
        adapter = _make_adapter(_relay_mangled_stream(deltas))
        resp = adapter.create(messages=[{"role": "user", "content": "summarize"}])
        assert resp.choices[0].message.content == "part one part two"

    def test_adapter_raises_on_terminal_failed(self):
        events = [
            _ev("response.created", response=SimpleNamespace(id="r", status="in_progress")),
            _ev("response.failed",
                response=SimpleNamespace(
                    id="r", status="failed",
                    usage=None,
                    error=SimpleNamespace(code="context_length_exceeded",
                                          message="Your input exceeds the context window."),
                )),
        ]
        adapter = _make_adapter(events)
        with pytest.raises(RuntimeError) as exc_info:
            adapter.create(messages=[{"role": "user", "content": "summarize"}])
        msg = str(exc_info.value)
        assert "context" in msg.lower()
        assert "exceeds" in msg.lower()

    def test_adapter_raises_on_incomplete_with_no_text(self):
        events = [
            _ev("response.created", response=SimpleNamespace(id="r", status="in_progress")),
            _ev("response.output_item.added", item=_reasoning_item()),
            _ev("response.output_item.done", item=_reasoning_item()),
            _ev("response.incomplete",
                response=SimpleNamespace(
                    id="r", status="incomplete",
                    usage=None,
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                )),
        ]
        adapter = _make_adapter(events)
        with pytest.raises(RuntimeError) as exc_info:
            adapter.create(messages=[{"role": "user", "content": "summarize"}])
        assert "max_output_tokens" in str(exc_info.value)

    def test_adapter_returns_partial_text_on_incomplete_with_text(self):
        # Incomplete WITH streamed text keeps existing salvage behavior:
        # the truncated text is returned and the validator/caller decides.
        events = [
            _ev("response.created", response=SimpleNamespace(id="r", status="in_progress")),
            _ev("response.output_text.delta", delta="partial summary"),
            _ev("response.incomplete",
                response=SimpleNamespace(
                    id="r", status="incomplete",
                    usage=None,
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                )),
        ]
        adapter = _make_adapter(events)
        resp = adapter.create(messages=[{"role": "user", "content": "summarize"}])
        assert resp.choices[0].message.content == "partial summary"
