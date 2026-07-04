"""Regression tests for final-only reply references in gateway streaming."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class _RecordingAdapter:
    MAX_MESSAGE_LENGTH = 2000
    REQUIRES_EDIT_FINALIZE = False
    SUPPORTS_MESSAGE_EDITING = True

    def __init__(self) -> None:
        self.send_calls = []
        self.edit_calls = []
        self.deleted = []
        self._next_id = 1

    def truncate_message(self, text, max_length=2000, len_fn=None):
        return [text]

    async def send(self, *, chat_id, content, reply_to=None, metadata=None):
        message_id = f"msg_{self._next_id}"
        self._next_id += 1
        self.send_calls.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
                "message_id": message_id,
            }
        )
        return SimpleNamespace(success=True, message_id=message_id)

    async def edit_message(self, **kwargs):
        self.edit_calls.append(kwargs)
        return SimpleNamespace(success=True, message_id=kwargs.get("message_id"))

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))
        return SimpleNamespace(success=True)


@pytest.mark.asyncio
async def test_streaming_preview_is_unreferenced_but_fresh_final_replies_to_user():
    adapter = _RecordingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=0.01,
        buffer_threshold=5,
        cursor="",
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )

    consumer.on_delta("Checking")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    assert adapter.send_calls, "expected an initial streaming preview send"
    preview_call = adapter.send_calls[0]
    assert preview_call["reply_to"] is None
    assert preview_call["metadata"].get("expect_edits") is True
    assert preview_call["metadata"].get("notify") is not True

    consumer.on_delta(" done")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    final_call = adapter.send_calls[-1]
    assert final_call["content"] == "Checking done"
    assert final_call["reply_to"] == "user_msg_1"
    assert final_call["metadata"].get("notify") is True
    assert adapter.deleted == [("channel_1", "msg_1")]
    assert consumer.final_response_sent is True


@pytest.mark.asyncio
async def test_segment_boundary_in_final_reply_mode_does_not_reply_or_notify():
    adapter = _RecordingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=0.01,
        buffer_threshold=5,
        cursor="",
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )

    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Using tools")
    await asyncio.sleep(0.05)
    consumer.on_segment_break()
    await asyncio.sleep(0.05)

    assert adapter.send_calls, "expected the interim segment to become visible"
    for call in adapter.send_calls:
        assert call["reply_to"] is None
        assert (call["metadata"] or {}).get("notify") is not True

    consumer.on_delta("Final answer")
    consumer.finish()
    await task

    final_call = adapter.send_calls[-1]
    assert final_call["content"] == "Final answer"
    assert final_call["reply_to"] == "user_msg_1"
    assert final_call["metadata"].get("notify") is True
    for call in adapter.send_calls[:-1]:
        assert call["reply_to"] is None
        assert (call["metadata"] or {}).get("notify") is not True


@pytest.mark.asyncio
async def test_turn_final_first_send_is_marked_final_without_preview():
    adapter = _RecordingAdapter()
    cfg = StreamConsumerConfig(
        cursor="",
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )

    ok = await consumer._send_or_edit(
        "Final answer",
        finalize=True,
        is_turn_final=True,
    )

    assert ok is True
    assert len(adapter.send_calls) == 1
    final_call = adapter.send_calls[0]
    assert final_call["content"] == "Final answer"
    assert final_call["reply_to"] == "user_msg_1"
    assert final_call["metadata"].get("notify") is True
    assert final_call["metadata"].get("expect_edits") is not True
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    assert consumer.last_delivered_text == "Final answer"


@pytest.mark.asyncio
async def test_oversized_final_with_preview_replies_on_first_final_chunk():
    class _SmallLimitAdapter(_RecordingAdapter):
        MAX_MESSAGE_LENGTH = 600

        def truncate_message(self, text, max_length=600, len_fn=None):
            if len(text) <= max_length:
                return [text]
            return [text[:max_length], text[max_length:]]

    adapter = _SmallLimitAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=0.01,
        buffer_threshold=5,
        cursor="",
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )

    prefix = "Intro section. "
    tail = "A" * 700
    full_final = prefix + tail

    task = asyncio.create_task(consumer.run())
    consumer.on_delta(prefix)
    await asyncio.sleep(0.05)
    assert adapter.send_calls, "expected an initial unreferenced preview"
    assert adapter.send_calls[0]["reply_to"] is None

    consumer.on_delta(tail)
    consumer.finish()
    await task

    final_call = adapter.send_calls[-1]
    assert final_call["content"] == full_final
    assert final_call["reply_to"] == "user_msg_1"
    assert final_call["metadata"].get("notify") is True
    assert adapter.deleted == [("channel_1", "msg_1")]


@pytest.mark.asyncio
async def test_pre_final_overflow_preview_is_replaced_by_full_split_final():
    class _ChunkingAdapter(_RecordingAdapter):
        MAX_MESSAGE_LENGTH = 600

        def truncate_message(self, text, max_length=600, len_fn=None):
            limit = 500
            if len(text) <= limit:
                return [text]
            return [text[:limit], text[limit:]]

    adapter = _ChunkingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=0.01,
        buffer_threshold=5,
        cursor="",
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )

    head = "H" * 700
    tail = "\n\n## Final block\n" + "T" * 80
    full_final = head + tail

    task = asyncio.create_task(consumer.run())
    consumer.on_delta(head)
    await asyncio.sleep(0.05)

    assert len(adapter.send_calls) == 2
    assert all(call["reply_to"] is None for call in adapter.send_calls)
    assert all(call["metadata"].get("notify") is not True for call in adapter.send_calls)

    consumer.on_delta(tail)
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    final_calls = [
        call for call in adapter.send_calls
        if call["metadata"].get("notify") is True
    ]
    assert [call["content"] for call in final_calls] == [full_final]
    assert [call["reply_to"] for call in final_calls] == ["user_msg_1"]
    assert sorted(adapter.deleted) == [("channel_1", "msg_1"), ("channel_1", "msg_2")]
    assert consumer.last_delivered_text == full_final


@pytest.mark.asyncio
async def test_single_final_deletes_stale_preview_after_message_reset():
    adapter = _RecordingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=10,
        buffer_threshold=999999,
        cursor="",
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )
    consumer._preview_message_ids = {"stale_preview"}

    consumer.on_delta("Final answer")
    consumer.finish()
    await consumer.run()

    assert [call["content"] for call in adapter.send_calls] == ["Final answer"]
    assert [call["reply_to"] for call in adapter.send_calls] == ["user_msg_1"]
    assert adapter.deleted == [("channel_1", "stale_preview")]
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    assert consumer._preview_message_ids == set()


@pytest.mark.asyncio
async def test_split_final_deletes_stale_preview_after_message_reset():
    class _ChunkingAdapter(_RecordingAdapter):
        MAX_MESSAGE_LENGTH = 600

        def truncate_message(self, text, max_length=600, len_fn=None):
            limit = 500
            if len(text) <= limit:
                return [text]
            return [text[:limit], text[limit:]]

    adapter = _ChunkingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=10,
        buffer_threshold=999999,
        cursor="",
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )
    consumer._preview_message_ids = {"stale_preview"}

    full_final = "F" * 700
    consumer.on_delta(full_final)
    consumer.finish()
    await consumer.run()

    assert [call["content"] for call in adapter.send_calls] == [
        full_final[:500],
        full_final[500:],
    ]
    assert [call["reply_to"] for call in adapter.send_calls] == [
        "user_msg_1",
        None,
    ]
    assert all(call["metadata"].get("notify") is True for call in adapter.send_calls)
    assert adapter.deleted == [("channel_1", "stale_preview")]
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    assert consumer._preview_message_ids == set()


@pytest.mark.asyncio
async def test_non_final_overflow_preview_created_before_fresh_final_is_cleaned_up():
    """Final-only Discord mode must not leave a pre-final overflow tail visible.

    This reproduces the live last-block duplicate shape: a long streamed answer
    overflows before DONE, creating one or more non-notifying preview fragments;
    the turn-final fresh reply then sends the full answer. Every pre-final
    fragment must be deleted, including fragments produced by the non-turn-final
    overflow/fresh-final path.
    """

    class _ChunkingAdapter(_RecordingAdapter):
        MAX_MESSAGE_LENGTH = 600

        def truncate_message(self, text, max_length=600, len_fn=None):
            # Match the stream consumer's overflow branch: safe limit is
            # MAX_MESSAGE_LENGTH - 100, so 500 chars per preview chunk here.
            limit = 500
            if len(text) <= limit:
                return [text]
            return [text[:limit], text[limit:]]

    adapter = _ChunkingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=0.01,
        buffer_threshold=50,
        cursor="",
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )

    head = "H" * 400
    tail = "\n" + "M" * 220 + "\n\n## Final block\n" + "T" * 40
    full_final = head + tail

    task = asyncio.create_task(consumer.run())
    consumer.on_delta(head)
    await asyncio.sleep(0.05)
    consumer.on_delta(tail)
    await asyncio.sleep(0.05)

    pre_final_ids = {
        call["message_id"]
        for call in adapter.send_calls
        if (call.get("metadata") or {}).get("notify") is not True
    }
    assert pre_final_ids, "expected pre-final preview fragments before DONE"

    consumer.finish()
    await task

    final_calls = [
        call for call in adapter.send_calls
        if (call.get("metadata") or {}).get("notify") is True
    ]
    assert [call["content"] for call in final_calls] == [full_final]
    assert [call["reply_to"] for call in final_calls] == ["user_msg_1"]

    deleted_ids = {message_id for _, message_id in adapter.deleted}
    assert pre_final_ids <= deleted_ids
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    assert consumer.last_delivered_text == full_final


@pytest.mark.asyncio
async def test_oversized_final_without_preview_replies_to_user_only_once():
    class _ChunkingAdapter(_RecordingAdapter):
        MAX_MESSAGE_LENGTH = 550

        def truncate_message(self, text, max_length=4096, len_fn=None):
            return ["chunk1", "chunk2", "chunk3"]

    adapter = _ChunkingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=10,
        buffer_threshold=999999,
        cursor="",
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )

    consumer.on_delta("x" * 600)
    consumer.finish()
    await consumer.run()

    assert [call["content"] for call in adapter.send_calls] == [
        "chunk1",
        "chunk2",
        "chunk3",
    ]
    assert [call["reply_to"] for call in adapter.send_calls] == [
        "user_msg_1",
        None,
        None,
    ]
    assert all(call["metadata"].get("notify") is True for call in adapter.send_calls)
    assert consumer.final_response_sent is True


@pytest.mark.asyncio
async def test_fresh_final_cleanup_cancellation_still_confirms_final_delivery():
    """A sent fresh final must not be duplicated if preview cleanup is cancelled.

    Discord can accept the fresh final reply, then cancel/fail while deleting the
    stale streaming preview.  The normal gateway final-send path must still see
    that the final content already reached the user.
    """

    class _CleanupCancelledAdapter(_RecordingAdapter):
        async def edit_message(self, **kwargs):
            self.edit_calls.append(kwargs)
            return SimpleNamespace(
                success=False,
                message_id=kwargs.get("message_id"),
                error="404 Not Found: Unknown Message",
            )

        async def delete_message(self, chat_id, message_id):
            self.deleted.append((chat_id, message_id))
            raise asyncio.CancelledError()

    adapter = _CleanupCancelledAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=10,
        buffer_threshold=999999,
        cursor="",
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )

    assert await consumer._send_or_edit("Draft", finalize=False)
    consumer._accumulated = "Draft final"
    consumer.finish()
    await consumer.run()

    assert [call["content"] for call in adapter.send_calls] == [
        "Draft",
        "Draft final",
    ]
    assert adapter.send_calls[-1]["reply_to"] == "user_msg_1"
    assert adapter.deleted == [("channel_1", "msg_1")]
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    assert consumer.last_delivered_text == "Draft final"


# ---------------------------------------------------------------------------
# Buffered final-reply mode (the root-cause fix for duplicated last blocks).
#
# In final-only Discord mode the turn-final answer must be a fresh reply to
# carry the @mention.  Streaming it as previews first forces the fresh-final
# re-send to DELETE those previews — and any delete Discord rate-limits or
# drops survives as a duplicated last block (worse the more the answer splits).
# Buffering (buffer_only=True) suppresses the answer previews entirely, so the
# final is the ONLY thing sent: it pings once and there is nothing to clean up,
# so no delete race can ever produce a duplicate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffered_final_sends_no_preview_and_never_needs_delete():
    """A short buffered final: one fresh reply, no preview, no delete."""
    adapter = _RecordingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=0.01,
        buffer_threshold=5,
        cursor="",
        buffer_only=True,
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )

    consumer.on_delta("Checking")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    # Buffered: nothing is streamed to the user before DONE.
    assert adapter.send_calls == []
    assert adapter.edit_calls == []

    consumer.on_delta(" done")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    # Exactly one send — the turn-final answer, replying to the user (ping).
    assert len(adapter.send_calls) == 1
    final_call = adapter.send_calls[0]
    assert final_call["content"] == "Checking done"
    assert final_call["reply_to"] == "user_msg_1"
    assert final_call["metadata"].get("notify") is True
    # No preview existed → the racy cleanup path never runs.
    assert adapter.deleted == []
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_buffered_split_final_pings_only_first_block_and_never_deletes():
    """The reported failure shape: a long final answer that splits into several
    Discord messages.  Only the first block may @ the user, and no block
    requires a delete (there are no previews to clean up)."""

    class _ChunkingAdapter(_RecordingAdapter):
        MAX_MESSAGE_LENGTH = 600

        def truncate_message(self, text, max_length=600, len_fn=None):
            # Match the consumer safe-limit (MAX - cursor - 100 = 500).
            limit = 500
            if len(text) <= limit:
                return [text]
            return [text[:limit], text[limit:]]

    adapter = _ChunkingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=0.01,
        buffer_threshold=50,
        cursor="",
        buffer_only=True,
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "channel_1",
        cfg,
        initial_reply_to_id="user_msg_1",
    )

    task = asyncio.create_task(consumer.run())
    consumer.on_delta("H" * 400)
    await asyncio.sleep(0.05)
    consumer.on_delta("T" * 300)
    await asyncio.sleep(0.05)
    # Still buffered — nothing sent before DONE despite crossing the limit.
    assert adapter.send_calls == []

    consumer.finish()
    await task

    # Two blocks; only the first replies-to-user (the single @mention).
    assert len(adapter.send_calls) == 2
    assert adapter.send_calls[0]["reply_to"] == "user_msg_1"
    assert adapter.send_calls[1]["reply_to"] is None
    # No preview fragments existed → no delete was ever attempted.
    assert adapter.deleted == []
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


# ── flush_barrier: order a direct platform send after buffered content ────
#
# Regression for the clarify/approval ordering bug: a mid-turn tool that
# sends a user-facing message (clarify card, approval prompt) schedules that
# send DIRECTLY on the event loop, while the assistant's preceding interim
# text sits buffered in this consumer and only lands on the consumer's poll
# tick.  In reply_to_mode=final (buffer_only), the text always waits for the
# segment-break flush, so the directly-scheduled card reliably posts ABOVE
# the text.  flush_barrier() lets the caller block until the buffered content
# has actually landed before it sends the card.


@pytest.mark.asyncio
async def test_flush_barrier_fires_only_after_buffered_interim_content_sent():
    adapter = _RecordingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=0.01,
        buffer_threshold=5,
        cursor="",
        buffer_only=True,
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter, "channel_1", cfg, initial_reply_to_id="user_msg_1"
    )

    # Interim assistant text (streamed as deltas) + the mid-turn tool-call
    # segment boundary. Buffered: not sent until the segment break flushes it.
    consumer.on_delta("Here is my recommendation")
    consumer.on_segment_break()
    barrier = consumer.flush_barrier()

    task = asyncio.create_task(consumer.run())

    # Block on the barrier from a worker thread (mirrors _clarify_callback_sync
    # blocking on fut.result before it schedules send_clarify).
    fired = await asyncio.get_running_loop().run_in_executor(None, barrier.wait, 2.0)
    assert fired, "flush_barrier event never fired"

    # When the barrier fires, the buffered interim content MUST already be on
    # the platform — otherwise the caller would send its card above the text.
    assert any(
        "recommendation" in c["content"] for c in adapter.send_calls
    ), "barrier fired before interim content was delivered"

    consumer.finish()
    await task


@pytest.mark.asyncio
async def test_flush_barrier_with_no_pending_content_fires_and_sends_nothing():
    adapter = _RecordingAdapter()
    cfg = StreamConsumerConfig(
        edit_interval=0.01,
        buffer_threshold=5,
        cursor="",
        buffer_only=True,
        reply_to_initial=False,
        force_fresh_final=True,
        fresh_final_reply_to_initial=True,
    )
    consumer = GatewayStreamConsumer(
        adapter, "channel_1", cfg, initial_reply_to_id="user_msg_1"
    )

    barrier = consumer.flush_barrier()
    task = asyncio.create_task(consumer.run())
    fired = await asyncio.get_running_loop().run_in_executor(None, barrier.wait, 2.0)
    assert fired, "flush_barrier event never fired with empty queue"
    # No content queued → nothing sent by the barrier.
    assert adapter.send_calls == []

    consumer.finish()
    await task
