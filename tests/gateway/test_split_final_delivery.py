from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    FINAL_MESSAGE_SPLIT_MARKER,
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    split_final_response_segments,
)
from gateway.session import SessionSource, build_session_key


SPLIT = FINAL_MESSAGE_SPLIT_MARKER


def test_split_helper_accepts_whitespace_marker_and_ignores_inline_literal():
    assert split_final_response_segments("before\n<!-- HERMES_SPLIT_MESSAGE -->\nafter") == [
        "before",
        "after",
    ]
    assert split_final_response_segments("before\n  <!--   HERMES_SPLIT_MESSAGE   -->  \nafter") == [
        "before",
        "after",
    ]
    inline = "keep <!-- HERMES_SPLIT_MESSAGE --> as literal text"
    assert split_final_response_segments(inline) == [inline]


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self, response: str):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.DISCORD)
        self._response = response
        self.deliveries: list[tuple[str, str]] = []
        self._message_handler = self._handle_message

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.deliveries.append(("send", str(content)))
        return SendResult(success=True, message_id=f"m{len(self.deliveries)}")

    async def get_chat_info(self, chat_id):
        return {}

    async def _handle_message(self, event):
        return self._response

    async def _send_with_retry(
        self,
        chat_id,
        content,
        reply_to=None,
        metadata=None,
        max_retries=2,
        base_delay=2.0,
    ):
        self.deliveries.append(("text", str(content)))
        return SendResult(success=True, message_id=f"m{len(self.deliveries)}")

    async def send_image_file(
        self,
        chat_id,
        image_path,
        caption=None,
        reply_to=None,
        metadata=None,
        **kwargs,
    ):
        self.deliveries.append(("image", str(image_path)))
        return SendResult(success=True, message_id=f"m{len(self.deliveries)}")


def _event() -> MessageEvent:
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="c1",
        chat_type="group",
        thread_id="t1",
    )
    return MessageEvent(
        text="show me the paper",
        message_type=MessageType.TEXT,
        source=source,
        message_id="u1",
    )


@pytest.mark.asyncio
async def test_split_marker_delivers_media_and_text_segments_in_order(tmp_path: Path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"not a real png, only path routing matters")
    response = f"MEDIA:{image}\n{SPLIT}\nFigure 1: caption text"
    adapter = _RecordingAdapter(response)
    event = _event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.deliveries == [
        ("image", str(image)),
        ("text", "Figure 1: caption text"),
    ]


@pytest.mark.asyncio
async def test_split_marker_skips_empty_segments_and_never_renders_marker(tmp_path: Path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"not a real png, only path routing matters")
    response = f"before\n{SPLIT}\n\n{SPLIT}\nMEDIA:{image}\n{SPLIT}\nafter"
    adapter = _RecordingAdapter(response)
    event = _event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.deliveries == [
        ("text", "before"),
        ("image", str(image)),
        ("text", "after"),
    ]
    assert all(SPLIT not in payload for _, payload in adapter.deliveries)
