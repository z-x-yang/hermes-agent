from types import SimpleNamespace

from gateway.config import Platform
from gateway.run import (
    _stream_consumer_buffers_answer,
    _stream_consumer_reply_options,
)


def test_discord_final_reply_mode_forces_fresh_final_only():
    adapter = SimpleNamespace(_reply_to_mode="final")

    assert _stream_consumer_reply_options(Platform.DISCORD, adapter) == {
        "reply_to_initial": False,
        "force_fresh_final": True,
        "fresh_final_reply_to_initial": True,
    }


def test_discord_default_reply_mode_keeps_preview_reply_policy():
    adapter = SimpleNamespace(_reply_to_mode="first")

    assert _stream_consumer_reply_options(Platform.DISCORD, adapter) == {
        "reply_to_initial": True,
        "force_fresh_final": False,
        "fresh_final_reply_to_initial": False,
    }


def test_final_reply_string_does_not_change_non_discord_platforms():
    adapter = SimpleNamespace(_reply_to_mode="final")

    assert _stream_consumer_reply_options(Platform.TELEGRAM, adapter) == {
        "reply_to_initial": True,
        "force_fresh_final": False,
        "fresh_final_reply_to_initial": False,
    }


def test_discord_final_reply_mode_buffers_the_answer():
    """Final-only Discord replies must buffer instead of streaming previews.

    The turn-final answer has to be a fresh reply so it can carry the @mention;
    streaming it as previews first forces the fresh-final re-send to delete
    those previews, and any delete Discord rate-limits/drops survives as a
    duplicated last block.  Buffering removes the preview path entirely.
    """
    adapter = SimpleNamespace(_reply_to_mode="final")

    assert _stream_consumer_buffers_answer(
        Platform.DISCORD, adapter, base_buffer_only=False
    ) is True


def test_discord_default_reply_mode_keeps_streaming_previews():
    adapter = SimpleNamespace(_reply_to_mode="first")

    assert _stream_consumer_buffers_answer(
        Platform.DISCORD, adapter, base_buffer_only=False
    ) is False


def test_final_reply_string_does_not_buffer_non_discord_platforms():
    adapter = SimpleNamespace(_reply_to_mode="final")

    assert _stream_consumer_buffers_answer(
        Platform.TELEGRAM, adapter, base_buffer_only=False
    ) is False


def test_base_buffer_only_is_preserved_regardless_of_platform():
    """Matrix (and any platform) already-decided buffer_only must not be lost."""
    adapter = SimpleNamespace(_reply_to_mode="first")

    assert _stream_consumer_buffers_answer(
        Platform.MATRIX, adapter, base_buffer_only=True
    ) is True
