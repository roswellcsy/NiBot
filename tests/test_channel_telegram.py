"""Telegram channel tests -- streaming, chunking, media, allow_from."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.bus import MessageBus
from nibot.config import TelegramChannelConfig
from nibot.channels.telegram import TelegramChannel, _TG_MAX_LENGTH
from nibot.types import Envelope


def _make_channel(allow_from: list[str] | None = None) -> TelegramChannel:
    cfg = TelegramChannelConfig(enabled=True, token="fake-token")
    if allow_from is not None:
        cfg.allow_from = allow_from
    bus = MagicMock(spec=MessageBus)
    bus.publish_inbound = AsyncMock()
    ch = TelegramChannel(cfg, bus)
    # Mock the Telegram app and bot
    mock_bot = AsyncMock()
    mock_app = MagicMock()
    mock_app.bot = mock_bot
    ch._app = mock_app
    return ch


# ---------------------------------------------------------------------------
# allow_from filtering (via BaseChannel.is_allowed)
# ---------------------------------------------------------------------------

class TestTelegramAllowFrom:

    def test_empty_allow_list_allows_all(self) -> None:
        ch = _make_channel(allow_from=[])
        assert ch.is_allowed("anyone") is True

    def test_allow_list_blocks_unlisted(self) -> None:
        ch = _make_channel(allow_from=["user1"])
        assert ch.is_allowed("user2") is False

    def test_allow_list_allows_listed(self) -> None:
        ch = _make_channel(allow_from=["user1", "user2"])
        assert ch.is_allowed("user1") is True

    def test_pipe_separated_sender_partial_match(self) -> None:
        ch = _make_channel(allow_from=["group1"])
        assert ch.is_allowed("user1|group1") is True


# ---------------------------------------------------------------------------
# send() text chunking
# ---------------------------------------------------------------------------

class TestTelegramSend:

    @pytest.mark.asyncio
    async def test_send_short_text(self) -> None:
        ch = _make_channel()
        envelope = Envelope(channel="telegram", chat_id="123", sender_id="u", content="hello")
        await ch.send(envelope)
        ch._app.bot.send_message.assert_called_once_with(chat_id=123, text="hello")

    @pytest.mark.asyncio
    async def test_send_long_text_chunks(self) -> None:
        ch = _make_channel()
        long_text = "x" * (_TG_MAX_LENGTH + 100)
        envelope = Envelope(channel="telegram", chat_id="123", sender_id="u", content=long_text)
        await ch.send(envelope)
        assert ch._app.bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_send_empty_content_no_call(self) -> None:
        ch = _make_channel()
        envelope = Envelope(channel="telegram", chat_id="123", sender_id="u", content="")
        await ch.send(envelope)
        ch._app.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_when_app_none_does_nothing(self) -> None:
        ch = _make_channel()
        ch._app = None
        envelope = Envelope(channel="telegram", chat_id="123", sender_id="u", content="hi")
        await ch.send(envelope)  # no crash


# ---------------------------------------------------------------------------
# Streaming (edit-in-place)
# ---------------------------------------------------------------------------

class TestTelegramStreaming:

    @pytest.mark.asyncio
    async def test_stream_first_chunk_sends(self) -> None:
        ch = _make_channel()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        ch._app.bot.send_message = AsyncMock(return_value=mock_msg)

        envelope = Envelope(
            channel="telegram", chat_id="123", sender_id="u", content="partial...",
            metadata={"streaming": True, "stream_seq": 0},
        )
        await ch.send(envelope)
        ch._app.bot.send_message.assert_called_once()
        assert ch._stream_msgs[123] == 42

    @pytest.mark.asyncio
    async def test_stream_subsequent_chunk_edits(self) -> None:
        ch = _make_channel()
        ch._stream_msgs[123] = 42  # pre-existing message

        envelope = Envelope(
            channel="telegram", chat_id="123", sender_id="u", content="updated text",
            metadata={"streaming": True, "stream_seq": 1},
        )
        await ch.send(envelope)
        ch._app.bot.edit_message_text.assert_called_once_with(
            chat_id=123, message_id=42, text="updated text",
        )

    @pytest.mark.asyncio
    async def test_stream_done_cleans_up(self) -> None:
        ch = _make_channel()
        ch._stream_msgs[123] = 42

        envelope = Envelope(
            channel="telegram", chat_id="123", sender_id="u", content="final",
            metadata={"streaming": True, "stream_seq": 5, "stream_done": True},
        )
        await ch.send(envelope)
        assert 123 not in ch._stream_msgs

    @pytest.mark.asyncio
    async def test_stream_empty_content_skipped(self) -> None:
        ch = _make_channel()
        envelope = Envelope(
            channel="telegram", chat_id="123", sender_id="u", content="",
            metadata={"streaming": True, "stream_seq": 0},
        )
        await ch.send(envelope)
        ch._app.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_truncates_at_tg_max(self) -> None:
        ch = _make_channel()
        ch._stream_msgs[123] = 42

        long_text = "y" * (_TG_MAX_LENGTH + 500)
        envelope = Envelope(
            channel="telegram", chat_id="123", sender_id="u", content=long_text,
            metadata={"streaming": True, "stream_seq": 2},
        )
        await ch.send(envelope)
        call_args = ch._app.bot.edit_message_text.call_args
        assert len(call_args.kwargs["text"]) == _TG_MAX_LENGTH


# ---------------------------------------------------------------------------
# Media sending
# ---------------------------------------------------------------------------

class TestTelegramMedia:

    @pytest.mark.asyncio
    async def test_send_image(self, tmp_path: Path) -> None:
        ch = _make_channel()
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n")

        envelope = Envelope(
            channel="telegram", chat_id="123", sender_id="u",
            content="", media=[str(img)],
        )
        await ch.send(envelope)
        ch._app.bot.send_photo.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_document(self, tmp_path: Path) -> None:
        ch = _make_channel()
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4")

        envelope = Envelope(
            channel="telegram", chat_id="123", sender_id="u",
            content="", media=[str(doc)],
        )
        await ch.send(envelope)
        ch._app.bot.send_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_missing_media_no_crash(self) -> None:
        ch = _make_channel()
        envelope = Envelope(
            channel="telegram", chat_id="123", sender_id="u",
            content="", media=["/nonexistent/file.png"],
        )
        await ch.send(envelope)
        ch._app.bot.send_photo.assert_not_called()
