"""Feishu channel tests -- webhook, streaming filter, image upload, allow_from."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nibot.bus import MessageBus
from nibot.config import FeishuChannelConfig
from nibot.channels.feishu import FeishuChannel
from nibot.types import Envelope


def _make_channel(allow_from: list[str] | None = None) -> FeishuChannel:
    cfg = FeishuChannelConfig(enabled=True, app_id="aid", app_secret="asec")
    if allow_from is not None:
        cfg.allow_from = allow_from
    bus = MagicMock(spec=MessageBus)
    bus.publish_inbound = AsyncMock()
    ch = FeishuChannel(cfg, bus)
    # Mock Feishu client
    ch._client = MagicMock()
    # Mock _send_text to avoid lark_oapi import requirement
    ch._send_text = MagicMock()
    return ch


# ---------------------------------------------------------------------------
# allow_from filtering (via BaseChannel.is_allowed)
# ---------------------------------------------------------------------------

class TestFeishuAllowFrom:

    def test_empty_allow_list_allows_all(self) -> None:
        ch = _make_channel(allow_from=[])
        assert ch.is_allowed("anyone") is True

    def test_allow_list_blocks_unlisted(self) -> None:
        ch = _make_channel(allow_from=["user1"])
        assert ch.is_allowed("user2") is False

    def test_allow_list_allows_listed(self) -> None:
        ch = _make_channel(allow_from=["user1"])
        assert ch.is_allowed("user1") is True


# ---------------------------------------------------------------------------
# Streaming filter
# ---------------------------------------------------------------------------

class TestFeishuStreamingFilter:

    @pytest.mark.asyncio
    async def test_intermediate_stream_chunk_skipped(self) -> None:
        ch = _make_channel()
        envelope = Envelope(
            channel="feishu", chat_id="c1", sender_id="u",
            content="partial...",
            metadata={"streaming": True, "stream_done": False},
        )
        await ch.send(envelope)
        ch._send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_final_stream_chunk_sent(self) -> None:
        ch = _make_channel()
        envelope = Envelope(
            channel="feishu", chat_id="c1", sender_id="u",
            content="final response",
            metadata={"streaming": True, "stream_done": True},
        )
        await ch.send(envelope)
        ch._send_text.assert_called_once_with("c1", "final response")

    @pytest.mark.asyncio
    async def test_non_streaming_sent_normally(self) -> None:
        ch = _make_channel()
        envelope = Envelope(
            channel="feishu", chat_id="c1", sender_id="u",
            content="normal message",
        )
        await ch.send(envelope)
        ch._send_text.assert_called_once_with("c1", "normal message")


# ---------------------------------------------------------------------------
# send() edge cases
# ---------------------------------------------------------------------------

class TestFeishuSend:

    @pytest.mark.asyncio
    async def test_send_when_client_none_does_nothing(self) -> None:
        ch = _make_channel()
        ch._client = None
        envelope = Envelope(channel="feishu", chat_id="c1", sender_id="u", content="hi")
        await ch.send(envelope)  # no crash

    @pytest.mark.asyncio
    async def test_send_empty_content_no_text_call(self) -> None:
        ch = _make_channel()
        envelope = Envelope(channel="feishu", chat_id="c1", sender_id="u", content="")
        await ch.send(envelope)
        ch._send_text.assert_not_called()


# ---------------------------------------------------------------------------
# Webhook handling
# ---------------------------------------------------------------------------

class TestFeishuWebhook:

    @pytest.mark.asyncio
    async def test_challenge_response(self) -> None:
        ch = _make_channel()
        result = await ch.handle_webhook({"challenge": "abc123"})
        assert result == {"challenge": "abc123"}

    @pytest.mark.asyncio
    async def test_message_event_publishes(self) -> None:
        ch = _make_channel()
        body = {
            "event": {
                "message": {
                    "content": '{"text": "hello bot"}',
                    "chat_id": "chat_1",
                },
                "sender": {
                    "sender_id": {"open_id": "user_1"},
                },
            },
        }
        result = await ch.handle_webhook(body)
        assert result == {"code": 0}
        ch.bus.publish_inbound.assert_called_once()
        call_args = ch.bus.publish_inbound.call_args[0][0]
        assert call_args.content == "hello bot"
        assert call_args.sender_id == "user_1"
        assert call_args.chat_id == "chat_1"

    @pytest.mark.asyncio
    async def test_malformed_content_json(self) -> None:
        ch = _make_channel()
        body = {
            "event": {
                "message": {"content": "not json", "chat_id": "c1"},
                "sender": {"sender_id": {"open_id": "u1"}},
            },
        }
        result = await ch.handle_webhook(body)
        assert result == {"code": 0}
        call_args = ch.bus.publish_inbound.call_args[0][0]
        assert call_args.content == "not json"

    @pytest.mark.asyncio
    async def test_webhook_allow_from_blocks(self) -> None:
        ch = _make_channel(allow_from=["allowed_user"])
        body = {
            "event": {
                "message": {"content": '{"text": "hi"}', "chat_id": "c1"},
                "sender": {"sender_id": {"open_id": "blocked_user"}},
            },
        }
        await ch.handle_webhook(body)
        ch.bus.publish_inbound.assert_not_called()
