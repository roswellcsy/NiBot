"""Tests for v1.0.0: WeCom + API channels + webhook server."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.bus import MessageBus
from nibot.channels.api import APIChannel
from nibot.channels.wecom import WeComChannel
from nibot.types import Envelope


# ---- Bus response waiter ----


class TestBusResponseWaiter:

    @pytest.mark.asyncio
    async def test_create_and_resolve_waiter(self) -> None:
        bus = MessageBus()
        key, future = bus.create_response_waiter(timeout=5.0)
        assert key.startswith("_response_")
        assert not future.done()

        envelope = Envelope(
            channel="api", chat_id="test", sender_id="bot", content="hello"
        )
        resolved = bus.resolve_response(key, envelope)
        assert resolved
        assert future.done()
        assert (await future).content == "hello"

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_key(self) -> None:
        bus = MessageBus()
        resolved = bus.resolve_response(
            "nonexistent",
            Envelope(channel="", chat_id="", sender_id="", content=""),
        )
        assert not resolved

    @pytest.mark.asyncio
    async def test_dispatch_outbound_skips_waiter_messages(self) -> None:
        bus = MessageBus()
        key, future = bus.create_response_waiter(timeout=5.0)

        # Put a message with response_key
        await bus.publish_outbound(
            Envelope(
                channel="api",
                chat_id="test",
                sender_id="bot",
                content="response",
                metadata={"response_key": key},
            )
        )

        # Start dispatch in background
        dispatch_task = asyncio.create_task(bus.dispatch_outbound())
        await asyncio.sleep(0.2)
        bus.stop()
        await asyncio.sleep(0.1)
        dispatch_task.cancel()
        try:
            await dispatch_task
        except asyncio.CancelledError:
            pass

        # The waiter should have received the response
        assert future.done()
        result = await future
        assert result.content == "response"


# ---- WeCom Channel ----


class TestWeComChannel:

    @pytest.mark.asyncio
    async def test_webhook_verification(self) -> None:
        config = MagicMock()
        config.allow_from = []
        bus = MessageBus()
        ch = WeComChannel(config, bus)

        result = await ch.handle_webhook(b"", {"echostr": "test_echo"})
        assert result["echostr"] == "test_echo"

    @pytest.mark.asyncio
    async def test_webhook_text_message(self) -> None:
        config = MagicMock()
        config.allow_from = []
        config.token = ""  # no token = skip signature verification
        bus = MessageBus()
        ch = WeComChannel(config, bus)

        xml = (
            b"<xml>"
            b"<MsgType>text</MsgType>"
            b"<Content>hello</Content>"
            b"<FromUserName>user1</FromUserName>"
            b"</xml>"
        )
        result = await ch.handle_webhook(xml, {})
        assert result["errcode"] == 0

        # Check inbound message was published
        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.content == "hello"
        assert msg.sender_id == "user1"

    @pytest.mark.asyncio
    async def test_webhook_event_message(self) -> None:
        config = MagicMock()
        config.allow_from = []
        config.token = ""
        bus = MessageBus()
        ch = WeComChannel(config, bus)

        xml = (
            b"<xml>"
            b"<MsgType>event</MsgType>"
            b"<Event>subscribe</Event>"
            b"<FromUserName>user2</FromUserName>"
            b"</xml>"
        )
        result = await ch.handle_webhook(xml, {})
        assert result["errcode"] == 0

        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.content == "[event:subscribe]"

    @pytest.mark.asyncio
    async def test_webhook_json_fallback(self) -> None:
        config = MagicMock()
        config.allow_from = []
        config.token = ""
        bus = MessageBus()
        ch = WeComChannel(config, bus)

        body = json.dumps(
            {"MsgType": "text", "Content": "json_msg", "FromUserName": "user3"}
        ).encode()
        result = await ch.handle_webhook(body, {})
        assert result["errcode"] == 0

        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.content == "json_msg"

    @pytest.mark.asyncio
    async def test_webhook_invalid_body(self) -> None:
        config = MagicMock()
        config.allow_from = []
        config.token = ""
        bus = MessageBus()
        ch = WeComChannel(config, bus)

        result = await ch.handle_webhook(b"not xml or json!!!", {})
        assert result["errcode"] == 0

    def test_verify_signature_no_token(self) -> None:
        config = MagicMock()
        config.token = ""
        config.allow_from = []
        bus = MessageBus()
        ch = WeComChannel(config, bus)
        # No token = always pass
        assert ch.verify_signature("any", "any", "any") is True

    def test_verify_signature_with_token(self) -> None:
        import hashlib

        config = MagicMock()
        config.token = "mytoken"
        config.allow_from = []
        bus = MessageBus()
        ch = WeComChannel(config, bus)

        timestamp = "1234567890"
        nonce = "abc"
        check_str = "".join(sorted(["mytoken", timestamp, nonce]))
        valid_sig = hashlib.sha1(check_str.encode()).hexdigest()

        assert ch.verify_signature(valid_sig, timestamp, nonce) is True
        assert ch.verify_signature("wrong_sig", timestamp, nonce) is False

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        config = MagicMock()
        config.allow_from = []
        bus = MessageBus()
        ch = WeComChannel(config, bus)
        await ch.start()
        assert ch._running
        await ch.stop()
        assert not ch._running


# ---- API Channel ----


class TestAPIChannel:

    @pytest.mark.asyncio
    async def test_handle_request_unauthorized(self) -> None:
        config = MagicMock()
        config.auth_tokens = ["valid_token"]
        config.allow_from = []
        bus = MessageBus()
        ch = APIChannel(config, bus)

        result = await ch.handle_request(content="hi", auth_token="bad_token")
        assert result["status"] == 401

    @pytest.mark.asyncio
    async def test_handle_request_empty_content(self) -> None:
        config = MagicMock()
        config.auth_tokens = []
        config.allow_from = []
        bus = MessageBus()
        ch = APIChannel(config, bus)

        result = await ch.handle_request(content="")
        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_handle_request_no_auth_required(self) -> None:
        config = MagicMock()
        config.auth_tokens = []
        config.allow_from = []
        bus = MessageBus()
        ch = APIChannel(config, bus)

        # Set up agent response in background
        async def fake_respond() -> None:
            await asyncio.sleep(0.1)
            msg = await bus.consume_inbound()
            response_key = msg.metadata.get("response_key", "")
            bus.resolve_response(
                response_key,
                Envelope(
                    channel="api",
                    chat_id=msg.chat_id,
                    sender_id="bot",
                    content="agent response",
                    metadata={"response_key": response_key},
                ),
            )

        asyncio.create_task(fake_respond())
        result = await ch.handle_request(content="hello", timeout=5.0)
        assert result["status"] == 200
        assert result["content"] == "agent response"

    @pytest.mark.asyncio
    async def test_handle_request_timeout(self) -> None:
        config = MagicMock()
        config.auth_tokens = []
        config.allow_from = []
        bus = MessageBus()
        ch = APIChannel(config, bus)

        result = await ch.handle_request(content="hello", timeout=0.1)
        assert result["status"] == 504

    @pytest.mark.asyncio
    async def test_send_with_response_key(self) -> None:
        config = MagicMock()
        config.allow_from = []
        bus = MessageBus()
        ch = APIChannel(config, bus)

        key, future = bus.create_response_waiter(timeout=5.0)
        envelope = Envelope(
            channel="api",
            chat_id="test",
            sender_id="bot",
            content="reply",
            metadata={"response_key": key},
        )
        await ch.send(envelope)
        assert future.done()
        assert (await future).content == "reply"

    @pytest.mark.asyncio
    async def test_send_without_response_key(self) -> None:
        config = MagicMock()
        config.allow_from = []
        bus = MessageBus()
        ch = APIChannel(config, bus)

        # Should not raise
        envelope = Envelope(
            channel="api",
            chat_id="test",
            sender_id="bot",
            content="fire and forget",
        )
        await ch.send(envelope)

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        config = MagicMock()
        config.allow_from = []
        bus = MessageBus()
        ch = APIChannel(config, bus)
        await ch.start()
        assert ch._running
        await ch.stop()
        assert not ch._running


# ---- Webhook Server ----


class TestWebhookServer:

    def test_server_init(self) -> None:
        from nibot.webhook_server import WebhookServer

        server = WebhookServer(host="0.0.0.0", port=8080)
        assert server._host == "0.0.0.0"
        assert server._port == 8080
        assert server._wecom is None
        assert server._api is None

    def test_server_init_with_channels(self) -> None:
        from nibot.webhook_server import WebhookServer

        wecom = MagicMock()
        api = MagicMock()
        server = WebhookServer(
            host="127.0.0.1", port=9090, wecom_channel=wecom, api_channel=api
        )
        assert server._wecom is wecom
        assert server._api is api

    @pytest.mark.asyncio
    async def test_route_wecom(self) -> None:
        from nibot.webhook_server import WebhookServer

        wecom = MagicMock()
        wecom.handle_webhook = AsyncMock(return_value={"errcode": 0, "errmsg": "ok"})
        server = WebhookServer(wecom_channel=wecom)

        result = await server._route("POST", "/webhook/wecom", b"<xml/>", {}, {})
        assert result["errcode"] == 0
        wecom.handle_webhook.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_api_chat(self) -> None:
        from nibot.webhook_server import WebhookServer

        api = MagicMock()
        api.handle_request = AsyncMock(
            return_value={"content": "hi", "status": 200}
        )
        server = WebhookServer(api_channel=api)

        body = json.dumps({"content": "hello", "sender_id": "user1"}).encode()
        headers = {"authorization": "Bearer tok123"}
        result = await server._route("POST", "/api/chat", body, headers, {})
        assert result["status"] == 200
        api.handle_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_not_found(self) -> None:
        from nibot.webhook_server import WebhookServer

        server = WebhookServer()
        result = await server._route("GET", "/unknown", b"", {}, {})
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_route_api_invalid_json(self) -> None:
        from nibot.webhook_server import WebhookServer

        api = MagicMock()
        server = WebhookServer(api_channel=api)

        result = await server._route(
            "POST", "/api/chat", b"not json", {}, {}
        )
        assert result["status"] == 400
