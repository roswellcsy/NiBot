"""Feishu (Lark) channel -- webhook-based integration."""

from __future__ import annotations

from typing import Any

from nibot.bus import MessageBus
from nibot.channel import BaseChannel
from nibot.config import FeishuChannelConfig
from nibot.log import logger
from nibot.types import Envelope


class FeishuChannel(BaseChannel):
    name = "feishu"

    def __init__(self, config: FeishuChannelConfig, bus: MessageBus) -> None:
        super().__init__(config, bus)
        self._client: Any = None

    async def start(self) -> None:
        try:
            import lark_oapi as lark
        except ImportError:
            logger.error("lark-oapi not installed. pip install 'nibot[feishu]'")
            return
        self._client = lark.Client.builder().app_id(
            self.config.app_id
        ).app_secret(self.config.app_secret).build()
        self._running = True
        logger.info("Feishu channel initialized (webhook mode)")

    async def stop(self) -> None:
        self._running = False

    async def send(self, envelope: Envelope) -> None:
        if not self._client:
            return
        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
            import json

            request = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
                CreateMessageRequestBody.builder()
                .receive_id(envelope.chat_id)
                .msg_type("text")
                .content(json.dumps({"text": envelope.content}))
                .build()
            ).build()
            self._client.im.v1.message.create(request)
        except Exception as e:
            logger.error(f"Feishu send error: {e}")

    async def handle_webhook(self, body: dict[str, Any]) -> dict[str, Any]:
        """Process incoming Feishu webhook event. Returns challenge response if needed."""
        if "challenge" in body:
            return {"challenge": body["challenge"]}
        event = body.get("event", {})
        msg = event.get("message", {})
        content_str = msg.get("content", "{}")
        import json
        try:
            content = json.loads(content_str).get("text", "")
        except (ValueError, AttributeError):
            content = content_str
        sender = event.get("sender", {}).get("sender_id", {}).get("open_id", "unknown")
        chat_id = msg.get("chat_id", "unknown")
        await self._handle_incoming(sender, chat_id, content)
        return {"code": 0}
