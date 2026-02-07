"""Feishu (Lark) channel -- webhook-based integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nibot.bus import MessageBus
from nibot.channel import BaseChannel
from nibot.config import FeishuChannelConfig
from nibot.log import logger
from nibot.types import Envelope

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


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
            import json

            # Send media first
            for media_path in (envelope.media or []):
                path = Path(media_path)
                if path.exists() and path.suffix.lower() in _IMAGE_EXTENSIONS:
                    if not await self._send_image(envelope.chat_id, path):
                        logger.warning(f"Feishu image send failed, skipping: {path.name}")

            # Then send text
            if envelope.content:
                self._send_text(envelope.chat_id, envelope.content)
        except Exception as e:
            logger.error(f"Feishu send error: {e}")

    def _send_text(self, chat_id: str, text: str) -> None:
        import json
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        request = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("text")
            .content(json.dumps({"text": text}))
            .build()
        ).build()
        self._client.im.v1.message.create(request)

    async def _send_image(self, chat_id: str, path: Path) -> bool:
        """Upload image to Feishu and send as image message. Returns False on failure."""
        try:
            from lark_oapi.api.im.v1 import (
                CreateImageRequest, CreateImageRequestBody,
                CreateMessageRequest, CreateMessageRequestBody,
            )
            import json

            # Step 1: Upload image
            with open(path, "rb") as f:
                upload_req = CreateImageRequest.builder().request_body(
                    CreateImageRequestBody.builder()
                    .image_type("message")
                    .image(f)
                    .build()
                ).build()
                upload_resp = self._client.im.v1.image.create(upload_req)
            if not upload_resp or not upload_resp.success():
                return False
            image_key = upload_resp.data.image_key

            # Step 2: Send image message
            msg_req = CreateMessageRequest.builder().receive_id_type("chat_id").request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("image")
                .content(json.dumps({"image_key": image_key}))
                .build()
            ).build()
            self._client.im.v1.message.create(msg_req)
            return True
        except Exception as e:
            logger.error(f"Feishu image upload/send error: {e}")
            return False

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
