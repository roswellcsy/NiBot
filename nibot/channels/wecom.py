"""WeCom (Enterprise WeChat) channel -- webhook-based messaging."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from nibot.channel import BaseChannel
from nibot.log import logger
from nibot.types import Envelope


class WeComChannel(BaseChannel):
    """WeCom channel using callback webhook for receiving messages.

    Requires: corp_id, secret, token, encoding_aes_key in config.
    Message flow:
      Inbound: WeCom webhook POST -> handle_webhook() -> bus.publish_inbound()
      Outbound: bus subscriber -> send() -> WeCom API POST
    """

    name = "wecom"

    async def start(self) -> None:
        """Initialize WeCom client. Actual webhook server is external (webhook_server.py)."""
        self._running = True
        logger.info("WeCom channel initialized")

    async def stop(self) -> None:
        self._running = False

    async def send(self, envelope: Envelope) -> None:
        """Send message via WeCom API."""
        import httpx

        corp_id = getattr(self.config, "corp_id", "")
        secret = getattr(self.config, "secret", "")
        agent_id = getattr(self.config, "agent_id", "")

        if not corp_id or not secret:
            logger.warning("WeCom send skipped: missing corp_id or secret")
            return

        try:
            token = await self._get_access_token(corp_id, secret)
            url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"

            payload = {
                "touser": envelope.chat_id,
                "msgtype": "text",
                "agentid": int(agent_id) if agent_id else 0,
                "text": {"content": envelope.content},
            }

            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=10.0)
                data = resp.json()
                if data.get("errcode", 0) != 0:
                    logger.error(f"WeCom send error: {data}")
        except Exception as e:
            logger.error(f"WeCom send failed: {e}")

    async def handle_webhook(
        self, body: bytes, query_params: dict[str, str]
    ) -> dict[str, Any]:
        """Handle incoming WeCom webhook callback.

        Returns response dict. Called by webhook_server.
        """
        # Verification request (GET with echostr)
        echostr = query_params.get("echostr", "")
        if echostr:
            return {"echostr": echostr}

        # Message callback (POST) -- verify signature first
        msg_signature = query_params.get("msg_signature", query_params.get("signature", ""))
        timestamp = query_params.get("timestamp", "")
        nonce = query_params.get("nonce", "")
        if not self.verify_signature(msg_signature, timestamp, nonce):
            logger.warning("WeCom webhook: signature verification failed")
            return {"errcode": 403, "errmsg": "signature verification failed"}

        try:
            data = self._parse_message(body, query_params)
            if not data:
                return {"errcode": 0, "errmsg": "ok"}

            msg_type = data.get("MsgType", "")
            content = ""
            if msg_type == "text":
                content = data.get("Content", "")
            elif msg_type == "event":
                content = f"[event:{data.get('Event', '')}]"
            else:
                content = f"[{msg_type}]"

            if content:
                sender_id = data.get("FromUserName", "")
                await self._handle_incoming(
                    sender_id=sender_id,
                    chat_id=sender_id,  # WeCom: user ID is chat ID
                    content=content,
                )
        except Exception as e:
            logger.error(f"WeCom webhook error: {e}")

        return {"errcode": 0, "errmsg": "ok"}

    def _parse_message(
        self, body: bytes, query_params: dict[str, str]
    ) -> dict[str, Any] | None:
        """Parse WeCom callback message body.

        For simplicity, handles plain XML. Full implementation would decrypt
        using encoding_aes_key, but that requires wechatpy or manual AES.
        """
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(body)
            return {child.tag: child.text or "" for child in root}
        except Exception as e:
            logger.warning(f"WeCom message parse failed: {e}")
            # Try JSON fallback
            try:
                return json.loads(body)
            except (json.JSONDecodeError, ValueError):
                return None

    def verify_signature(
        self, signature: str, timestamp: str, nonce: str
    ) -> bool:
        """Verify WeCom callback signature."""
        token = getattr(self.config, "token", "")
        if not token:
            return True  # No token configured, skip verification
        check_str = "".join(sorted([token, timestamp, nonce]))
        calculated = hashlib.sha1(check_str.encode()).hexdigest()
        return calculated == signature

    async def _get_access_token(self, corp_id: str, secret: str) -> str:
        """Get WeCom API access token (no caching for simplicity)."""
        import httpx

        url = (
            f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
            f"?corpid={corp_id}&corpsecret={secret}"
        )
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10.0)
            data = resp.json()
            return data.get("access_token", "")
