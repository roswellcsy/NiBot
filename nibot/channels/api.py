"""API channel -- synchronous request/response via HTTP."""
from __future__ import annotations

import asyncio
from typing import Any

from nibot.channel import BaseChannel
from nibot.log import logger
from nibot.types import Envelope


class APIChannel(BaseChannel):
    """HTTP API channel for synchronous request/response.

    Unlike Telegram/WeCom (push-based), the API channel waits for the
    agent's response before returning the HTTP response.
    """

    name = "api"

    async def start(self) -> None:
        self._running = True
        logger.info("API channel initialized")

    async def stop(self) -> None:
        self._running = False

    async def send(self, envelope: Envelope) -> None:
        """Outbound messages from API channel are handled via response waiters."""
        # Skip streaming chunks -- API is synchronous, waits for final response
        meta = envelope.metadata or {}
        if meta.get("streaming"):
            return
        # Check if this message has a response_key (sync request waiting)
        response_key = meta.get("response_key", "")
        if response_key:
            self.bus.resolve_response(response_key, envelope)
        # Otherwise, it's a fire-and-forget outbound -- log and drop
        else:
            logger.debug(f"API outbound without waiter: {envelope.content[:100]}")

    async def handle_request(
        self,
        content: str,
        sender_id: str = "api",
        chat_id: str = "",
        auth_token: str = "",
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Handle a synchronous API request.

        Publishes to inbound, waits for outbound response, returns it.
        """
        # Auth check
        allowed_tokens = getattr(self.config, "auth_tokens", []) or []
        if allowed_tokens and auth_token not in allowed_tokens:
            return {"error": "unauthorized", "status": 401}

        if not content:
            return {"error": "empty content", "status": 400}

        # Create response waiter
        waiter_key, future = self.bus.create_response_waiter(timeout=timeout)

        # Publish inbound with response_key in metadata
        chat_id = chat_id or f"api_{sender_id}"
        await self.bus.publish_inbound(
            Envelope(
                channel=self.name,
                chat_id=chat_id,
                sender_id=sender_id,
                content=content,
                metadata={"response_key": waiter_key},
            )
        )

        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return {
                "content": response.content,
                "channel": response.channel,
                "status": 200,
            }
        except asyncio.TimeoutError:
            return {"error": "response timeout", "status": 504}
        except Exception as e:
            return {"error": str(e), "status": 500}
