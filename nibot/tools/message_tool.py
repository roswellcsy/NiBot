"""Cross-channel messaging tool."""

from __future__ import annotations

from typing import Any

from nibot.bus import MessageBus
from nibot.registry import Tool
from nibot.types import Envelope


class MessageTool(Tool):
    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to a specific channel and chat."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Target channel name"},
                "chat_id": {"type": "string", "description": "Target chat ID"},
                "content": {"type": "string", "description": "Message content"},
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths to send as media attachments",
                },
            },
            "required": ["channel", "chat_id", "content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        kwargs.pop("_tool_ctx", None)
        media = kwargs.get("media") or []
        await self._bus.publish_outbound(
            Envelope(
                channel=kwargs["channel"],
                chat_id=kwargs["chat_id"],
                sender_id="assistant",
                content=kwargs["content"],
                media=media,
            )
        )
        return f"Sent to {kwargs['channel']}:{kwargs['chat_id']}"
