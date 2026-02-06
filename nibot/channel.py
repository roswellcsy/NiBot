"""Base channel -- abstract interface for messaging platforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nibot.bus import MessageBus
from nibot.types import Envelope


class BaseChannel(ABC):
    """Implement this to connect a messaging platform."""

    name: str = "base"

    def __init__(self, config: Any, bus: MessageBus) -> None:
        self.config = config
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send(self, envelope: Envelope) -> None: ...

    def is_allowed(self, sender_id: str) -> bool:
        allow_list = getattr(self.config, "allow_from", []) or []
        if not allow_list:
            return True
        sid = str(sender_id)
        if sid in allow_list:
            return True
        for part in sid.split("|"):
            if part and part in allow_list:
                return True
        return False

    async def _handle_incoming(
        self, sender_id: str, chat_id: str, content: str, **kwargs: Any
    ) -> None:
        if not self.is_allowed(sender_id):
            return
        await self.bus.publish_inbound(
            Envelope(
                channel=self.name,
                chat_id=str(chat_id),
                sender_id=str(sender_id),
                content=content,
                **kwargs,
            )
        )
