"""Message Bus -- decouples Channels from Agent."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from nibot.log import logger
from nibot.types import Envelope


class MessageBus:
    """Async message bus with inbound/outbound queues and subscriber dispatch."""

    def __init__(self) -> None:
        self._inbound: asyncio.Queue[Envelope] = asyncio.Queue()
        self._outbound: asyncio.Queue[Envelope] = asyncio.Queue()
        self._subscribers: dict[str, list[Callable[[Envelope], Awaitable[None]]]] = {}
        self._running = False

    async def publish_inbound(self, envelope: Envelope) -> None:
        await self._inbound.put(envelope)

    async def consume_inbound(self) -> Envelope:
        return await self._inbound.get()

    async def publish_outbound(self, envelope: Envelope) -> None:
        await self._outbound.put(envelope)

    def subscribe_outbound(self, channel: str, callback: Callable[[Envelope], Awaitable[None]]) -> None:
        self._subscribers.setdefault(channel, []).append(callback)

    async def dispatch_outbound(self) -> None:
        """Background loop: dequeue outbound messages and dispatch to subscribers."""
        self._running = True
        while self._running:
            try:
                msg = await asyncio.wait_for(self._outbound.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            for cb in self._subscribers.get(msg.channel, []):
                try:
                    await cb(msg)
                except Exception as e:
                    logger.error(f"Dispatch error to {msg.channel}: {e}")

    def stop(self) -> None:
        self._running = False
