"""Message Bus -- decouples Channels from Agent."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

from nibot.log import logger
from nibot.types import Envelope


class MessageBus:
    """Async message bus with inbound/outbound queues and subscriber dispatch."""

    def __init__(self, maxsize: int = 0) -> None:
        self._inbound: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=maxsize)
        self._outbound: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=maxsize)
        self._subscribers: dict[str, list[Callable[[Envelope], Awaitable[None]]]] = {}
        self._response_waiters: dict[str, asyncio.Future[Envelope]] = {}
        self._running = False

    async def publish_inbound(self, envelope: Envelope) -> None:
        await self._inbound.put(envelope)

    async def consume_inbound(self) -> Envelope:
        return await self._inbound.get()

    async def publish_outbound(self, envelope: Envelope) -> None:
        await self._outbound.put(envelope)

    def subscribe_outbound(self, channel: str, callback: Callable[[Envelope], Awaitable[None]]) -> None:
        self._subscribers.setdefault(channel, []).append(callback)

    def create_response_waiter(
        self, timeout: float = 30.0
    ) -> tuple[str, "asyncio.Future[Envelope]"]:
        """Create a waiter that will receive the response for a specific request.

        Returns (waiter_key, future). Set waiter_key in envelope.metadata["response_key"].
        """
        key = f"_response_{uuid.uuid4().hex[:8]}"
        future: asyncio.Future[Envelope] = asyncio.get_event_loop().create_future()
        self._response_waiters[key] = future

        # Auto-cleanup on timeout
        async def _cleanup() -> None:
            await asyncio.sleep(timeout)
            if key in self._response_waiters:
                f = self._response_waiters.pop(key)
                if not f.done():
                    f.set_exception(
                        asyncio.TimeoutError(f"Response waiter {key} timed out")
                    )

        asyncio.create_task(_cleanup())
        return key, future

    def resolve_response(self, key: str, envelope: Envelope) -> bool:
        """Resolve a response waiter. Returns True if waiter existed."""
        future = self._response_waiters.pop(key, None)
        if future and not future.done():
            future.set_result(envelope)
            return True
        return False

    async def dispatch_outbound(self) -> None:
        """Background loop: dequeue outbound messages and dispatch to subscribers."""
        self._running = True
        while self._running:
            try:
                msg = await asyncio.wait_for(self._outbound.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Check for response waiter -- deliver directly, skip normal dispatch
            response_key = (msg.metadata or {}).get("response_key", "")
            if response_key and self.resolve_response(response_key, msg):
                continue

            handlers = self._subscribers.get(msg.channel, [])
            if not handlers:
                logger.warning(f"No subscriber for channel '{msg.channel}', message dropped")
            for cb in handlers:
                try:
                    await cb(msg)
                except Exception as e:
                    logger.error(f"Dispatch error to {msg.channel}: {e}")

    def stop(self) -> None:
        self._running = False
