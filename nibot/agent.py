"""Agent loop -- LLM + Tool iteration until completion."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from nibot.bus import MessageBus
from nibot.config import NiBotConfig
from nibot.context import ContextBuilder
from nibot.log import logger
from nibot.provider import LLMProvider
from nibot.registry import ToolRegistry
from nibot.session import SessionManager
from nibot.types import Envelope, LLMResponse, ToolContext


def _log_task_exception(task: asyncio.Task[Any]) -> None:
    """Callback for fire-and-forget tasks: log exceptions instead of swallowing them."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"Background task failed: {exc!r}")


class AgentLoop:
    """Consume inbound messages, run LLM+Tool loop, publish outbound responses."""

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        registry: ToolRegistry,
        sessions: SessionManager,
        context_builder: ContextBuilder,
        config: NiBotConfig,
        evo_trigger: Any | None = None,
        rate_limiter: Any | None = None,
        provider_pool: Any | None = None,
        event_log: Any | None = None,
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.registry = registry
        self.sessions = sessions
        self.context_builder = context_builder
        self.max_iterations = config.agent.max_iterations
        self._gateway_tools: list[str] = config.agent.gateway_tools
        self._streaming = config.agent.streaming
        self._fallback_chain: list[str] = config.agent.provider_fallback_chain
        self._provider_pool = provider_pool
        self._evo_trigger = evo_trigger
        self._rate_limiter = rate_limiter
        self._event_log = event_log
        self._running = False
        self._semaphore = asyncio.Semaphore(10)
        self._tasks: set[asyncio.Task[None]] = set()
        self._bg_tasks: set[asyncio.Task[Any]] = set()  # fire-and-forget tasks (evo checks)

    async def run(self) -> None:
        self._running = True
        while self._running:
            envelope = await self.bus.consume_inbound()
            await self._semaphore.acquire()
            task = asyncio.create_task(self._guarded_handle(envelope))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _guarded_handle(self, envelope: Envelope) -> None:
        try:
            await self._handle(envelope)
        finally:
            self._semaphore.release()

    async def _handle(self, envelope: Envelope) -> None:
        try:
            response = await self._process(envelope)
            # Skip publishing if streaming already delivered the content,
            # BUT always publish if there's a response_key (API channel waiter).
            meta = response.metadata or {}
            if not meta.get("streamed") or meta.get("response_key"):
                await self.bus.publish_outbound(response)
        except Exception as e:
            logger.error(f"Agent error [{envelope.channel}:{envelope.chat_id}]: {e}")
            await self.bus.publish_outbound(Envelope(
                channel=envelope.channel,
                chat_id=envelope.chat_id,
                sender_id="assistant",
                content="Sorry, an internal error occurred. Please try again.",
                metadata=envelope.metadata,
            ))

    async def _process(self, envelope: Envelope) -> Envelope:
        # Rate limit check (before any expensive work)
        if self._rate_limiter and self._rate_limiter.enabled:
            allowed, reason = self._rate_limiter.check(
                user_key=envelope.sender_id,
                channel_key=envelope.channel,
            )
            if not allowed:
                return Envelope(
                    channel=envelope.channel,
                    chat_id=envelope.chat_id,
                    sender_id="assistant",
                    content=f"Rate limit exceeded. Please wait a moment. ({reason})",
                    metadata=envelope.metadata,
                )

        t0 = time.monotonic()
        session_key = f"{envelope.channel}:{envelope.chat_id}"
        tool_ctx = ToolContext(
            channel=envelope.channel,
            chat_id=envelope.chat_id,
            session_key=session_key,
            sender_id=envelope.sender_id,
        )

        async with self.sessions.lock_for(session_key):
            session = self.sessions.get_or_create(session_key)
            messages = self.context_builder.build(session=session, current=envelope)
            pre_loop_len = len(messages)  # snapshot before LLM loop adds messages
            tool_defs = (
                self.registry.get_definitions(allow=self._gateway_tools)
                if self._gateway_tools
                else self.registry.get_definitions()
            )
            final_content = ""
            stream_seq = 0
            tool_count = 0
            total_tokens = 0

            # Only stream if provider actually overrides chat_stream (not base fallback)
            can_stream = (
                self._streaming
                and hasattr(self.provider, "chat_stream")
                and type(self.provider).chat_stream is not LLMProvider.chat_stream
            )

            for _iteration in range(self.max_iterations):
                # Progress: thinking event
                _sid = (envelope.metadata or {}).get("stream_id")
                if _sid:
                    _pmeta = {
                        k: v for k, v in (envelope.metadata or {}).items()
                        if k != "response_key"
                    }
                    await self.bus.publish_outbound(Envelope(
                        channel=envelope.channel, chat_id=envelope.chat_id,
                        sender_id="assistant", content="",
                        metadata={**_pmeta, "progress": "thinking",
                                  "iteration": _iteration + 1,
                                  "max_iterations": self.max_iterations},
                    ))

                if can_stream:
                    response = None
                    full_text = ""
                    acc = ""
                    # Strip response_key to prevent API waiter from resolving on chunks
                    stream_meta = {
                        k: v for k, v in (envelope.metadata or {}).items()
                        if k != "response_key"
                    }
                    async for item in self.provider.chat_stream(
                        messages=messages, tools=tool_defs or None
                    ):
                        if isinstance(item, LLMResponse):
                            response = item
                        elif isinstance(item, str):
                            full_text += item
                            acc += item
                            if len(acc) >= 30:
                                await self.bus.publish_outbound(Envelope(
                                    channel=envelope.channel,
                                    chat_id=envelope.chat_id,
                                    sender_id="assistant",
                                    content=full_text,
                                    metadata={
                                        **stream_meta,
                                        "streaming": True,
                                        "stream_seq": stream_seq,
                                    },
                                ))
                                acc = ""
                                stream_seq += 1
                    # Always send stream_done when streaming happened
                    if stream_seq > 0:
                        await self.bus.publish_outbound(Envelope(
                            channel=envelope.channel,
                            chat_id=envelope.chat_id,
                            sender_id="assistant",
                            content=full_text,
                            metadata={
                                **stream_meta,
                                "streaming": True,
                                "stream_seq": stream_seq,
                                "stream_done": True,
                                "has_tool_calls": bool(
                                    response and response.has_tool_calls
                                ),
                            },
                        ))
                        stream_seq += 1
                    if response is None:
                        response = LLMResponse(content=full_text)
                    if response.usage:
                        total_tokens += response.usage.get("total_tokens", 0)
                    if not response.has_tool_calls:
                        final_content = response.content or full_text or ""
                        break
                else:
                    if self._fallback_chain and self._provider_pool:
                        response = await self._provider_pool.chat_with_fallback(
                            messages=messages, tools=tool_defs or None,
                            chain=self._fallback_chain,
                        )
                    else:
                        response = await self.provider.chat(
                            messages=messages, tools=tool_defs or None
                        )
                    # Track tokens from every LLM response
                    if response.usage:
                        total_tokens += response.usage.get("total_tokens", 0)
                    if not response.has_tool_calls:
                        final_content = response.content or ""
                        break

                # Append assistant message with tool_calls
                tc_dicts: list[dict[str, Any]] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": tc_dicts,
                })

                # Execute each tool, append results
                for tc in response.tool_calls:
                    # Progress: tool_start
                    if _sid:
                        await self.bus.publish_outbound(Envelope(
                            channel=envelope.channel, chat_id=envelope.chat_id,
                            sender_id="assistant", content="",
                            metadata={**_pmeta, "progress": "tool_start",
                                      "tool_name": tc.name},
                        ))
                    _t0_tool = time.monotonic()
                    result = await self.registry.execute(tc.name, tc.arguments, call_id=tc.id, ctx=tool_ctx)
                    tool_count += 1
                    # Progress: tool_done
                    if _sid:
                        await self.bus.publish_outbound(Envelope(
                            channel=envelope.channel, chat_id=envelope.chat_id,
                            sender_id="assistant", content="",
                            metadata={**_pmeta, "progress": "tool_done",
                                      "tool_name": tc.name,
                                      "elapsed": round(time.monotonic() - _t0_tool, 1)},
                        ))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "name": tc.name,
                        "content": result.content,
                    })

            if not final_content:
                final_content = "I was unable to complete the task within the allowed steps."

            # Persist: user message + all messages added during LLM loop
            session.add_message("user", envelope.content)
            for msg in messages[pre_loop_len:]:
                extras = {k: v for k, v in msg.items() if k not in ("role", "content")}
                session.add_message(msg["role"], msg.get("content") or "", **extras)
            if final_content:
                session.add_message("assistant", final_content)
            self.sessions.save(session)

            # Log request-level event
            if self._event_log:
                latency_ms = (time.monotonic() - t0) * 1000
                self._event_log.log_request(
                    channel=envelope.channel,
                    session_key=session_key,
                    latency_ms=latency_ms,
                    tool_count=tool_count,
                    total_tokens=total_tokens,
                    provider="fallback" if (self._fallback_chain and self._provider_pool) else "default",
                )

            # Fire-and-forget evolution check after session save
            if self._evo_trigger:
                t = asyncio.create_task(self._evo_trigger.check())
                self._bg_tasks.add(t)
                t.add_done_callback(lambda done: (self._bg_tasks.discard(done), _log_task_exception(done)))

        out_meta = dict(envelope.metadata or {})
        if stream_seq > 0:
            out_meta["streamed"] = True
        return Envelope(
            channel=envelope.channel,
            chat_id=envelope.chat_id,
            sender_id="assistant",
            content=final_content,
            metadata=out_meta,
        )

    def stop(self) -> None:
        self._running = False
