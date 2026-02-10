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
from nibot.types import Envelope, LLMResponse, ToolCallDelta, ToolContext


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
        self._stream_chunk_size = config.agent.streaming_chunk_size
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
            pre_loop_len = len(messages)
            tool_defs = (
                self.registry.get_definitions(allow=self._gateway_tools)
                if self._gateway_tools
                else self.registry.get_definitions()
            )

            final_content, stream_seq, tool_count, total_tokens = await self._llm_loop(
                messages, tool_defs, envelope, tool_ctx,
            )

            if not final_content:
                final_content = "I was unable to complete the task within the allowed steps."

            self._persist(session, envelope, messages, final_content, pre_loop_len)
            self._log_event(envelope, session_key, t0, tool_count, total_tokens)
            self._maybe_trigger_evolution()

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

    # -- Sub-methods split from _process() --

    async def _llm_loop(
        self,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        envelope: Envelope,
        tool_ctx: ToolContext,
    ) -> tuple[str, int, int, int]:
        """Unified LLM iteration loop. Returns (final_content, stream_seq, tool_count, total_tokens)."""
        can_stream = (
            self._streaming
            and hasattr(self.provider, "chat_stream")
            and type(self.provider).chat_stream is not LLMProvider.chat_stream
        )
        stream_seq = 0
        tool_count = 0
        total_tokens = 0

        _sid = (envelope.metadata or {}).get("stream_id")
        _pmeta = (
            {k: v for k, v in (envelope.metadata or {}).items() if k != "response_key"}
            if _sid else {}
        )

        for _iteration in range(self.max_iterations):
            if _sid:
                await self.bus.publish_outbound(Envelope(
                    channel=envelope.channel, chat_id=envelope.chat_id,
                    sender_id="assistant", content="",
                    metadata={**_pmeta, "progress": "thinking",
                              "iteration": _iteration + 1,
                              "max_iterations": self.max_iterations},
                ))

            response, text, stream_seq = await self._llm_call(
                messages, tool_defs, envelope, can_stream, stream_seq,
            )
            if response.usage:
                total_tokens += response.usage.get("total_tokens", 0)
            if not response.has_tool_calls:
                return (response.content or text or "", stream_seq, tool_count, total_tokens)

            tool_count += await self._execute_tools(
                response, messages, envelope, tool_ctx, _sid, _pmeta,
            )

        return ("", stream_seq, tool_count, total_tokens)

    async def _llm_call(
        self,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        envelope: Envelope,
        can_stream: bool,
        stream_seq: int,
    ) -> tuple[LLMResponse, str, int]:
        """Single LLM call (streaming or not). Returns (response, full_text, stream_seq)."""
        if can_stream:
            return await self._llm_call_stream(messages, tool_defs, envelope, stream_seq)

        if self._fallback_chain and self._provider_pool:
            response = await self._provider_pool.chat_with_fallback(
                messages=messages, tools=tool_defs or None,
                chain=self._fallback_chain,
            )
        else:
            response = await self.provider.chat(
                messages=messages, tools=tool_defs or None,
            )
        return response, response.content or "", stream_seq

    async def _llm_call_stream(
        self,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        envelope: Envelope,
        stream_seq: int,
    ) -> tuple[LLMResponse, str, int]:
        """Streaming LLM call with chunk publishing. Returns (response, full_text, stream_seq)."""
        response = None
        full_text = ""
        acc = ""
        stream_meta = {
            k: v for k, v in (envelope.metadata or {}).items()
            if k != "response_key"
        }
        async for item in self.provider.chat_stream(
            messages=messages, tools=tool_defs or None
        ):
            if isinstance(item, LLMResponse):
                response = item
            elif isinstance(item, ToolCallDelta):
                # Progressive tool-call args display for web panel
                _sid = (envelope.metadata or {}).get("stream_id")
                if _sid:
                    await self.bus.publish_outbound(Envelope(
                        channel=envelope.channel,
                        chat_id=envelope.chat_id,
                        sender_id="assistant",
                        content="",
                        metadata={
                            **stream_meta,
                            "progress": "tool_args_delta",
                            "tool_name": item.name,
                            "partial_args": item.partial_args[:200],
                        },
                    ))
            elif isinstance(item, str):
                full_text += item
                acc += item
                if len(acc) >= self._stream_chunk_size:
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
                    "has_tool_calls": bool(response and response.has_tool_calls),
                },
            ))
            stream_seq += 1
        if response is None:
            response = LLMResponse(content=full_text)
        return response, full_text, stream_seq

    async def _execute_tools(
        self,
        response: LLMResponse,
        messages: list[dict[str, Any]],
        envelope: Envelope,
        tool_ctx: ToolContext,
        stream_id: str | None,
        progress_meta: dict[str, Any],
    ) -> int:
        """Execute tool calls from response, append results to messages. Returns tool count."""
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

        count = 0
        for tc in response.tool_calls:
            if stream_id:
                await self.bus.publish_outbound(Envelope(
                    channel=envelope.channel, chat_id=envelope.chat_id,
                    sender_id="assistant", content="",
                    metadata={**progress_meta, "progress": "tool_start",
                              "tool_name": tc.name},
                ))
            t0 = time.monotonic()
            result = await self.registry.execute(tc.name, tc.arguments, call_id=tc.id, ctx=tool_ctx)
            count += 1
            if stream_id:
                await self.bus.publish_outbound(Envelope(
                    channel=envelope.channel, chat_id=envelope.chat_id,
                    sender_id="assistant", content="",
                    metadata={**progress_meta, "progress": "tool_done",
                              "tool_name": tc.name,
                              "elapsed": round(time.monotonic() - t0, 1)},
                ))
            messages.append({
                "role": "tool",
                "tool_call_id": result.call_id,
                "name": tc.name,
                "content": result.content,
            })
        return count

    def _persist(
        self,
        session: Any,
        envelope: Envelope,
        messages: list[dict[str, Any]],
        final_content: str,
        pre_loop_len: int,
    ) -> None:
        """Persist user message + LLM loop messages + final response to session."""
        session.add_message("user", envelope.content)
        for msg in messages[pre_loop_len:]:
            extras = {k: v for k, v in msg.items() if k not in ("role", "content")}
            session.add_message(msg["role"], msg.get("content") or "", **extras)
        if final_content:
            session.add_message("assistant", final_content)
        self.sessions.save(session)

    def _log_event(
        self,
        envelope: Envelope,
        session_key: str,
        t0: float,
        tool_count: int,
        total_tokens: int,
    ) -> None:
        """Log request-level event if event_log is configured."""
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

    def _maybe_trigger_evolution(self) -> None:
        """Fire-and-forget evolution check."""
        if self._evo_trigger:
            t = asyncio.create_task(self._evo_trigger.check())
            self._bg_tasks.add(t)
            t.add_done_callback(lambda done: (self._bg_tasks.discard(done), _log_task_exception(done)))

    def stop(self) -> None:
        self._running = False
