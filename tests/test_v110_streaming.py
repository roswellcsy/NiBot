"""v1.1 Streaming responses tests."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nibot.agent import AgentLoop
from nibot.bus import MessageBus
from nibot.config import NiBotConfig
from nibot.provider import LLMProvider
from nibot.registry import ToolRegistry
from nibot.session import SessionManager
from nibot.types import Envelope, LLMResponse, ToolCall


class StreamingFakeProvider(LLMProvider):
    """Fake provider that inherits LLMProvider (gets chat_stream from base class)."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses: list[LLMResponse] = responses or []
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, messages=None, tools=None, model="", max_tokens=4096, temperature=0.7) -> LLMResponse:
        self.calls.append([dict(m) for m in (messages or [])])
        if not self.responses:
            return LLMResponse(content="(no more)")
        return self.responses.pop(0)


class MultiChunkProvider(LLMProvider):
    """Provider that yields text in small chunks, simulating real streaming."""

    def __init__(self, responses: list[LLMResponse] | None = None, chunk_size: int = 10) -> None:
        self.responses: list[LLMResponse] = responses or []
        self.calls: list[list[dict[str, Any]]] = []
        self.chunk_size = chunk_size

    async def chat(self, messages=None, tools=None, model="", max_tokens=4096, temperature=0.7) -> LLMResponse:
        self.calls.append([dict(m) for m in (messages or [])])
        if not self.responses:
            return LLMResponse(content="(no more)")
        return self.responses.pop(0)

    async def chat_stream(self, messages=None, tools=None, model="", max_tokens=4096, temperature=0.7):
        resp = await self.chat(messages, tools, model, max_tokens, temperature)
        if resp.has_tool_calls:
            yield resp
            return
        text = resp.content or ""
        for i in range(0, len(text), self.chunk_size):
            yield text[i:i + self.chunk_size]
        yield resp


class FakeContextBuilder:
    def build(self, session, current):
        return [{"role": "system", "content": "test"}, {"role": "user", "content": current.content}]


def _make_agent(bus, provider, registry, sessions, streaming=True):
    config = NiBotConfig()
    config.agent.streaming = streaming
    return AgentLoop(
        bus=bus, provider=provider, registry=registry,
        sessions=sessions, context_builder=FakeContextBuilder(), config=config,
    )


class TestProviderChatStreamBase:
    """Base LLMProvider.chat_stream behavior."""

    @pytest.mark.asyncio
    async def test_text_response_yields_str(self) -> None:
        provider = StreamingFakeProvider([LLMResponse(content="hello world")])
        items = []
        async for item in provider.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            items.append(item)
        assert items == ["hello world"]
        # Only str, no LLMResponse (base class behavior for non-tool text)

    @pytest.mark.asyncio
    async def test_tool_calls_yield_llm_response(self) -> None:
        resp = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="foo", arguments={"x": "1"})],
        )
        provider = StreamingFakeProvider([resp])
        items = []
        async for item in provider.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "foo"}}],
        ):
            items.append(item)
        # Should yield LLMResponse with tool_calls
        assert len(items) == 1
        assert isinstance(items[0], LLMResponse)
        assert items[0].has_tool_calls

    @pytest.mark.asyncio
    async def test_no_content_yields_nothing(self) -> None:
        provider = StreamingFakeProvider([LLMResponse(content="")])
        items = []
        async for item in provider.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            items.append(item)
        assert items == []


class TestStreamingAgentChunks:
    """Agent publishes streaming chunks to bus."""

    @pytest.mark.asyncio
    async def test_long_response_produces_streaming_envelopes(self, tmp_path) -> None:
        # 65 chars with 10-char chunks: 7 chunks yielded by provider.
        # Agent accumulates 30 chars (3 chunks) -> publish seq=0, then 30 more -> publish seq=1,
        # remaining 5 chars flushed as stream_done.
        long_text = "A" * 65
        bus = MessageBus()
        provider = MultiChunkProvider([LLMResponse(content=long_text)], chunk_size=10)
        registry = ToolRegistry()
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions, streaming=True)

        captured: list[Envelope] = []

        async def capture(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", capture)

        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="go")
        )

        agent_task = asyncio.create_task(agent.run())
        dispatch_task = asyncio.create_task(bus.dispatch_outbound())
        await asyncio.sleep(0.5)
        agent.stop()
        bus.stop()
        agent_task.cancel()
        dispatch_task.cancel()
        try:
            await asyncio.gather(agent_task, dispatch_task)
        except asyncio.CancelledError:
            pass

        # Should have streaming envelopes: 2 intermediate + 1 stream_done
        streaming = [e for e in captured if (e.metadata or {}).get("streaming")]
        assert len(streaming) >= 3
        # First chunk has seq=0, cumulative content = first 30 chars
        assert streaming[0].metadata["stream_seq"] == 0
        assert streaming[0].content == "A" * 30
        # Second chunk has seq=1, cumulative content = first 60 chars
        assert streaming[1].metadata["stream_seq"] == 1
        assert streaming[1].content == "A" * 60
        # stream_done chunk has full content
        last = streaming[-1]
        assert last.metadata.get("stream_done") is True
        assert last.content == long_text
        # No response_key leaked into streaming metadata
        for s in streaming:
            assert "response_key" not in s.metadata

    @pytest.mark.asyncio
    async def test_short_response_no_streaming_envelopes(self, tmp_path) -> None:
        # < 30 char response doesn't trigger streaming chunk publishing
        bus = MessageBus()
        provider = StreamingFakeProvider([LLMResponse(content="short")])
        registry = ToolRegistry()
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions, streaming=True)

        captured: list[Envelope] = []

        async def capture(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", capture)

        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="go")
        )

        agent_task = asyncio.create_task(agent.run())
        dispatch_task = asyncio.create_task(bus.dispatch_outbound())
        await asyncio.sleep(0.5)
        agent.stop()
        bus.stop()
        agent_task.cancel()
        dispatch_task.cancel()
        try:
            await asyncio.gather(agent_task, dispatch_task)
        except asyncio.CancelledError:
            pass

        # No streaming chunks (content < 30 chars), just the final envelope
        streaming = [e for e in captured if (e.metadata or {}).get("streaming")]
        assert len(streaming) == 0
        # Final envelope delivered normally (not streamed)
        non_streaming = [e for e in captured if not (e.metadata or {}).get("streaming")]
        assert len(non_streaming) == 1
        assert non_streaming[0].content == "short"


class TestStreamingWithTools:
    """Tool calls use non-streaming fallback."""

    @pytest.mark.asyncio
    async def test_tool_call_then_text_no_streaming_chunks(self, tmp_path) -> None:
        bus = MessageBus()
        provider = StreamingFakeProvider([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="dummy", arguments={"x": "1"})],
            ),
            LLMResponse(content="done"),
        ])
        registry = ToolRegistry()

        from nibot.registry import Tool

        class DummyTool(Tool):
            @property
            def name(self): return "dummy"
            @property
            def description(self): return "d"
            @property
            def parameters(self): return {"type": "object", "properties": {"x": {"type": "string"}}}
            async def execute(self, **kw): return "ok"

        registry.register(DummyTool())
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions, streaming=True)

        captured: list[Envelope] = []

        async def capture(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", capture)

        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="go")
        )

        agent_task = asyncio.create_task(agent.run())
        dispatch_task = asyncio.create_task(bus.dispatch_outbound())
        await asyncio.sleep(0.5)
        agent.stop()
        bus.stop()
        agent_task.cancel()
        dispatch_task.cancel()
        try:
            await asyncio.gather(agent_task, dispatch_task)
        except asyncio.CancelledError:
            pass

        # "done" is only 4 chars -- no streaming chunks
        streaming = [e for e in captured if (e.metadata or {}).get("streaming")]
        assert len(streaming) == 0
        # Final envelope has the content
        assert any(e.content == "done" for e in captured)


class TestStreamingDisabled:
    """streaming=False uses original chat() path."""

    @pytest.mark.asyncio
    async def test_disabled_uses_chat_path(self, tmp_path) -> None:
        bus = MessageBus()
        long_text = "B" * 100
        provider = StreamingFakeProvider([LLMResponse(content=long_text)])
        registry = ToolRegistry()
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions, streaming=False)

        captured: list[Envelope] = []

        async def capture(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", capture)

        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="go")
        )

        agent_task = asyncio.create_task(agent.run())
        dispatch_task = asyncio.create_task(bus.dispatch_outbound())
        await asyncio.sleep(0.5)
        agent.stop()
        bus.stop()
        agent_task.cancel()
        dispatch_task.cancel()
        try:
            await asyncio.gather(agent_task, dispatch_task)
        except asyncio.CancelledError:
            pass

        # No streaming chunks even for long text
        streaming = [e for e in captured if (e.metadata or {}).get("streaming")]
        assert len(streaming) == 0
        assert len(captured) == 1
        assert captured[0].content == long_text


class TestTelegramStreamingChannel:
    """Telegram channel handles streaming edit-in-place."""

    @pytest.mark.asyncio
    async def test_stream_edit_in_place(self) -> None:
        from nibot.channels.telegram import TelegramChannel
        from nibot.config import TelegramChannelConfig

        bus = MessageBus()
        config = TelegramChannelConfig(token="fake", enabled=True)
        ch = TelegramChannel(config, bus)

        # Mock telegram bot
        sent_msg = MagicMock()
        sent_msg.message_id = 42
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(return_value=sent_msg)
        mock_bot.edit_message_text = AsyncMock()

        ch._app = MagicMock()
        ch._app.bot = mock_bot

        # First chunk: sends new message
        await ch.send(Envelope(
            channel="telegram", chat_id="123", sender_id="assistant",
            content="Hello", metadata={"streaming": True, "stream_seq": 0},
        ))
        mock_bot.send_message.assert_called_once_with(chat_id=123, text="Hello")
        assert ch._stream_msgs[123] == 42

        # Second chunk: edits existing message
        await ch.send(Envelope(
            channel="telegram", chat_id="123", sender_id="assistant",
            content="Hello world!", metadata={"streaming": True, "stream_seq": 1, "stream_done": True},
        ))
        mock_bot.edit_message_text.assert_called_once_with(
            chat_id=123, message_id=42, text="Hello world!",
        )
        # stream_done cleans up tracking
        assert 123 not in ch._stream_msgs

    @pytest.mark.asyncio
    async def test_non_streaming_message_works_normally(self) -> None:
        from nibot.channels.telegram import TelegramChannel
        from nibot.config import TelegramChannelConfig

        bus = MessageBus()
        config = TelegramChannelConfig(token="fake", enabled=True)
        ch = TelegramChannel(config, bus)

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        ch._app = MagicMock()
        ch._app.bot = mock_bot

        await ch.send(Envelope(
            channel="telegram", chat_id="123", sender_id="assistant",
            content="Normal message", metadata={},
        ))
        mock_bot.send_message.assert_called_once()


class TestFeishuStreamingChannel:
    """Feishu ignores intermediate streaming chunks."""

    @pytest.mark.asyncio
    async def test_ignores_intermediate_chunks(self) -> None:
        from nibot.channels.feishu import FeishuChannel
        from nibot.config import FeishuChannelConfig

        bus = MessageBus()
        config = FeishuChannelConfig(app_id="a", app_secret="s", enabled=True)
        ch = FeishuChannel(config, bus)
        ch._client = None  # No client -- send() returns early

        # Intermediate chunk -- should return without error
        await ch.send(Envelope(
            channel="feishu", chat_id="c1", sender_id="assistant",
            content="partial", metadata={"streaming": True, "stream_seq": 0},
        ))
        # No error means early return worked


class TestAPIStreamingChannel:
    """API channel ignores streaming chunks."""

    @pytest.mark.asyncio
    async def test_ignores_streaming(self) -> None:
        from nibot.channels.api import APIChannel
        from nibot.config import NiBotConfig

        bus = MessageBus()
        config = NiBotConfig()
        ch = APIChannel(config, bus)

        # Streaming chunk -- should be silently ignored
        await ch.send(Envelope(
            channel="api", chat_id="c1", sender_id="assistant",
            content="chunk", metadata={"streaming": True, "stream_seq": 0},
        ))
        # If it reached resolve_response, it would fail since no waiter exists
        # No error means early return worked
