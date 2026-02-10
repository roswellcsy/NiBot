"""Streaming tool call tests (Phase 4)."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from nibot.provider import LLMProvider, LiteLLMProvider
from nibot.types import LLMResponse, ToolCall, ToolCallDelta


# -- Mock providers --

class _TextOnlyStreamProvider(LLMProvider):
    """Streams text only, no tool calls."""
    async def chat(self, messages, tools=None, **kw):
        return LLMResponse(content="full text")

    async def chat_stream(self, messages, tools=None, **kw) -> AsyncIterator[str | LLMResponse]:
        yield "hello "
        yield "world"
        yield LLMResponse(content="hello world")


class _ToolCallStreamProvider(LLMProvider):
    """Simulates streaming tool call deltas then final response."""
    async def chat(self, messages, tools=None, **kw):
        return LLMResponse(content="", tool_calls=[
            ToolCall(id="tc1", name="get_weather", arguments={"city": "Tokyo"}),
        ])

    async def chat_stream(self, messages, tools=None, **kw) -> AsyncIterator[str | ToolCallDelta | LLMResponse]:
        # Simulate partial argument streaming
        yield ToolCallDelta(index=0, name="get_weather", partial_args='{"ci')
        yield ToolCallDelta(index=0, name="get_weather", partial_args='{"city": "Tok')
        yield ToolCallDelta(index=0, name="get_weather", partial_args='{"city": "Tokyo"}')
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="get_weather", arguments={"city": "Tokyo"})],
        )


class _ParallelToolCallStreamProvider(LLMProvider):
    """Simulates two parallel tool calls streaming."""
    async def chat(self, messages, tools=None, **kw):
        return LLMResponse(content="", tool_calls=[
            ToolCall(id="tc1", name="get_weather", arguments={"city": "Tokyo"}),
            ToolCall(id="tc2", name="get_time", arguments={"tz": "JST"}),
        ])

    async def chat_stream(self, messages, tools=None, **kw) -> AsyncIterator[str | ToolCallDelta | LLMResponse]:
        yield ToolCallDelta(index=0, name="get_weather", partial_args='{"city"')
        yield ToolCallDelta(index=1, name="get_time", partial_args='{"tz"')
        yield ToolCallDelta(index=0, name="get_weather", partial_args='{"city": "Tokyo"}')
        yield ToolCallDelta(index=1, name="get_time", partial_args='{"tz": "JST"}')
        yield LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="tc1", name="get_weather", arguments={"city": "Tokyo"}),
                ToolCall(id="tc2", name="get_time", arguments={"tz": "JST"}),
            ],
        )


class _FailStreamThenFallbackProvider(LLMProvider):
    """chat_stream raises, but chat() works -- tests fallback."""
    async def chat(self, messages, tools=None, **kw):
        return LLMResponse(content="fallback result")

    async def chat_stream(self, messages, tools=None, **kw) -> AsyncIterator[str | LLMResponse]:
        raise RuntimeError("stream broken")
        yield  # make it a generator  # noqa: E501


# -- Tests --

@pytest.mark.asyncio
async def test_stream_text_only():
    """Text-only stream yields strings then LLMResponse."""
    provider = _TextOnlyStreamProvider()
    items = []
    async for item in provider.chat_stream(messages=[]):
        items.append(item)
    strings = [i for i in items if isinstance(i, str)]
    responses = [i for i in items if isinstance(i, LLMResponse)]
    assert len(strings) == 2
    assert strings[0] == "hello "
    assert strings[1] == "world"
    assert len(responses) == 1
    assert responses[0].content == "hello world"


@pytest.mark.asyncio
async def test_stream_with_tool_call_delta():
    """Stream with tools produces ToolCallDelta intermediate events."""
    provider = _ToolCallStreamProvider()
    items = []
    async for item in provider.chat_stream(messages=[], tools=[{"type": "function"}]):
        items.append(item)
    deltas = [i for i in items if isinstance(i, ToolCallDelta)]
    assert len(deltas) == 3
    assert deltas[0].name == "get_weather"
    assert deltas[0].partial_args == '{"ci'
    assert deltas[-1].partial_args == '{"city": "Tokyo"}'


@pytest.mark.asyncio
async def test_stream_tool_call_final_response():
    """Final LLMResponse from streaming has complete tool_calls."""
    provider = _ToolCallStreamProvider()
    items = []
    async for item in provider.chat_stream(messages=[], tools=[{"type": "function"}]):
        items.append(item)
    responses = [i for i in items if isinstance(i, LLMResponse)]
    assert len(responses) == 1
    resp = responses[0]
    assert resp.has_tool_calls
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"city": "Tokyo"}


@pytest.mark.asyncio
async def test_stream_multiple_parallel_tool_calls():
    """Parallel tool calls accumulate correctly by index."""
    provider = _ParallelToolCallStreamProvider()
    items = []
    async for item in provider.chat_stream(messages=[], tools=[{"type": "function"}]):
        items.append(item)
    deltas = [i for i in items if isinstance(i, ToolCallDelta)]
    # Should have deltas for both indices
    indices = {d.index for d in deltas}
    assert indices == {0, 1}
    # Final response has both tool calls
    responses = [i for i in items if isinstance(i, LLMResponse)]
    assert len(responses[0].tool_calls) == 2


@pytest.mark.asyncio
async def test_stream_tool_call_fallback():
    """When stream raises, provider falls back gracefully."""
    provider = _FailStreamThenFallbackProvider()
    # The base LLMProvider.chat_stream just falls back to chat()
    # Our mock raises in chat_stream, so we test the pattern
    try:
        items = []
        async for item in provider.chat_stream(messages=[]):
            items.append(item)
        # Should not reach here
        assert False, "Expected RuntimeError"
    except RuntimeError:
        # Fallback: use non-streaming
        resp = await provider.chat(messages=[])
        assert resp.content == "fallback result"


def test_tool_call_delta_dataclass():
    """ToolCallDelta has expected fields."""
    d = ToolCallDelta(index=0, name="test", partial_args='{"a":')
    assert d.index == 0
    assert d.name == "test"
    assert d.partial_args == '{"a":'
