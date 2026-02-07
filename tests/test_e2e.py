"""End-to-end tests: full message flow through Bus -> AgentLoop -> outbound."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from nibot.agent import AgentLoop
from nibot.bus import MessageBus
from nibot.config import NiBotConfig
from nibot.provider import LLMProvider
from nibot.registry import Tool, ToolRegistry
from nibot.session import Session, SessionManager
from nibot.types import Envelope, LLMResponse, ToolCall


class FakeProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses: list[LLMResponse] = responses or []
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str = "", max_tokens: int = 4096, temperature: float = 0.7) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            return LLMResponse(content="(no more responses)")
        return self.responses.pop(0)


class FakeContextBuilder:
    def build(self, session: Session, current: Envelope) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": "You are a test bot."}]
        for m in session.messages[-10:]:
            msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        msgs.append({"role": "user", "content": current.content})
        return msgs


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo the input"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, **kwargs: Any) -> str:
        return f"echo: {kwargs.get('text', '')}"


class FailTool(Tool):
    @property
    def name(self) -> str:
        return "fail_tool"

    @property
    def description(self) -> str:
        return "Always fails"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("intentional failure")


def _make_agent(
    bus: MessageBus,
    provider: FakeProvider,
    registry: ToolRegistry,
    sessions: SessionManager,
    rate_limiter: Any = None,
) -> AgentLoop:
    config = NiBotConfig()
    return AgentLoop(
        bus=bus,
        provider=provider,
        registry=registry,
        sessions=sessions,
        context_builder=FakeContextBuilder(),
        config=config,
        rate_limiter=rate_limiter,
    )


class TestE2ESimpleConversation:
    """Scenario 1: Simple conversation without tool calls."""

    @pytest.mark.asyncio
    async def test_message_flows_through_bus_agent_outbound(self, tmp_path) -> None:
        bus = MessageBus()
        provider = FakeProvider([LLMResponse(content="Hello back!")])
        registry = ToolRegistry()
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions)

        # Set up outbound capture
        captured: list[Envelope] = []

        async def capture(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", capture)

        # Publish inbound message
        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="Hello")
        )

        # Run agent + dispatch in parallel, stop after first response
        agent_task = asyncio.create_task(agent.run())
        dispatch_task = asyncio.create_task(bus.dispatch_outbound())

        await asyncio.sleep(0.5)
        agent.stop()
        bus.stop()
        agent_task.cancel()
        dispatch_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        try:
            await dispatch_task
        except asyncio.CancelledError:
            pass

        assert len(captured) == 1
        assert captured[0].content == "Hello back!"
        assert captured[0].channel == "test"

        # Verify provider received the message
        assert len(provider.calls) == 1


class TestE2EWithToolCalls:
    """Scenario 2: Conversation with tool calls."""

    @pytest.mark.asyncio
    async def test_tool_execution_and_response(self, tmp_path) -> None:
        bus = MessageBus()
        # First LLM response requests a tool call, second gives final answer
        provider = FakeProvider([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="echo", arguments={"text": "world"})],
            ),
            LLMResponse(content="The echo returned: echo: world"),
        ])
        registry = ToolRegistry()
        registry.register(EchoTool())
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions)

        captured: list[Envelope] = []

        async def capture(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", capture)

        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="Echo test")
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

        assert len(captured) == 1
        assert "echo: world" in captured[0].content

        # Provider called twice (tool call + final)
        assert len(provider.calls) == 2


class TestE2EToolError:
    """Scenario 3: Tool throws exception, agent recovers gracefully."""

    @pytest.mark.asyncio
    async def test_tool_error_recovery(self, tmp_path) -> None:
        bus = MessageBus()
        provider = FakeProvider([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fail_tool", arguments={})],
            ),
            LLMResponse(content="Sorry, the tool failed. Let me try differently."),
        ])
        registry = ToolRegistry()
        registry.register(FailTool())
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions)

        captured: list[Envelope] = []

        async def capture(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", capture)

        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="Do something")
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

        assert len(captured) == 1
        # Agent should provide a response despite tool failure
        assert captured[0].content != ""


class TestE2ESessionPersistence:
    """Scenario 4: Session persists messages across calls."""

    @pytest.mark.asyncio
    async def test_messages_saved_to_session(self, tmp_path) -> None:
        bus = MessageBus()
        provider = FakeProvider([LLMResponse(content="Got it.")])
        registry = ToolRegistry()
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions)

        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="Remember this")
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

        # Check session was saved with messages
        session = sessions.get_or_create("test:c1")
        assert len(session.messages) >= 2  # at least user + assistant
        roles = [m["role"] for m in session.messages]
        assert "user" in roles
        assert "assistant" in roles


class TestE2ERateLimiting:
    """Scenario 5: Rate-limited messages get rejected."""

    @pytest.mark.asyncio
    async def test_rate_limited_message_rejected(self, tmp_path) -> None:
        from nibot.rate_limiter import SlidingWindowRateLimiter, RateLimitConfig

        bus = MessageBus()
        provider = FakeProvider([LLMResponse(content="ok")] * 5)
        registry = ToolRegistry()
        sessions = SessionManager(tmp_path / "sessions")

        rl = SlidingWindowRateLimiter(RateLimitConfig(per_user_rpm=2, per_channel_rpm=100, enabled=True))
        agent = _make_agent(bus, provider, registry, sessions, rate_limiter=rl)

        captured: list[Envelope] = []

        async def capture(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", capture)

        # Send 3 messages (limit is 2/min)
        for i in range(3):
            await bus.publish_inbound(
                Envelope(channel="test", chat_id="c1", sender_id="user1", content=f"msg{i}")
            )

        agent_task = asyncio.create_task(agent.run())
        dispatch_task = asyncio.create_task(bus.dispatch_outbound())
        await asyncio.sleep(1.0)
        agent.stop()
        bus.stop()
        agent_task.cancel()
        dispatch_task.cancel()
        try:
            await asyncio.gather(agent_task, dispatch_task)
        except asyncio.CancelledError:
            pass

        assert len(captured) == 3
        # Third message should be rate-limited
        rate_limited = [e for e in captured if "rate limit" in e.content.lower()]
        assert len(rate_limited) >= 1


class TestE2EMultiTurnToolCalls:
    """Scenario 6: Multiple tool calls in one message."""

    @pytest.mark.asyncio
    async def test_multi_tool_iteration(self, tmp_path) -> None:
        bus = MessageBus()
        provider = FakeProvider([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="echo", arguments={"text": "first"})],
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="echo", arguments={"text": "second"})],
            ),
            LLMResponse(content="Done! Called echo twice."),
        ])
        registry = ToolRegistry()
        registry.register(EchoTool())
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions)

        captured: list[Envelope] = []

        async def capture(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", capture)

        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="Call echo twice")
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

        assert len(captured) == 1
        assert "Done" in captured[0].content
        # Provider called 3 times (2 tool rounds + final)
        assert len(provider.calls) == 3
