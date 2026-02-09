"""Complex processing: multi-turn context, concurrency, tool chains,
progress events, subagent flow, and max-iterations boundary."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nibot.agent import AgentLoop
from nibot.bus import MessageBus
from nibot.config import NiBotConfig
from nibot.provider import LLMProvider
from nibot.registry import Tool, ToolRegistry
from nibot.session import Session, SessionManager
from nibot.subagent import SubagentManager
from nibot.types import Envelope, LLMResponse, ToolCall


# ---- Test doubles ----


class _Provider(LLMProvider):
    """Pop-based fake LLM. Records all calls."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses: list[LLMResponse] = list(responses or [])
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None, **kw: Any) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0) if self.responses else LLMResponse(content="(empty)")


class _SlowProvider(_Provider):
    """Adds 0.1s latency per call for concurrency timing tests."""

    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None, **kw: Any) -> LLMResponse:
        await asyncio.sleep(0.1)
        return await super().chat(messages, tools, **kw)


class _CtxBuilder:
    """Includes session.messages[-10:] + current user message."""

    def build(self, session: Session, current: Envelope) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": "test bot"}]
        for m in session.messages[-10:]:
            msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        msgs.append({"role": "user", "content": current.content})
        return msgs


class _Echo(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo input"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, **kw: Any) -> str:
        return f"echo: {kw.get('text', '')}"


class _Transform(Tool):
    @property
    def name(self) -> str:
        return "transform"

    @property
    def description(self) -> str:
        return "Transform input"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}

    async def execute(self, **kw: Any) -> str:
        return f"transformed({kw.get('input', '')})"


def _make_agent(bus, provider, registry, sessions, config=None, **kw):
    return AgentLoop(
        bus=bus, provider=provider, registry=registry, sessions=sessions,
        context_builder=_CtxBuilder(), config=config or NiBotConfig(), **kw,
    )


async def _run_for(agent, bus, secs=0.5):
    """Run agent + dispatch for `secs` then tear down."""
    at = asyncio.create_task(agent.run())
    dt = asyncio.create_task(bus.dispatch_outbound())
    await asyncio.sleep(secs)
    agent.stop()
    bus.stop()
    at.cancel()
    dt.cancel()
    for t in (at, dt):
        try:
            await t
        except asyncio.CancelledError:
            pass


async def _dispatch_then_stop(bus, secs=0.1):
    """Start dispatch, sleep, stop. For use after _handle calls."""
    dt = asyncio.create_task(bus.dispatch_outbound())
    await asyncio.sleep(secs)
    bus.stop()
    dt.cancel()
    try:
        await dt
    except asyncio.CancelledError:
        pass


# ---- P0 #1: Multi-turn Context ----


class TestMultiTurnContext:

    @pytest.mark.asyncio
    async def test_third_message_sees_full_history(self, tmp_path) -> None:
        """Three sequential messages to same session: 3rd LLM call includes all prior history."""
        bus = MessageBus()
        provider = _Provider([
            LLMResponse(content="reply_1"),
            LLMResponse(content="reply_2"),
            LLMResponse(content="reply_3"),
        ])
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, ToolRegistry(), sessions)

        for content in ["msg_1", "msg_2", "msg_3"]:
            await agent._handle(
                Envelope(channel="test", chat_id="c1", sender_id="user1", content=content)
            )

        # Third LLM call should see history from turns 1 and 2
        third_call_msgs = provider.calls[2]
        non_system = [m for m in third_call_msgs if m["role"] != "system"]

        # Expected: user(msg_1) + assistant(reply_1) + user(msg_2) + assistant(reply_2) + user(msg_3)
        assert len(non_system) >= 5
        contents = [m["content"] for m in non_system]
        assert "msg_1" in contents
        assert "reply_1" in contents
        assert "msg_2" in contents
        assert "reply_2" in contents
        assert "msg_3" in contents


# ---- P0 #2: Concurrent Messages ----


class TestConcurrentMessages:

    @pytest.mark.asyncio
    async def test_different_sessions_parallel(self, tmp_path) -> None:
        """Messages to different sessions are processed in parallel, not serially."""
        bus = MessageBus()
        provider = _SlowProvider([LLMResponse(content=f"r{i}") for i in range(5)])
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, ToolRegistry(), sessions)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        for i in range(5):
            await bus.publish_inbound(
                Envelope(channel="test", chat_id=f"c{i}", sender_id="user1", content=f"msg{i}")
            )

        # 0.35s budget: parallel (5 * 0.1s concurrent = ~0.15s) fits;
        # serial (5 * 0.1s = 0.5s) would only complete ~3 messages.
        await _run_for(agent, bus, 0.35)
        assert len(captured) == 5

    @pytest.mark.asyncio
    async def test_same_session_serialized(self, tmp_path) -> None:
        """Messages to same session are serialized by session lock."""
        bus = MessageBus()
        provider = _SlowProvider([LLMResponse(content=f"r{i}") for i in range(3)])
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, ToolRegistry(), sessions)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        for i in range(3):
            await bus.publish_inbound(
                Envelope(channel="test", chat_id="same", sender_id="user1", content=f"msg{i}")
            )

        await _run_for(agent, bus, 1.0)

        assert len(captured) == 3

        session = sessions.get_or_create("test:same")
        assert len(session.messages) == 6
        roles = [m["role"] for m in session.messages]
        assert roles == ["user", "assistant"] * 3
        user_msgs = [m["content"] for m in session.messages if m["role"] == "user"]
        assert set(user_msgs) == {"msg0", "msg1", "msg2"}


# ---- P0 #3: Tool Chains ----


class TestToolChains:

    @pytest.mark.asyncio
    async def test_three_step_tool_chain(self, tmp_path) -> None:
        """Tool A output feeds Tool B input across 3 sequential iterations."""
        bus = MessageBus()
        provider = _Provider([
            LLMResponse(content="", tool_calls=[
                ToolCall(id="t1", name="transform", arguments={"input": "raw"}),
            ]),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="t2", name="transform", arguments={"input": "transformed(raw)"}),
            ]),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="t3", name="transform",
                         arguments={"input": "transformed(transformed(raw))"}),
            ]),
            LLMResponse(content="Chain complete"),
        ])
        registry = ToolRegistry()
        registry.register(_Transform())
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="chain test")
        )
        await _run_for(agent, bus, 0.5)

        assert len(provider.calls) == 4

        # Third call sees tool results from steps 1 and 2
        third_call = provider.calls[2]
        tool_results = [m for m in third_call if m.get("role") == "tool"]
        assert len(tool_results) == 2
        assert "transformed(raw)" in tool_results[0]["content"]
        assert "transformed(transformed(raw))" in tool_results[1]["content"]

        assert len(captured) == 1
        assert "Chain complete" in captured[0].content

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_in_single_response(self, tmp_path) -> None:
        """Two tool calls in one LLM response: both executed, both results visible."""
        bus = MessageBus()
        provider = _Provider([
            LLMResponse(content="", tool_calls=[
                ToolCall(id="t1", name="echo", arguments={"text": "aaa"}),
                ToolCall(id="t2", name="echo", arguments={"text": "bbb"}),
            ]),
            LLMResponse(content="Both echoed"),
        ])
        registry = ToolRegistry()
        registry.register(_Echo())
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        await bus.publish_inbound(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="parallel tools")
        )
        await _run_for(agent, bus, 0.5)

        assert len(provider.calls) == 2

        second_call = provider.calls[1]
        tool_results = [m for m in second_call if m.get("role") == "tool"]
        assert len(tool_results) == 2
        contents = {r["content"] for r in tool_results}
        assert "echo: aaa" in contents
        assert "echo: bbb" in contents


# ---- P1 #4: Progress Events from AgentLoop ----


class TestProgressEventsFromAgent:

    @pytest.mark.asyncio
    async def test_thinking_event_with_stream_id(self, tmp_path) -> None:
        """stream_id in metadata triggers thinking progress events."""
        bus = MessageBus()
        provider = _Provider([LLMResponse(content="hello")])
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, ToolRegistry(), sessions)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        await agent._handle(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="hi",
                     metadata={"stream_id": "sid1"})
        )
        await _dispatch_then_stop(bus)

        assert len(captured) == 2
        assert captured[0].metadata.get("progress") == "thinking"
        assert captured[0].metadata.get("iteration") == 1
        assert captured[1].content == "hello"

    @pytest.mark.asyncio
    async def test_tool_progress_events(self, tmp_path) -> None:
        """Tool calls emit thinking + tool_start + tool_done + thinking(next iter) events."""
        bus = MessageBus()
        provider = _Provider([
            LLMResponse(content="", tool_calls=[
                ToolCall(id="t1", name="echo", arguments={"text": "data"}),
            ]),
            LLMResponse(content="done"),
        ])
        registry = ToolRegistry()
        registry.register(_Echo())
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, registry, sessions)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        await agent._handle(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="go",
                     metadata={"stream_id": "sid1"})
        )
        await _dispatch_then_stop(bus)

        progress = [e for e in captured if e.metadata.get("progress")]
        assert len(progress) == 4
        types = [e.metadata["progress"] for e in progress]
        assert types == ["thinking", "tool_start", "tool_done", "thinking"]

        assert progress[1].metadata["tool_name"] == "echo"
        assert "elapsed" in progress[2].metadata

        final = [e for e in captured if not e.metadata.get("progress")]
        assert len(final) == 1
        assert final[0].content == "done"

    @pytest.mark.asyncio
    async def test_no_progress_without_stream_id(self, tmp_path) -> None:
        """Without stream_id, no progress events are emitted."""
        bus = MessageBus()
        provider = _Provider([LLMResponse(content="plain")])
        sessions = SessionManager(tmp_path / "sessions")
        agent = _make_agent(bus, provider, ToolRegistry(), sessions)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        await agent._handle(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="hi")
        )
        await _dispatch_then_stop(bus)

        assert len(captured) == 1
        assert captured[0].content == "plain"
        assert not captured[0].metadata.get("progress")


# ---- P1 #5: Subagent Flow ----


class TestSubagentFlow:

    @pytest.mark.asyncio
    async def test_subagent_completes_and_publishes(self) -> None:
        """Subagent runs LLM, publishes result to bus, updates task_info."""
        bus = MessageBus()
        provider = _Provider([LLMResponse(content="subagent result")])
        mgr = SubagentManager(provider=provider, registry=ToolRegistry(), bus=bus)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        dt = asyncio.create_task(bus.dispatch_outbound())
        task_id = await mgr.spawn(
            task="do something", label="test-sub",
            origin_channel="test", origin_chat_id="c1",
        )
        await asyncio.sleep(0.5)
        bus.stop()
        dt.cancel()
        try:
            await dt
        except asyncio.CancelledError:
            pass

        assert len(captured) == 1
        assert "test-sub" in captured[0].content
        assert "subagent result" in captured[0].content

        info = mgr.get_task_info(task_id)
        assert info is not None
        assert info.status == "completed"

    @pytest.mark.asyncio
    async def test_subagent_with_tool_calls(self) -> None:
        """Subagent executes tools during its LLM loop."""
        bus = MessageBus()
        provider = _Provider([
            LLMResponse(content="", tool_calls=[
                ToolCall(id="t1", name="echo", arguments={"text": "sub_data"}),
            ]),
            LLMResponse(content="sub done"),
        ])
        registry = ToolRegistry()
        registry.register(_Echo())
        mgr = SubagentManager(provider=provider, registry=registry, bus=bus)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        dt = asyncio.create_task(bus.dispatch_outbound())
        await mgr.spawn(
            task="use echo", label="tool-sub",
            origin_channel="test", origin_chat_id="c1",
        )
        await asyncio.sleep(0.5)
        bus.stop()
        dt.cancel()
        try:
            await dt
        except asyncio.CancelledError:
            pass

        assert len(captured) == 1
        assert "sub done" in captured[0].content
        assert len(provider.calls) == 2

    @pytest.mark.asyncio
    async def test_subagent_error_sets_status(self) -> None:
        """LLM error sets task_info.status to 'error' but still publishes result."""
        bus = MessageBus()

        class _ErrorProv(LLMProvider):
            async def chat(self, messages: list[dict[str, Any]],
                           tools: list[dict[str, Any]] | None = None, **kw: Any) -> LLMResponse:
                raise RuntimeError("LLM down")

        mgr = SubagentManager(provider=_ErrorProv(), registry=ToolRegistry(), bus=bus)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        dt = asyncio.create_task(bus.dispatch_outbound())
        task_id = await mgr.spawn(
            task="fail", label="error-sub",
            origin_channel="test", origin_chat_id="c1",
        )
        await asyncio.sleep(0.5)
        bus.stop()
        dt.cancel()
        try:
            await dt
        except asyncio.CancelledError:
            pass

        info = mgr.get_task_info(task_id)
        assert info is not None
        assert info.status == "error"
        assert len(captured) == 1
        assert "error" in captured[0].content.lower()


# ---- P1 #6: Max Iterations ----


class TestMaxIterations:

    @pytest.mark.asyncio
    async def test_agent_stops_at_max_iterations(self, tmp_path) -> None:
        """Agent terminates after max_iterations even if LLM keeps returning tool calls."""
        bus = MessageBus()
        provider = _Provider([
            LLMResponse(content="", tool_calls=[
                ToolCall(id=f"t{i}", name="echo", arguments={"text": f"step{i}"}),
            ])
            for i in range(10)
        ])
        registry = ToolRegistry()
        registry.register(_Echo())
        sessions = SessionManager(tmp_path / "sessions")
        config = NiBotConfig()
        config.agent.max_iterations = 3
        agent = _make_agent(bus, provider, registry, sessions, config=config)

        captured: list[Envelope] = []

        async def cap(env: Envelope) -> None:
            captured.append(env)

        bus.subscribe_outbound("test", cap)

        await agent._handle(
            Envelope(channel="test", chat_id="c1", sender_id="user1", content="loop forever")
        )
        await _dispatch_then_stop(bus)

        assert len(provider.calls) == 3
        assert len(captured) == 1
        assert "unable to complete" in captured[0].content.lower()

        session = sessions.get_or_create("test:c1")
        assert len(session.messages) > 0
