"""Tests for v0.2.0 features: bug fixes, architecture improvements, production features."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from nibot.agent import AgentLoop
from nibot.bus import MessageBus
from nibot.config import NiBotConfig
from nibot.context import ContextBuilder, _estimate_tokens
from nibot.memory import MemoryStore
from nibot.provider import LiteLLMProvider
from nibot.registry import Tool, ToolRegistry
from nibot.session import Session, SessionManager
from nibot.skills import SkillsLoader
from nibot.types import Envelope, LLMResponse, ToolCall, ToolContext, ToolResult


# ---- Test doubles ----

class DummyTool(Tool):
    def __init__(self, name: str = "dummy") -> None:
        self._name = name
        self._last_ctx: ToolContext | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A dummy tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"x": {"type": "string"}}}

    def receive_context(self, ctx: ToolContext) -> None:
        self._last_ctx = ctx

    async def execute(self, **kwargs: Any) -> str:
        kwargs.pop("_tool_ctx", None)  # backward compat: ignore if present
        return f"ok:{kwargs.get('x', '')}"


class FakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[Any] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return self._responses.pop(0) if self._responses else LLMResponse(content="done")


class DummyContextBuilder:
    def build(self, session: Any, current: Any) -> list[dict[str, Any]]:
        history = session.get_history()
        return [
            {"role": "system", "content": "sys"},
            *history,
            {"role": "user", "content": current.content},
        ]


# ---- Phase 1: Bug fixes ----

class TestToolResultCallId:
    """1.1: ToolResult.call_id propagation."""

    @pytest.mark.asyncio
    async def test_registry_execute_propagates_call_id(self) -> None:
        reg = ToolRegistry()
        reg.register(DummyTool())
        result = await reg.execute("dummy", {"x": "1"}, call_id="call_abc")
        assert result.call_id == "call_abc"

    @pytest.mark.asyncio
    async def test_registry_execute_unknown_tool_preserves_call_id(self) -> None:
        reg = ToolRegistry()
        result = await reg.execute("nonexistent", {}, call_id="call_xyz")
        assert result.call_id == "call_xyz"
        assert result.is_error


class TestErrorNotification:
    """1.3: Agent sends error envelope to user on exception."""

    @pytest.mark.asyncio
    async def test_agent_publishes_error_envelope(self) -> None:
        bus = MessageBus()

        class ExplodingProvider:
            async def chat(self, **kwargs: Any) -> LLMResponse:
                raise RuntimeError("boom")

        reg = ToolRegistry()
        sessions = SessionManager(Path("nonexistent_sessions"))
        cfg = NiBotConfig()
        loop = AgentLoop(bus, ExplodingProvider(), reg, sessions, DummyContextBuilder(), cfg)

        # Simulate: start run loop, feed one message, check outbound
        envelope = Envelope(channel="test", chat_id="1", sender_id="u", content="hello")
        await bus.publish_inbound(envelope)

        # Run one iteration
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.1)
        loop.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have an error envelope in outbound
        msg = await asyncio.wait_for(bus._outbound.get(), timeout=1.0)
        assert "error" in msg.content.lower()
        assert msg.channel == "test"


class TestMaxIterationsNotification:
    """1.4: Agent returns fallback content when max_iterations exhausted."""

    @pytest.mark.asyncio
    async def test_exhaustion_returns_fallback(self, tmp_path: Path) -> None:
        bus = MessageBus()
        responses = [
            LLMResponse(
                content="again",
                tool_calls=[ToolCall(id=f"t{i}", name="dummy", arguments={"x": str(i)})],
            )
            for i in range(10)
        ]
        provider = FakeProvider(responses)
        reg = ToolRegistry()
        reg.register(DummyTool())
        sessions = SessionManager(tmp_path / "sessions")
        cfg = NiBotConfig()
        cfg.agent.max_iterations = 2

        loop = AgentLoop(bus, provider, reg, sessions, DummyContextBuilder(), cfg)
        out = await loop._process(Envelope(channel="t", chat_id="1", sender_id="u", content="q"))
        assert "unable to complete" in out.content


class TestSessionCompleteThread:
    """1.2: Session saves complete message thread including tool calls/results."""

    @pytest.mark.asyncio
    async def test_session_saves_tool_messages(self, tmp_path: Path) -> None:
        bus = MessageBus()
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="dummy", arguments={"x": "val"})],
            ),
            LLMResponse(content="final answer"),
        ]
        provider = FakeProvider(responses)
        reg = ToolRegistry()
        reg.register(DummyTool())
        sessions = SessionManager(tmp_path / "sessions")
        cfg = NiBotConfig()

        loop = AgentLoop(bus, provider, reg, sessions, DummyContextBuilder(), cfg)
        await loop._process(Envelope(channel="ch", chat_id="1", sender_id="u", content="go"))

        session = sessions.get_or_create("ch:1")
        history = session.get_history()

        # Should have: user, assistant(tool_calls), tool(result), assistant(final)
        roles = [m["role"] for m in history]
        assert "tool" in roles
        # The tool message should have tool_call_id
        tool_msgs = [m for m in history if m["role"] == "tool"]
        assert tool_msgs[0]["tool_call_id"] == "tc1"

    def test_get_history_preserves_extra_fields(self) -> None:
        session = Session(key="test")
        session.add_message("assistant", "", tool_calls=[{"id": "x", "type": "function"}])
        session.add_message("tool", "result", tool_call_id="x", name="dummy")
        history = session.get_history()
        assert "tool_calls" in history[0]
        assert history[1]["tool_call_id"] == "x"


# ---- Phase 2: Architecture improvements ----

class TestToolExecutionContext:
    """2.1: ToolContext passed to tools via registry."""

    @pytest.mark.asyncio
    async def test_tool_receives_context(self) -> None:
        reg = ToolRegistry()
        tool = DummyTool()
        reg.register(tool)
        ctx = ToolContext(channel="tg", chat_id="42", session_key="tg:42")
        await reg.execute("dummy", {"x": "1"}, call_id="c1", ctx=ctx)
        assert tool._last_ctx is not None
        assert tool._last_ctx.channel == "tg"
        assert tool._last_ctx.chat_id == "42"

    @pytest.mark.asyncio
    async def test_tool_without_context_still_works(self) -> None:
        reg = ToolRegistry()
        tool = DummyTool()
        reg.register(tool)
        result = await reg.execute("dummy", {"x": "1"}, call_id="c1")
        assert not result.is_error
        assert tool._last_ctx is None


class TestConfigEnvPriority:
    """2.2: Environment variables override file config."""

    def test_env_overrides_init_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NIBOT_AGENT__NAME", "FromEnv")
        cfg = NiBotConfig(agent={"name": "FromFile"})
        assert cfg.agent.name == "FromEnv"


class TestBootstrapFilesConfigurable:
    """2.4: BOOTSTRAP_FILES configurable via AgentConfig."""

    def test_default_bootstrap_files(self) -> None:
        cfg = NiBotConfig()
        assert "IDENTITY.md" in cfg.agent.bootstrap_files
        assert len(cfg.agent.bootstrap_files) == 5

    def test_custom_bootstrap_files(self) -> None:
        cfg = NiBotConfig(agent={"bootstrap_files": ["CUSTOM.md"]})
        assert cfg.agent.bootstrap_files == ["CUSTOM.md"]

    def test_context_builder_uses_config_bootstrap(self, tmp_path: Path) -> None:
        (tmp_path / "CUSTOM.md").write_text("Custom identity", encoding="utf-8")
        cfg = NiBotConfig(agent={"bootstrap_files": ["CUSTOM.md"]})
        memory = MemoryStore(tmp_path / "memory")
        skills = SkillsLoader([tmp_path / "skills"])
        cb = ContextBuilder(config=cfg, memory=memory, skills=skills, workspace=tmp_path)
        session = Session(key="test")
        msgs = cb.build(session, Envelope(channel="t", chat_id="1", sender_id="u", content="hi"))
        assert "Custom identity" in msgs[0]["content"]


class TestTokenBudget:
    """2.3: Token budget management in context builder."""

    def test_estimate_tokens_fallback(self) -> None:
        msgs = [{"role": "user", "content": "hello world"}]
        tokens = _estimate_tokens(msgs)
        assert tokens > 0

    def test_long_history_truncated(self, tmp_path: Path) -> None:
        cfg = NiBotConfig(agent={"context_window": 200, "context_reserve": 50})
        memory = MemoryStore(tmp_path / "memory")
        skills = SkillsLoader([tmp_path / "skills"])
        cb = ContextBuilder(config=cfg, memory=memory, skills=skills, workspace=tmp_path)

        session = Session(key="test")
        # Add lots of history
        for i in range(100):
            session.add_message("user", f"message {i} " * 20)
            session.add_message("assistant", f"response {i} " * 20)

        msgs = cb.build(session, Envelope(channel="t", chat_id="1", sender_id="u", content="now"))
        # Should have fewer history messages than the full 200
        history_msgs = [m for m in msgs if m["role"] not in ("system",)]
        # last message is always user "now", the rest is truncated history
        assert len(history_msgs) < 200


class TestCleanup:
    """2.5: Cleanup items."""

    def test_builtin_tools_registered_in_init(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_register_builtin_tools called in __init__, not run()."""
        monkeypatch.setenv("NIBOT_AGENT__WORKSPACE", str(tmp_path))
        monkeypatch.setenv("NIBOT_PROVIDERS__OPENAI__API_KEY", "sk-test")
        from nibot.app import NiBot
        app = NiBot()
        # Tools should already be registered before run() is called
        assert app.registry.has("file_read")
        assert app.registry.has("delegate")


# ---- Phase 3: Production features ----

class TestProviderRetry:
    """3.1: Provider retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self) -> None:
        call_count = 0

        async def fake_acompletion(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary failure")
            # Return a mock response
            from types import SimpleNamespace
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="success", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=None,
            )

        provider = LiteLLMProvider(model="test-model", max_retries=3, retry_base_delay=0.01)
        import nibot.provider as prov_mod
        original = None
        try:
            # Monkey-patch litellm.acompletion inside the chat method
            import litellm
            original = litellm.acompletion
            litellm.acompletion = fake_acompletion
            result = await provider.chat(messages=[{"role": "user", "content": "hi"}])
            assert result.content == "success"
            assert call_count == 3
        finally:
            if original:
                litellm.acompletion = original

    @pytest.mark.asyncio
    async def test_retry_exhausted_returns_error(self) -> None:
        async def always_fail(**kwargs: Any) -> Any:
            raise ConnectionError("permanent failure")

        provider = LiteLLMProvider(model="test-model", max_retries=2, retry_base_delay=0.01)
        try:
            import litellm
            original = litellm.acompletion
            litellm.acompletion = always_fail
            result = await provider.chat(messages=[{"role": "user", "content": "hi"}])
            assert result.finish_reason == "error"
            assert "ConnectionError" in result.content
        finally:
            litellm.acompletion = original


class TestBusBackpressure:
    """3.2: MessageBus backpressure via maxsize."""

    def test_bus_default_maxsize_is_unlimited(self) -> None:
        bus = MessageBus()
        assert bus._inbound.maxsize == 0

    def test_bus_with_maxsize(self) -> None:
        bus = MessageBus(maxsize=5)
        assert bus._inbound.maxsize == 5
        assert bus._outbound.maxsize == 5


class TestStreamingPreview:
    """3.3: Streaming interface exists and falls back correctly."""

    @pytest.mark.asyncio
    async def test_base_provider_stream_fallback(self) -> None:
        from nibot.provider import LLMProvider

        class MinimalProvider(LLMProvider):
            async def chat(self, messages=None, tools=None, model="", max_tokens=4096, temperature=0.7) -> LLMResponse:
                return LLMResponse(content="hello from chat")

        provider = MinimalProvider()
        chunks = []
        async for chunk in provider.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        assert chunks == ["hello from chat"]


class TestCodexP1SessionWithTruncation:
    """Codex P1: Session saves correctly even when context builder truncates history."""

    @pytest.mark.asyncio
    async def test_new_messages_saved_despite_truncated_history(self, tmp_path: Path) -> None:
        """When token budget truncates old history, new tool messages must still be saved."""
        bus = MessageBus()
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="dummy", arguments={"x": "v"})],
            ),
            LLMResponse(content="done"),
        ]
        provider = FakeProvider(responses)
        reg = ToolRegistry()
        reg.register(DummyTool())
        sessions = SessionManager(tmp_path / "sessions")
        # Set a very small context window to trigger truncation
        cfg = NiBotConfig(agent={"context_window": 500, "context_reserve": 100})

        # Pre-populate session with some history that will be truncated
        session = sessions.get_or_create("ch:1")
        for i in range(20):
            session.add_message("user", f"old message {i} " * 10)
            session.add_message("assistant", f"old response {i} " * 10)
        sessions.save(session)

        memory = MemoryStore(tmp_path / "memory")
        skills = SkillsLoader([tmp_path / "skills"])
        cb = ContextBuilder(config=cfg, memory=memory, skills=skills, workspace=tmp_path)

        loop = AgentLoop(bus, provider, reg, sessions, cb, cfg)
        await loop._process(Envelope(channel="ch", chat_id="1", sender_id="u", content="new msg"))

        # Reload session and verify new messages were saved
        reloaded = sessions.get_or_create("ch:1")
        all_msgs = reloaded.get_history(max_messages=1000)
        roles = [m["role"] for m in all_msgs]
        # Must contain tool messages from this turn
        assert "tool" in roles
        # Must contain the user message from this turn
        user_msgs = [m for m in all_msgs if m["role"] == "user"]
        assert any("new msg" in m.get("content", "") for m in user_msgs)


class TestCodexP3RetryZero:
    """Codex P3: max_retries=0 should still make one API call."""

    @pytest.mark.asyncio
    async def test_zero_retries_makes_one_call(self) -> None:
        call_count = 0

        async def fake_acompletion(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            from types import SimpleNamespace
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=None,
            )

        provider = LiteLLMProvider(model="test", max_retries=0, retry_base_delay=0.01)
        import litellm
        original = litellm.acompletion
        try:
            litellm.acompletion = fake_acompletion
            result = await provider.chat(messages=[{"role": "user", "content": "hi"}])
            assert result.content == "ok"
            assert call_count == 1  # exactly one call, not zero
        finally:
            litellm.acompletion = original


class TestNewConfigFields:
    """Verify new config fields have correct defaults and are settable."""

    def test_new_agent_config_defaults(self) -> None:
        cfg = NiBotConfig()
        assert cfg.agent.context_window == 200000
        assert cfg.agent.context_reserve == 4096
        assert cfg.agent.llm_max_retries == 3
        assert cfg.agent.llm_retry_base_delay == 1.0
        assert cfg.agent.bus_queue_maxsize == 0

    def test_new_fields_settable(self) -> None:
        cfg = NiBotConfig(agent={
            "context_window": 32000,
            "llm_max_retries": 5,
            "bus_queue_maxsize": 100,
        })
        assert cfg.agent.context_window == 32000
        assert cfg.agent.llm_max_retries == 5
        assert cfg.agent.bus_queue_maxsize == 100
