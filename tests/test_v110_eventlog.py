"""Tests for structured event log and instrumentation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.event_log import EventLog
from nibot.types import LLMResponse, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_events(path: Path) -> list[dict[str, Any]]:
    """Read all JSONL events from path."""
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# EventLog core
# ---------------------------------------------------------------------------

class TestEventLogCore:
    def test_log_llm_call(self, tmp_path: Path) -> None:
        el = EventLog(tmp_path / "events.jsonl")
        el.log_llm_call(
            provider="anthropic", model="claude-sonnet-4-5",
            input_tokens=100, output_tokens=50,
            latency_ms=1234.5, success=True,
        )
        events = _read_events(tmp_path / "events.jsonl")
        assert len(events) == 1
        e = events[0]
        assert e["type"] == "llm_call"
        assert e["provider"] == "anthropic"
        assert e["model"] == "claude-sonnet-4-5"
        assert e["input_tokens"] == 100
        assert e["output_tokens"] == 50
        assert e["latency_ms"] == 1234.5
        assert e["success"] is True
        assert "ts" in e
        assert "error" not in e

    def test_log_llm_call_with_error(self, tmp_path: Path) -> None:
        el = EventLog(tmp_path / "events.jsonl")
        el.log_llm_call(
            provider="openai", model="gpt-4",
            input_tokens=0, output_tokens=0,
            latency_ms=500, success=False, error="429 rate limit",
        )
        events = _read_events(tmp_path / "events.jsonl")
        assert events[0]["success"] is False
        assert events[0]["error"] == "429 rate limit"

    def test_log_tool_call(self, tmp_path: Path) -> None:
        el = EventLog(tmp_path / "events.jsonl")
        el.log_tool_call(tool="web_search", duration_ms=890.3, success=True)
        events = _read_events(tmp_path / "events.jsonl")
        assert len(events) == 1
        e = events[0]
        assert e["type"] == "tool_call"
        assert e["tool"] == "web_search"
        assert e["duration_ms"] == 890.3
        assert e["success"] is True

    def test_log_tool_call_with_error(self, tmp_path: Path) -> None:
        el = EventLog(tmp_path / "events.jsonl")
        el.log_tool_call(tool="exec", duration_ms=100, success=False, error="timeout")
        events = _read_events(tmp_path / "events.jsonl")
        assert events[0]["error"] == "timeout"

    def test_log_provider_switch(self, tmp_path: Path) -> None:
        el = EventLog(tmp_path / "events.jsonl")
        el.log_provider_switch(
            chain=["anthropic", "openai"],
            selected="openai",
            skipped=["anthropic"],
            reason="quota_exhausted",
        )
        events = _read_events(tmp_path / "events.jsonl")
        assert len(events) == 1
        e = events[0]
        assert e["type"] == "provider_switch"
        assert e["chain"] == ["anthropic", "openai"]
        assert e["selected"] == "openai"
        assert e["skipped"] == ["anthropic"]
        assert e["reason"] == "quota_exhausted"

    def test_log_request(self, tmp_path: Path) -> None:
        el = EventLog(tmp_path / "events.jsonl")
        el.log_request(
            channel="telegram", session_key="tg:123",
            latency_ms=3500, tool_count=2,
            total_tokens=1550, provider="openai",
        )
        events = _read_events(tmp_path / "events.jsonl")
        assert len(events) == 1
        e = events[0]
        assert e["type"] == "request"
        assert e["channel"] == "telegram"
        assert e["total_tokens"] == 1550

    def test_disabled_writes_nothing(self, tmp_path: Path) -> None:
        el = EventLog(tmp_path / "events.jsonl", enabled=False)
        el.log_llm_call("a", "m", 0, 0, 0.0, True)
        el.log_tool_call("t", 0.0, True)
        el.log_provider_switch([], "", [], "")
        el.log_request("c", "s", 0.0, 0, 0, "p")
        assert not (tmp_path / "events.jsonl").exists()

    def test_multiple_events_append(self, tmp_path: Path) -> None:
        el = EventLog(tmp_path / "events.jsonl")
        el.log_llm_call("a", "m", 10, 5, 100, True)
        el.log_tool_call("t", 50, True)
        el.log_request("c", "s", 200, 1, 15, "a")
        events = _read_events(tmp_path / "events.jsonl")
        assert len(events) == 3
        assert [e["type"] for e in events] == ["llm_call", "tool_call", "request"]

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        el = EventLog(tmp_path / "sub" / "dir" / "events.jsonl")
        el.log_llm_call("a", "m", 0, 0, 0, True)
        assert (tmp_path / "sub" / "dir" / "events.jsonl").exists()

    def test_latency_rounding(self, tmp_path: Path) -> None:
        el = EventLog(tmp_path / "events.jsonl")
        el.log_llm_call("a", "m", 0, 0, 1234.5678, True)
        el.log_tool_call("t", 99.999, True)
        events = _read_events(tmp_path / "events.jsonl")
        assert events[0]["latency_ms"] == 1234.6
        assert events[1]["duration_ms"] == 100.0


# ---------------------------------------------------------------------------
# ToolRegistry instrumentation
# ---------------------------------------------------------------------------

class TestRegistryInstrumentation:
    @pytest.mark.asyncio
    async def test_tool_success_logged(self, tmp_path: Path) -> None:
        from nibot.registry import ToolRegistry

        el = EventLog(tmp_path / "events.jsonl")
        reg = ToolRegistry(event_log=el)

        tool = MagicMock()
        tool.name = "test_tool"
        tool.execute = AsyncMock(return_value="ok")
        reg.register(tool)

        result = await reg.execute("test_tool", {})
        assert not result.is_error

        events = _read_events(tmp_path / "events.jsonl")
        assert len(events) == 1
        assert events[0]["type"] == "tool_call"
        assert events[0]["tool"] == "test_tool"
        assert events[0]["success"] is True

    @pytest.mark.asyncio
    async def test_tool_error_logged(self, tmp_path: Path) -> None:
        from nibot.registry import ToolRegistry

        el = EventLog(tmp_path / "events.jsonl")
        reg = ToolRegistry(event_log=el)

        tool = MagicMock()
        tool.name = "fail_tool"
        tool.execute = AsyncMock(side_effect=RuntimeError("boom"))
        reg.register(tool)

        result = await reg.execute("fail_tool", {})
        assert result.is_error

        events = _read_events(tmp_path / "events.jsonl")
        assert len(events) == 1
        assert events[0]["success"] is False
        assert "boom" in events[0]["error"]

    @pytest.mark.asyncio
    async def test_no_event_log_still_works(self) -> None:
        from nibot.registry import ToolRegistry

        reg = ToolRegistry()  # no event_log
        tool = MagicMock()
        tool.name = "t"
        tool.execute = AsyncMock(return_value="ok")
        reg.register(tool)
        result = await reg.execute("t", {})
        assert result.content == "ok"


# ---------------------------------------------------------------------------
# ProviderPool instrumentation
# ---------------------------------------------------------------------------

class _FakeProvider:
    """Minimal LLMProvider for testing."""
    def __init__(self, model: str = "test-model", response: LLMResponse | None = None) -> None:
        self.model = model
        self._response = response or LLMResponse(
            content="hi", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    async def chat(self, **kwargs: Any) -> LLMResponse:
        return self._response


class TestProviderPoolInstrumentation:
    def _make_pool(
        self, tmp_path: Path, providers: dict[str, _FakeProvider] | None = None,
    ) -> tuple:
        from nibot.config import ProviderConfig, ProviderQuotaConfig, ProvidersConfig
        from nibot.provider_pool import ProviderPool

        el = EventLog(tmp_path / "events.jsonl")
        default = _FakeProvider()
        pcfg = ProvidersConfig()
        pool = ProviderPool(pcfg, default, event_log=el)
        # Inject fake providers into cache
        for name, prov in (providers or {}).items():
            pool._cache[name] = prov
        return pool, el

    @pytest.mark.asyncio
    async def test_success_logs_llm_call(self, tmp_path: Path) -> None:
        pool, el = self._make_pool(tmp_path, {"anthropic": _FakeProvider("claude-sonnet")})
        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["anthropic"],
        )
        assert result.content == "hi"

        events = _read_events(tmp_path / "events.jsonl")
        llm_events = [e for e in events if e["type"] == "llm_call"]
        assert len(llm_events) == 1
        assert llm_events[0]["provider"] == "anthropic"
        assert llm_events[0]["model"] == "claude-sonnet"
        assert llm_events[0]["success"] is True
        assert llm_events[0]["input_tokens"] == 10
        assert llm_events[0]["output_tokens"] == 5

    @pytest.mark.asyncio
    async def test_error_logs_llm_call(self, tmp_path: Path) -> None:
        fail = _FakeProvider()
        fail.chat = AsyncMock(side_effect=RuntimeError("connection refused"))
        pool, el = self._make_pool(tmp_path, {"bad": fail})
        # Also need default to respond
        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["bad"],
        )
        # Falls through to default
        events = _read_events(tmp_path / "events.jsonl")
        llm_events = [e for e in events if e["type"] == "llm_call"]
        assert any(e["success"] is False for e in llm_events)

    @pytest.mark.asyncio
    async def test_provider_switch_logged(self, tmp_path: Path) -> None:
        from nibot.provider_pool import ProviderQuota

        pool, el = self._make_pool(tmp_path, {
            "anthropic": _FakeProvider("claude"),
            "openai": _FakeProvider("gpt-4"),
        })
        # Exhaust anthropic quota
        pool._quotas["anthropic"] = ProviderQuota("anthropic", rpm_limit=60)
        pool._quotas["anthropic"].record_rate_limit(300)

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["anthropic", "openai"],
        )
        assert result.content == "hi"

        events = _read_events(tmp_path / "events.jsonl")
        switch_events = [e for e in events if e["type"] == "provider_switch"]
        assert len(switch_events) == 1
        assert switch_events[0]["skipped"] == ["anthropic"]
        assert switch_events[0]["selected"] == "openai"


# ---------------------------------------------------------------------------
# AgentLoop instrumentation
# ---------------------------------------------------------------------------

class TestAgentLoopInstrumentation:
    @pytest.mark.asyncio
    async def test_request_event_logged(self, tmp_path: Path) -> None:
        from nibot.agent import AgentLoop
        from nibot.bus import MessageBus
        from nibot.config import NiBotConfig
        from nibot.context import ContextBuilder
        from nibot.provider import LLMProvider
        from nibot.session import SessionManager
        from nibot.types import Envelope

        el = EventLog(tmp_path / "events.jsonl")
        bus = MessageBus()
        config = NiBotConfig()
        config.agent.streaming = False
        sessions = SessionManager(tmp_path / "sessions")

        provider = MagicMock(spec=LLMProvider)
        provider.chat = AsyncMock(return_value=LLMResponse(content="response"))

        ctx_builder = MagicMock()
        ctx_builder.build.return_value = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

        from nibot.registry import ToolRegistry
        registry = ToolRegistry()

        agent = AgentLoop(
            bus=bus, provider=provider, registry=registry,
            sessions=sessions, context_builder=ctx_builder,
            config=config, event_log=el,
        )

        envelope = Envelope(channel="test", chat_id="c1", sender_id="u1", content="hello")
        result = await agent._process(envelope)
        assert result.content == "response"

        events = _read_events(tmp_path / "events.jsonl")
        req_events = [e for e in events if e["type"] == "request"]
        assert len(req_events) == 1
        assert req_events[0]["channel"] == "test"
        assert req_events[0]["session_key"] == "test:c1"
        assert req_events[0]["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_no_event_log_still_works(self, tmp_path: Path) -> None:
        from nibot.agent import AgentLoop
        from nibot.bus import MessageBus
        from nibot.config import NiBotConfig
        from nibot.context import ContextBuilder
        from nibot.provider import LLMProvider
        from nibot.session import SessionManager
        from nibot.types import Envelope

        bus = MessageBus()
        config = NiBotConfig()
        config.agent.streaming = False
        sessions = SessionManager(tmp_path / "sessions")

        provider = MagicMock(spec=LLMProvider)
        provider.chat = AsyncMock(return_value=LLMResponse(content="ok"))

        ctx_builder = MagicMock()
        ctx_builder.build.return_value = [
            {"role": "system", "content": "sys"},
        ]

        from nibot.registry import ToolRegistry
        registry = ToolRegistry()

        agent = AgentLoop(
            bus=bus, provider=provider, registry=registry,
            sessions=sessions, context_builder=ctx_builder,
            config=config,
        )

        envelope = Envelope(channel="t", chat_id="c", sender_id="u", content="x")
        result = await agent._process(envelope)
        assert result.content == "ok"


# ---------------------------------------------------------------------------
# Health endpoint with provider status
# ---------------------------------------------------------------------------

class TestHealthProviderStatus:
    def test_health_includes_provider_quota(self) -> None:
        from nibot.config import ProviderQuotaConfig, ProvidersConfig
        from nibot.health import _build_health
        from nibot.provider_pool import ProviderPool, ProviderQuota

        app = MagicMock()
        app.config.agent.model = "test"
        app.sessions._cache = {}
        app._channels = []
        app.agent._running = True
        app.agent._tasks = set()
        app.scheduler._jobs = {}

        # Real ProviderPool with quota
        default = MagicMock()
        pcfg = ProvidersConfig()
        pool = ProviderPool(pcfg, default, quota_configs={
            "anthropic": ProviderQuotaConfig(rpm=60),
        })
        app.provider_pool = pool

        result = _build_health(app)
        assert "providers" in result
        assert "anthropic" in result["providers"]
        assert result["providers"]["anthropic"]["available"] is True
        assert result["providers"]["anthropic"]["rpm_limit"] == 60

    def test_health_without_provider_pool(self) -> None:
        """MagicMock provider_pool should not crash health."""
        from nibot.health import _build_health

        app = MagicMock()
        app.config.agent.model = "test"
        app.sessions._cache = {}
        app._channels = []
        app.agent._running = True
        app.agent._tasks = set()
        app.scheduler._jobs = {}

        result = _build_health(app)
        assert result["status"] == "ok"
        # Should not have providers section (MagicMock is not ProviderPool)
        assert "providers" not in result


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

class TestEventLogConfig:
    def test_default_config(self) -> None:
        from nibot.config import NiBotConfig
        config = NiBotConfig()
        assert config.event_log.enabled is True
        assert config.event_log.file == ""
        assert config.event_log.rotation == "50 MB"
        assert config.event_log.retention == "90 days"

    def test_disabled_config(self) -> None:
        from nibot.config import NiBotConfig
        config = NiBotConfig(event_log={"enabled": False})
        assert config.event_log.enabled is False


class TestEventLogNoThreadingLock:
    """Phase 3 v1.4: EventLog must not use threading.Lock in asyncio context."""

    def test_no_threading_import(self) -> None:
        import inspect
        from nibot import event_log as mod
        source = inspect.getsource(mod)
        assert "threading" not in source

    def test_no_lock_attribute(self, tmp_path: Path) -> None:
        from nibot.event_log import EventLog
        elog = EventLog(tmp_path / "events.jsonl")
        assert not hasattr(elog, "_lock")

    def test_concurrent_append_no_data_loss(self, tmp_path: Path) -> None:
        """100 rapid appends produce exactly 100 lines."""
        from nibot.event_log import EventLog
        elog = EventLog(tmp_path / "events.jsonl")
        for i in range(100):
            elog.log_tool_call(f"tool_{i}", 1.0, True)
        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == 100
