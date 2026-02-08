"""v1.1 Task routing, provider failover, and quota management tests."""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.bus import MessageBus
from nibot.config import AgentTypeConfig, NiBotConfig, ProviderQuotaConfig
from nibot.provider_pool import ProviderPool, ProviderQuota
from nibot.types import Envelope, LLMResponse


# ---- Helpers ----

class FakeProvider:
    """Provider that returns a fixed response."""

    def __init__(self, content: str = "ok", finish_reason: str = "stop", usage: dict | None = None):
        self.content = content
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.call_count = 0

    async def chat(self, messages=None, tools=None, **kwargs) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=self.content, finish_reason=self.finish_reason, usage=self.usage)


class FailProvider:
    """Provider that raises an exception."""

    def __init__(self, error: Exception | None = None):
        self.error = error or RuntimeError("boom")

    async def chat(self, **kwargs) -> LLMResponse:
        raise self.error


class RateLimitProvider:
    """Provider that raises a 429 rate limit error."""

    def __init__(self, retry_after: int = 30):
        self.retry_after = retry_after

    async def chat(self, **kwargs) -> LLMResponse:
        raise RuntimeError(f"Error code: 429 - Rate limited. Retry after {self.retry_after} seconds")


class ErrorResponseProvider:
    """Provider that returns an error LLMResponse (not exception)."""

    def __init__(self, content: str = "LLM error: ValueError"):
        self.content = content

    async def chat(self, **kwargs) -> LLMResponse:
        return LLMResponse(content=self.content, finish_reason="error")


def _make_pool(providers: dict[str, Any], default: Any = None, quotas: dict[str, ProviderQuotaConfig] | None = None):
    """Create a ProviderPool with pre-cached providers (skip lazy creation)."""
    from nibot.config import ProvidersConfig
    pool = ProviderPool(ProvidersConfig(), default or FakeProvider("default"), quota_configs=quotas)
    pool._cache = providers
    return pool


# ---- ProviderQuota unit tests ----

class TestProviderQuota:
    """Three-layer quota tracking."""

    def test_unlimited_always_available(self) -> None:
        q = ProviderQuota("test")
        assert q.is_available() is True

    def test_rpm_limit_blocks_when_exhausted(self) -> None:
        q = ProviderQuota("test", rpm_limit=3)
        q.record_usage()
        q.record_usage()
        assert q.is_available() is True
        q.record_usage()
        assert q.is_available() is False

    def test_tpm_limit_blocks_when_exhausted(self) -> None:
        q = ProviderQuota("test", tpm_limit=1000)
        q.record_usage(tokens=500)
        assert q.is_available() is True
        q.record_usage(tokens=600)
        assert q.is_available() is False

    def test_rpm_recovers_after_window(self) -> None:
        q = ProviderQuota("test", rpm_limit=2)
        now = time.monotonic()
        # Simulate requests 61 seconds ago (expired)
        q._minute_requests.append(now - 61)
        q._minute_requests.append(now - 61)
        assert q.is_available() is True

    def test_rate_limit_exhaustion(self) -> None:
        q = ProviderQuota("test")
        q.record_rate_limit(retry_after=0.1)
        assert q.is_available() is False
        time.sleep(0.15)
        assert q.is_available() is True

    def test_header_calibration_blocks(self) -> None:
        q = ProviderQuota("test")
        assert q.is_available() is True
        q.update_from_headers(remaining_requests=0, remaining_tokens=None)
        assert q.is_available() is False

    def test_header_calibration_allows(self) -> None:
        q = ProviderQuota("test")
        q.update_from_headers(remaining_requests=100, remaining_tokens=50000)
        assert q.is_available() is True

    def test_header_tokens_blocks(self) -> None:
        q = ProviderQuota("test")
        q.update_from_headers(remaining_requests=None, remaining_tokens=0)
        assert q.is_available() is False

    def test_429_takes_precedence_over_headers(self) -> None:
        q = ProviderQuota("test")
        q.update_from_headers(remaining_requests=100, remaining_tokens=100000)
        q.record_rate_limit(retry_after=60.0)
        assert q.is_available() is False

    def test_record_usage_increments(self) -> None:
        q = ProviderQuota("test", rpm_limit=10)
        for _ in range(5):
            q.record_usage(tokens=100)
        assert len(q._minute_requests) == 5
        assert len(q._minute_tokens) == 5

    def test_header_exhaustion_expires_after_60s(self) -> None:
        """P1 fix: header remaining=0 must expire so provider can recover."""
        q = ProviderQuota("test")
        q.update_from_headers(remaining_requests=0, remaining_tokens=None)
        assert q.is_available() is False
        # Simulate 61 seconds passing by backdating the timestamp
        q._header_updated_at = time.monotonic() - 61
        assert q.is_available() is True
        # Stale headers should be cleared
        assert q._header_remaining_requests is None

    def test_zero_remaining_not_swallowed_by_or(self) -> None:
        """P1 fix: 0 is a valid remaining value, must not be treated as falsy."""
        q = ProviderQuota("test")
        q.update_from_headers(remaining_requests=0, remaining_tokens=5000)
        assert q.is_available() is False


# ---- ProviderPool.chat_with_fallback tests ----

class TestChatWithFallback:
    """Failover chain behavior."""

    @pytest.mark.asyncio
    async def test_first_provider_succeeds(self) -> None:
        p1 = FakeProvider("from-p1")
        p2 = FakeProvider("from-p2")
        pool = _make_pool({"p1": p1, "p2": p2})

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1", "p2"],
        )
        assert result.content == "from-p1"
        assert p1.call_count == 1
        assert p2.call_count == 0

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self) -> None:
        p1 = FailProvider()
        p2 = FakeProvider("from-p2")
        pool = _make_pool({"p1": p1, "p2": p2})

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1", "p2"],
        )
        assert result.content == "from-p2"

    @pytest.mark.asyncio
    async def test_fallback_on_error_response(self) -> None:
        p1 = ErrorResponseProvider()
        p2 = FakeProvider("from-p2")
        pool = _make_pool({"p1": p1, "p2": p2})

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1", "p2"],
        )
        assert result.content == "from-p2"

    @pytest.mark.asyncio
    async def test_all_fail_returns_error(self) -> None:
        p1 = FailProvider()
        p2 = FailProvider(RuntimeError("kaboom"))
        pool = _make_pool({"p1": p1, "p2": p2})

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1", "p2"],
        )
        assert result.finish_reason == "error"
        assert "All providers failed" in result.content

    @pytest.mark.asyncio
    async def test_empty_chain_uses_default(self) -> None:
        default = FakeProvider("from-default")
        pool = _make_pool({}, default=default)

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=[],
        )
        assert result.content == "from-default"

    @pytest.mark.asyncio
    async def test_kwargs_passed_through(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.chat = AsyncMock(return_value=LLMResponse(content="ok"))
        pool = _make_pool({"p1": mock_provider})

        await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1"],
            model="custom-model",
        )
        mock_provider.chat.assert_called_once()
        call_kwargs = mock_provider.chat.call_args
        assert call_kwargs.kwargs["model"] == "custom-model"


# ---- Quota-aware fallback tests ----

class TestQuotaAwareFallback:
    """Failover respects quota state."""

    @pytest.mark.asyncio
    async def test_skips_exhausted_provider(self) -> None:
        p1 = FakeProvider("from-p1")
        p2 = FakeProvider("from-p2")
        quotas = {
            "p1": ProviderQuotaConfig(rpm=1),
            "p2": ProviderQuotaConfig(),
        }
        pool = _make_pool({"p1": p1, "p2": p2}, quotas=quotas)
        # Exhaust p1's RPM
        pool._quotas["p1"].record_usage()

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1", "p2"],
        )
        assert result.content == "from-p2"
        assert p1.call_count == 0

    @pytest.mark.asyncio
    async def test_429_marks_provider_exhausted(self) -> None:
        p1 = RateLimitProvider(retry_after=30)
        p2 = FakeProvider("from-p2")
        quotas = {
            "p1": ProviderQuotaConfig(),
            "p2": ProviderQuotaConfig(),
        }
        pool = _make_pool({"p1": p1, "p2": p2}, quotas=quotas)

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1", "p2"],
        )
        assert result.content == "from-p2"
        # p1 should now be marked exhausted
        assert pool._quotas["p1"].is_available() is False

    @pytest.mark.asyncio
    async def test_success_records_usage(self) -> None:
        p1 = FakeProvider("ok", usage={"total_tokens": 500})
        quotas = {"p1": ProviderQuotaConfig(rpm=100)}
        pool = _make_pool({"p1": p1}, quotas=quotas)

        await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1"],
        )
        assert len(pool._quotas["p1"]._minute_requests) == 1
        assert pool._quotas["p1"]._minute_tokens[0][1] == 500

    @pytest.mark.asyncio
    async def test_header_calibration_on_success(self) -> None:
        p1 = FakeProvider("ok")
        quotas = {"p1": ProviderQuotaConfig()}
        pool = _make_pool({"p1": p1}, quotas=quotas)

        # Simulate response with ratelimit headers
        original_chat = p1.chat

        async def chat_with_headers(**kwargs):
            resp = await original_chat(**kwargs)
            resp.ratelimit_info = {"x-ratelimit-remaining-requests": 42, "x-ratelimit-remaining-tokens": 9999}
            return resp

        p1.chat = chat_with_headers

        await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1"],
        )
        assert pool._quotas["p1"]._header_remaining_requests == 42
        assert pool._quotas["p1"]._header_remaining_tokens == 9999

    @pytest.mark.asyncio
    async def test_zero_remaining_header_calibrates_correctly(self) -> None:
        """P1 fix: x-ratelimit-remaining-requests: 0 must block provider."""
        p1 = FakeProvider("p1")
        quotas = {"p1": ProviderQuotaConfig()}
        pool = _make_pool({"p1": p1}, quotas=quotas)

        original_chat = p1.chat

        async def chat_zero_remaining(**kwargs):
            resp = await original_chat(**kwargs)
            resp.ratelimit_info = {"x-ratelimit-remaining-requests": 0}
            return resp

        p1.chat = chat_zero_remaining

        # First call succeeds but calibrates remaining to 0
        await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1"],
        )
        assert pool._quotas["p1"]._header_remaining_requests == 0
        assert pool._quotas["p1"].is_available() is False

    @pytest.mark.asyncio
    async def test_all_exhausted_falls_to_default(self) -> None:
        p1 = FakeProvider("p1")
        p2 = FakeProvider("p2")
        default = FakeProvider("default")
        quotas = {
            "p1": ProviderQuotaConfig(rpm=1),
            "p2": ProviderQuotaConfig(rpm=1),
        }
        pool = _make_pool({"p1": p1, "p2": p2}, default=default, quotas=quotas)
        pool._quotas["p1"].record_usage()
        pool._quotas["p2"].record_usage()

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["p1", "p2"],
        )
        assert result.content == "default"


# ---- parse_retry_after tests ----

class TestParseRetryAfter:
    def test_extracts_retry_after(self) -> None:
        err = RuntimeError("Rate limited. Retry after 45 seconds")
        assert ProviderPool._parse_retry_after(err) == 45.0

    def test_default_60s(self) -> None:
        err = RuntimeError("Something failed")
        assert ProviderPool._parse_retry_after(err) == 60.0

    def test_retry_after_pattern_variations(self) -> None:
        assert ProviderPool._parse_retry_after(RuntimeError("retry-after 120")) == 120.0
        assert ProviderPool._parse_retry_after(RuntimeError("Retry After 90")) == 90.0


# ---- AgentLoop failover integration ----

class TestAgentLoopFailover:
    """AgentLoop uses chat_with_fallback for non-streaming path."""

    @pytest.mark.asyncio
    async def test_agent_uses_fallback_chain(self) -> None:
        from nibot.agent import AgentLoop

        bus = MessageBus()
        provider = FakeProvider("direct")

        mock_pool = AsyncMock()
        mock_pool.chat_with_fallback = AsyncMock(
            return_value=LLMResponse(content="via-fallback")
        )

        config = NiBotConfig()
        config.agent.provider_fallback_chain = ["anthropic", "openai"]
        config.agent.streaming = False

        import tempfile
        from pathlib import Path
        from nibot.context import ContextBuilder
        from nibot.memory import MemoryStore
        from nibot.session import SessionManager
        from nibot.skills import SkillsLoader
        from nibot.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as tmp:
            sessions = SessionManager(Path(tmp))
            memory = MemoryStore(Path(tmp) / "mem")
            skills = SkillsLoader([])
            ctx = ContextBuilder(config=config, memory=memory, skills=skills, workspace=Path(tmp))

            agent = AgentLoop(
                bus=bus, provider=provider, registry=ToolRegistry(),
                sessions=sessions, context_builder=ctx, config=config,
                provider_pool=mock_pool,
            )

            # Post a message and process it
            await bus.publish_inbound(Envelope(
                channel="test", chat_id="1", sender_id="user", content="hello",
            ))
            envelope = await bus.consume_inbound()
            result = await agent._process(envelope)

            assert result.content == "via-fallback"
            mock_pool.chat_with_fallback.assert_called()
            call_kwargs = mock_pool.chat_with_fallback.call_args
            assert call_kwargs.kwargs["chain"] == ["anthropic", "openai"]

    @pytest.mark.asyncio
    async def test_agent_no_fallback_uses_direct_provider(self) -> None:
        from nibot.agent import AgentLoop

        bus = MessageBus()
        provider = FakeProvider("direct")

        config = NiBotConfig()
        config.agent.provider_fallback_chain = []
        config.agent.streaming = False

        import tempfile
        from pathlib import Path
        from nibot.context import ContextBuilder
        from nibot.memory import MemoryStore
        from nibot.session import SessionManager
        from nibot.skills import SkillsLoader
        from nibot.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as tmp:
            sessions = SessionManager(Path(tmp))
            memory = MemoryStore(Path(tmp) / "mem")
            skills = SkillsLoader([])
            ctx = ContextBuilder(config=config, memory=memory, skills=skills, workspace=Path(tmp))

            agent = AgentLoop(
                bus=bus, provider=provider, registry=ToolRegistry(),
                sessions=sessions, context_builder=ctx, config=config,
            )

            await bus.publish_inbound(Envelope(
                channel="test", chat_id="1", sender_id="user", content="hello",
            ))
            envelope = await bus.consume_inbound()
            result = await agent._process(envelope)

            assert result.content == "direct"
            assert provider.call_count == 1


# ---- SubagentManager routing tests ----

class TestSubagentRouting:
    """SubagentManager routes to correct provider based on agent_config."""

    @pytest.mark.asyncio
    async def test_fallback_chain_used_when_configured(self) -> None:
        from nibot.subagent import SubagentManager

        bus = MessageBus()
        default_provider = FakeProvider("default")
        registry = MagicMock()
        registry.get_definitions.return_value = []
        registry.has.return_value = False

        mock_pool = AsyncMock()
        mock_pool.chat_with_fallback = AsyncMock(
            return_value=LLMResponse(content="via-chain")
        )

        mgr = SubagentManager(default_provider, registry, bus, provider_pool=mock_pool)

        agent_config = AgentTypeConfig(
            tools=["exec"],
            fallback_chain=["anthropic", "openai"],
        )
        task_id = await mgr.spawn(
            task="test", label="test",
            origin_channel="test", origin_chat_id="1",
            agent_type="coder", agent_config=agent_config,
            max_iterations=1,
        )
        await asyncio.sleep(0.2)

        mock_pool.chat_with_fallback.assert_called()
        call_kwargs = mock_pool.chat_with_fallback.call_args
        assert call_kwargs.kwargs["chain"] == ["anthropic", "openai"]

    @pytest.mark.asyncio
    async def test_named_provider_used_without_fallback_chain(self) -> None:
        from nibot.subagent import SubagentManager

        bus = MessageBus()
        default_provider = FakeProvider("default")
        named_provider = FakeProvider("from-named")
        registry = MagicMock()
        registry.get_definitions.return_value = []
        registry.has.return_value = False

        mock_pool = MagicMock()
        mock_pool.get = MagicMock(return_value=named_provider)

        mgr = SubagentManager(default_provider, registry, bus, provider_pool=mock_pool)

        agent_config = AgentTypeConfig(tools=["exec"], provider="kimi")
        await mgr.spawn(
            task="test", label="test",
            origin_channel="test", origin_chat_id="1",
            agent_type="writer", agent_config=agent_config,
            max_iterations=1,
        )
        await asyncio.sleep(0.2)

        mock_pool.get.assert_called_with("kimi")
        assert named_provider.call_count == 1
        assert default_provider.call_count == 0

    @pytest.mark.asyncio
    async def test_default_provider_used_without_config(self) -> None:
        from nibot.subagent import SubagentManager

        bus = MessageBus()
        default_provider = FakeProvider("default")
        registry = MagicMock()
        registry.get_definitions.return_value = []
        registry.has.return_value = False

        mgr = SubagentManager(default_provider, registry, bus)

        await mgr.spawn(
            task="test", label="test",
            origin_channel="test", origin_chat_id="1",
            max_iterations=1,
        )
        await asyncio.sleep(0.2)

        assert default_provider.call_count == 1


# ---- Config tests ----

class TestRoutingConfig:
    """Configuration schema for routing and quota."""

    def test_agent_type_config_fallback_chain(self) -> None:
        config = AgentTypeConfig(
            tools=["exec"],
            fallback_chain=["anthropic", "openai", "deepseek"],
        )
        assert config.fallback_chain == ["anthropic", "openai", "deepseek"]

    def test_agent_type_config_fallback_chain_default_empty(self) -> None:
        config = AgentTypeConfig()
        assert config.fallback_chain == []

    def test_provider_quota_config(self) -> None:
        from nibot.config import ProviderConfig
        pc = ProviderConfig(api_key="sk-test", quota=ProviderQuotaConfig(rpm=60, tpm=100000))
        assert pc.quota.rpm == 60
        assert pc.quota.tpm == 100000

    def test_provider_quota_config_defaults(self) -> None:
        from nibot.config import ProviderConfig
        pc = ProviderConfig()
        assert pc.quota.rpm == 0
        assert pc.quota.tpm == 0

    def test_ratelimit_info_in_llm_response(self) -> None:
        resp = LLMResponse(content="ok", ratelimit_info={"x-ratelimit-remaining-requests": 42})
        assert resp.ratelimit_info["x-ratelimit-remaining-requests"] == 42

    def test_ratelimit_info_default_empty(self) -> None:
        resp = LLMResponse(content="ok")
        assert resp.ratelimit_info == {}


# ---- Provider._parse ratelimit extraction ----

class TestProviderParseRatelimit:
    """LiteLLMProvider._parse extracts rate limit headers."""

    def test_parse_extracts_ratelimit_headers(self) -> None:
        from nibot.provider import LiteLLMProvider

        provider = LiteLLMProvider(model="test", max_retries=1)

        # Build a fake litellm response with _hidden_params
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message = MagicMock()
        mock_resp.choices[0].message.content = "hello"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = None
        mock_resp._hidden_params = {
            "additional_headers": {
                "x-ratelimit-remaining-requests": "42",
                "x-ratelimit-remaining-tokens": "9999",
            }
        }

        result = provider._parse(mock_resp)
        assert result.ratelimit_info["x-ratelimit-remaining-requests"] == 42
        assert result.ratelimit_info["x-ratelimit-remaining-tokens"] == 9999

    def test_parse_no_hidden_params(self) -> None:
        from nibot.provider import LiteLLMProvider

        provider = LiteLLMProvider(model="test", max_retries=1)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message = MagicMock()
        mock_resp.choices[0].message.content = "hello"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = None
        # No _hidden_params
        del mock_resp._hidden_params

        result = provider._parse(mock_resp)
        assert result.ratelimit_info == {}

    def test_parse_anthropic_headers(self) -> None:
        from nibot.provider import LiteLLMProvider

        provider = LiteLLMProvider(model="test", max_retries=1)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message = MagicMock()
        mock_resp.choices[0].message.content = "hello"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = None
        mock_resp._hidden_params = {
            "additional_headers": {
                "anthropic-ratelimit-requests-remaining": "10",
                "anthropic-ratelimit-tokens-remaining": "5000",
            }
        }

        result = provider._parse(mock_resp)
        assert result.ratelimit_info["anthropic-ratelimit-requests-remaining"] == 10
        assert result.ratelimit_info["anthropic-ratelimit-tokens-remaining"] == 5000
