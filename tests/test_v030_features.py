"""Tests for v0.3.0 features: bug fixes, security hardening."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nibot.bus import MessageBus
from nibot.provider import LiteLLMProvider
from nibot.registry import Tool, ToolRegistry
from nibot.session import SessionManager
from nibot.subagent import SubagentManager
from nibot.config import AgentTypeConfig
from nibot.tools.spawn_tool import DelegateTool
from nibot.tools.web_tools import _is_private_url
from nibot.types import Envelope, LLMResponse, ToolCall, ToolContext


# ---- Helpers ----

class MinimalProvider:
    async def chat(self, messages=None, tools=None, model="", max_tokens=4096, temperature=0.7):
        return LLMResponse(content="done", finish_reason="stop")


# ---- Phase 1: Bug fixes ----

class TestSubagentDelivery:
    """BUG-1: Subagent results must reach users via publish_outbound with original channel."""

    @pytest.mark.asyncio
    async def test_subagent_publishes_to_outbound_with_original_channel(self) -> None:
        bus = MessageBus()
        provider = MinimalProvider()
        registry = ToolRegistry()
        mgr = SubagentManager(provider, registry, bus)

        task_id = await mgr.spawn(
            task="test task", label="test",
            origin_channel="telegram", origin_chat_id="123",
            max_iterations=1,
        )
        # Wait for the background task to complete
        await asyncio.sleep(0.1)

        # Result should be on outbound queue, not inbound
        assert bus._outbound.qsize() > 0
        msg = await bus._outbound.get()
        assert msg.channel == "telegram"
        assert msg.chat_id == "123"
        assert "test" in msg.content

    @pytest.mark.asyncio
    async def test_subagent_error_still_delivers_to_correct_channel(self) -> None:
        class FailProvider:
            async def chat(self, **kwargs):
                raise RuntimeError("boom")

        bus = MessageBus()
        registry = ToolRegistry()
        mgr = SubagentManager(FailProvider(), registry, bus)

        await mgr.spawn(
            task="fail task", label="fail",
            origin_channel="feishu", origin_chat_id="456",
            max_iterations=1,
        )
        await asyncio.sleep(0.1)

        msg = await bus._outbound.get()
        assert msg.channel == "feishu"
        assert msg.chat_id == "456"
        assert "error" in msg.content.lower()


class TestDelegateToolStateless:
    """BUG-3: DelegateTool (was SpawnTool) must not store shared mutable state."""

    @pytest.mark.asyncio
    async def test_delegate_tool_gets_context_via_receive_context(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        mock_mgr.spawn = AsyncMock(return_value="abc123")
        agents = {"coder": AgentTypeConfig(tools=["exec"])}
        tool = DelegateTool(mock_mgr, agents)
        ctx = ToolContext(channel="tg", chat_id="99", session_key="tg:99")

        tool.receive_context(ctx)
        result = await tool.execute(
            agent_type="coder", task="do something", label="test",
        )
        assert "abc123" in result
        mock_mgr.spawn.assert_called_once_with(
            task="do something", label="test",
            origin_channel="tg", origin_chat_id="99",
            agent_type="coder", agent_config=agents["coder"],
        )

    @pytest.mark.asyncio
    async def test_delegate_tool_no_instance_state(self) -> None:
        """DelegateTool should not have _origin_channel/_origin_chat_id attributes."""
        mock_mgr = AsyncMock(spec=SubagentManager)
        tool = DelegateTool(mock_mgr, {})
        assert not hasattr(tool, "_origin_channel")
        assert not hasattr(tool, "_origin_chat_id")

    @pytest.mark.asyncio
    async def test_delegate_tool_without_context(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        mock_mgr.spawn = AsyncMock(return_value="def456")
        agents = {"system": AgentTypeConfig(tools=["exec"])}
        tool = DelegateTool(mock_mgr, agents)

        result = await tool.execute(agent_type="system", task="do something")
        assert "def456" in result
        mock_mgr.spawn.assert_called_once_with(
            task="do something", label="system",
            origin_channel="", origin_chat_id="",
            agent_type="system", agent_config=agents["system"],
        )


class TestConcurrentProcessing:
    """BUG-2/4: AgentLoop processes messages concurrently with per-session locking."""

    @pytest.mark.asyncio
    async def test_session_manager_lock_for(self) -> None:
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            sm = SessionManager(Path(tmp))
            lock1 = sm.lock_for("key1")
            lock2 = sm.lock_for("key1")
            lock3 = sm.lock_for("key2")
            assert lock1 is lock2  # same key, same lock
            assert lock1 is not lock3  # different key, different lock

    @pytest.mark.asyncio
    async def test_agent_run_uses_create_task(self) -> None:
        """AgentLoop.run() should dispatch via create_task (not block inline)."""
        from nibot.agent import AgentLoop
        import inspect

        # Verify the source code uses asyncio.create_task in run()
        source = inspect.getsource(AgentLoop.run)
        assert "create_task" in source

    @pytest.mark.asyncio
    async def test_session_lock_serializes_same_session(self) -> None:
        """Same session key should return same lock object."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            sm = SessionManager(Path(tmp))
            lock = sm.lock_for("a:1")
            # Acquire the lock
            async with lock:
                # Trying to acquire again would block -- same lock returned
                assert sm.lock_for("a:1") is lock


class TestTelegramChunking:
    """BUG-5: Messages > 4096 chars should be sent in chunks."""

    @pytest.mark.asyncio
    async def test_long_message_split_into_chunks(self) -> None:
        from nibot.channels.telegram import TelegramChannel, _TG_MAX_LENGTH
        from nibot.config import TelegramChannelConfig

        config = TelegramChannelConfig(enabled=True, token="fake")
        bus = MessageBus()
        ch = TelegramChannel(config, bus)

        # Mock the bot
        sent_texts: list[str] = []
        mock_bot = AsyncMock()
        async def capture_send(chat_id, text):
            sent_texts.append(text)
        mock_bot.send_message = capture_send
        ch._app = MagicMock()
        ch._app.bot = mock_bot

        long_text = "A" * 5000
        await ch.send(Envelope(channel="telegram", chat_id="123", sender_id="bot", content=long_text))

        assert len(sent_texts) == 2
        assert len(sent_texts[0]) == _TG_MAX_LENGTH
        assert len(sent_texts[1]) == 5000 - _TG_MAX_LENGTH

    @pytest.mark.asyncio
    async def test_short_message_not_split(self) -> None:
        from nibot.channels.telegram import TelegramChannel
        from nibot.config import TelegramChannelConfig

        config = TelegramChannelConfig(enabled=True, token="fake")
        bus = MessageBus()
        ch = TelegramChannel(config, bus)

        sent_texts: list[str] = []
        mock_bot = AsyncMock()
        async def capture_send(chat_id, text):
            sent_texts.append(text)
        mock_bot.send_message = capture_send
        ch._app = MagicMock()
        ch._app.bot = mock_bot

        await ch.send(Envelope(channel="telegram", chat_id="123", sender_id="bot", content="short"))
        assert len(sent_texts) == 1
        assert sent_texts[0] == "short"


# ---- Phase 2: Security ----

class TestSSRFProtection:
    """SEC-1: WebFetchTool blocks private/reserved IPs."""

    def test_localhost_blocked(self) -> None:
        assert _is_private_url("http://127.0.0.1/secret") is True

    def test_private_ip_blocked(self) -> None:
        assert _is_private_url("http://10.0.0.1/admin") is True
        assert _is_private_url("http://192.168.1.1/") is True

    def test_link_local_blocked(self) -> None:
        assert _is_private_url("http://169.254.169.254/metadata") is True

    def test_empty_hostname_blocked(self) -> None:
        assert _is_private_url("not-a-url") is True

    def test_public_url_allowed(self) -> None:
        # Mock DNS to return a known public IP, avoiding network dependency
        from unittest.mock import patch
        fake_result = [(2, 1, 6, "", ("142.250.80.46", 443))]
        with patch("socket.getaddrinfo", return_value=fake_result):
            assert _is_private_url("https://www.google.com") is False


class TestErrorSanitization:
    """SEC-2: Provider errors only expose exception type, not details."""

    @pytest.mark.asyncio
    async def test_error_response_only_shows_type(self) -> None:
        async def fail_with_secret(**kwargs):
            raise ValueError("api_key=sk-secret-12345 is invalid")

        provider = LiteLLMProvider(model="test", max_retries=1, retry_base_delay=0.01)
        try:
            import litellm
            original = litellm.acompletion
            litellm.acompletion = fail_with_secret
            result = await provider.chat(messages=[{"role": "user", "content": "hi"}])
            assert result.finish_reason == "error"
            assert "ValueError" in result.content
            assert "sk-secret" not in result.content
            assert "api_key" not in result.content
        finally:
            litellm.acompletion = original
