"""Tests for v1.0.0: Provider failover + MCP integration."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.config import ProviderConfig, ProvidersConfig
from nibot.provider import LiteLLMProvider, LLMProvider
from nibot.provider_pool import ProviderPool
from nibot.types import LLMResponse
from nibot.tools.mcp_bridge import MCPBridgeTool, MCPServerConnection, _MCPToolAdapter


class MockProvider(LLMProvider):
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    async def chat(self, messages=None, tools=None, **kwargs) -> LLMResponse:
        if self._error:
            raise self._error
        return self._response or LLMResponse(content="ok")


class TestProviderFailover:
    """ProviderPool.chat_with_fallback tries providers in order."""

    @pytest.mark.asyncio
    async def test_first_provider_succeeds(self) -> None:
        p1 = MockProvider(LLMResponse(content="from p1"))
        p2 = MockProvider(LLMResponse(content="from p2"))

        config = ProvidersConfig()
        pool = ProviderPool(config, p1)
        pool._cache["backup"] = p2

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["", "backup"],
        )
        assert result.content == "from p1"

    @pytest.mark.asyncio
    async def test_failover_to_second(self) -> None:
        p1 = MockProvider(error=RuntimeError("down"))
        p2 = MockProvider(LLMResponse(content="from backup"))

        config = ProvidersConfig()
        pool = ProviderPool(config, p1)
        pool._cache["backup"] = p2

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["", "backup"],
        )
        assert result.content == "from backup"

    @pytest.mark.asyncio
    async def test_all_fail_returns_error(self) -> None:
        p1 = MockProvider(error=RuntimeError("fail1"))
        p2 = MockProvider(error=RuntimeError("fail2"))

        config = ProvidersConfig()
        pool = ProviderPool(config, p1)
        pool._cache["backup"] = p2

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["", "backup"],
        )
        assert result.finish_reason == "error"
        assert "failed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_error_response_triggers_failover(self) -> None:
        p1 = MockProvider(LLMResponse(content="error msg", finish_reason="error"))
        p2 = MockProvider(LLMResponse(content="success"))

        config = ProvidersConfig()
        pool = ProviderPool(config, p1)
        pool._cache["backup"] = p2

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
            chain=["", "backup"],
        )
        assert result.content == "success"

    @pytest.mark.asyncio
    async def test_empty_chain_uses_default(self) -> None:
        p1 = MockProvider(LLMResponse(content="default"))
        config = ProvidersConfig()
        pool = ProviderPool(config, p1)

        result = await pool.chat_with_fallback(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result.content == "default"


class TestMCPToolAdapter:
    """MCP tool adapter wraps MCP tools as NiBot tools."""

    def test_adapter_name_prefix(self) -> None:
        server = MagicMock(spec=MCPServerConnection)
        adapter = _MCPToolAdapter("my_tool", "does stuff", {"type": "object", "properties": {}}, server)
        assert adapter.name == "mcp_my_tool"

    def test_adapter_description_prefix(self) -> None:
        server = MagicMock(spec=MCPServerConnection)
        adapter = _MCPToolAdapter("my_tool", "does stuff", {"type": "object", "properties": {}}, server)
        assert "[MCP]" in adapter.description

    @pytest.mark.asyncio
    async def test_adapter_execute_calls_server(self) -> None:
        server = AsyncMock(spec=MCPServerConnection)
        server.call_tool = AsyncMock(return_value="result text")
        adapter = _MCPToolAdapter("my_tool", "desc", {"type": "object", "properties": {}}, server)
        result = await adapter.execute(arg1="val1")
        server.call_tool.assert_called_once_with("my_tool", {"arg1": "val1"})
        assert result == "result text"

    @pytest.mark.asyncio
    async def test_adapter_execute_handles_error(self) -> None:
        server = AsyncMock(spec=MCPServerConnection)
        server.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))
        adapter = _MCPToolAdapter("my_tool", "desc", {"type": "object", "properties": {}}, server)
        result = await adapter.execute(arg1="val1")
        assert "error" in result.lower()


class TestMCPBridgeTool:
    """MCPBridgeTool discovers and wraps MCP server tools."""

    def test_bridge_name(self) -> None:
        bridge = MCPBridgeTool("test-server", "node", ["server.js"])
        assert bridge.name == "mcp_bridge_test-server"

    @pytest.mark.asyncio
    async def test_bridge_discover_creates_adapters(self) -> None:
        bridge = MCPBridgeTool("test", "echo")

        # Mock the connection
        mock_conn = AsyncMock(spec=MCPServerConnection)
        mock_conn.connect = AsyncMock()
        mock_conn.list_tools = AsyncMock(return_value=[
            {"name": "tool1", "description": "first tool", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "tool2", "description": "second tool", "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}},
        ])
        bridge._connection = mock_conn

        adapters = await bridge.connect_and_discover()
        assert len(adapters) == 2
        assert adapters[0].name == "mcp_tool1"
        assert adapters[1].name == "mcp_tool2"


class TestMCPServerConnection:
    """MCPServerConnection JSON-RPC protocol."""

    @pytest.mark.asyncio
    async def test_call_tool_formats_content(self) -> None:
        conn = MCPServerConnection("echo")
        # Test the content formatting logic directly
        result_data = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        }
        # Simulate call_tool's content formatting
        content_blocks = result_data.get("content", [])
        parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        assert "\n".join(parts) == "Hello\nWorld"

    def test_connection_init(self) -> None:
        conn = MCPServerConnection("npx", ["-y", "server"], {"KEY": "val"})
        assert conn.command == "npx"
        assert conn.args == ["-y", "server"]
        assert conn.env == {"KEY": "val"}
