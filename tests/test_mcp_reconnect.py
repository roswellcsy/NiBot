"""Tests for MCP bridge reconnection logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.tools.mcp_bridge import MCPServerConnection, _MCPToolAdapter


class TestMCPIsAlive:
    def test_alive_when_process_running(self):
        conn = MCPServerConnection("dummy")
        conn._process = MagicMock()
        conn._process.returncode = None  # still running
        conn._reader_task = MagicMock()
        conn._reader_task.done.return_value = False
        assert conn.is_alive() is True

    def test_dead_when_no_process(self):
        conn = MCPServerConnection("dummy")
        assert conn.is_alive() is False

    def test_dead_when_process_exited(self):
        conn = MCPServerConnection("dummy")
        conn._process = MagicMock()
        conn._process.returncode = 1
        assert conn.is_alive() is False

    def test_dead_when_reader_task_done(self):
        conn = MCPServerConnection("dummy")
        conn._process = MagicMock()
        conn._process.returncode = None
        conn._reader_task = MagicMock()
        conn._reader_task.done.return_value = True
        assert conn.is_alive() is False


class TestMCPReconnect:
    @pytest.mark.asyncio
    async def test_reconnect_calls_disconnect_then_connect(self):
        conn = MCPServerConnection("dummy")
        conn.disconnect = AsyncMock()
        conn.connect = AsyncMock()
        conn._pending = {1: asyncio.Future(), 2: asyncio.Future()}
        # Must appear dead so reconnect doesn't short-circuit
        conn._process = None

        await conn.reconnect()

        conn.disconnect.assert_awaited_once()
        conn.connect.assert_awaited_once()
        assert len(conn._pending) == 0

    @pytest.mark.asyncio
    async def test_reconnect_survives_disconnect_error(self):
        conn = MCPServerConnection("dummy")
        conn.disconnect = AsyncMock(side_effect=OSError("already dead"))
        conn.connect = AsyncMock()
        conn._process = None  # appear dead

        await conn.reconnect()

        conn.connect.assert_awaited_once()


class TestMCPAdapterRetry:
    @pytest.mark.asyncio
    async def test_adapter_reconnects_on_dead_server(self):
        conn = MCPServerConnection("dummy")
        conn.is_alive = MagicMock(return_value=False)
        conn.reconnect = AsyncMock()
        conn.call_tool = AsyncMock(return_value="result_data")

        adapter = _MCPToolAdapter("test_tool", "desc", {}, conn)
        result = await adapter.execute(arg1="val1")

        conn.reconnect.assert_awaited_once()
        conn.call_tool.assert_awaited_once_with("test_tool", {"arg1": "val1"})
        assert result == "result_data"

    @pytest.mark.asyncio
    async def test_adapter_retries_on_first_call_failure(self):
        conn = MCPServerConnection("dummy")
        conn.is_alive = MagicMock(return_value=True)
        conn.reconnect = AsyncMock()
        # First call fails, second (after reconnect) succeeds
        conn.call_tool = AsyncMock(side_effect=[RuntimeError("broken pipe"), "recovered"])

        adapter = _MCPToolAdapter("test_tool", "desc", {}, conn)
        result = await adapter.execute(x="1")

        conn.reconnect.assert_awaited_once()
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_adapter_returns_error_after_retry_fails(self):
        conn = MCPServerConnection("dummy")
        conn.is_alive = MagicMock(return_value=True)
        conn.reconnect = AsyncMock()
        conn.call_tool = AsyncMock(side_effect=RuntimeError("still broken"))

        adapter = _MCPToolAdapter("test_tool", "desc", {}, conn)
        result = await adapter.execute()

        assert "MCP tool error (after reconnect)" in result
        assert "still broken" in result

    @pytest.mark.asyncio
    async def test_adapter_succeeds_without_reconnect_when_alive(self):
        conn = MCPServerConnection("dummy")
        conn.is_alive = MagicMock(return_value=True)
        conn.reconnect = AsyncMock()
        conn.call_tool = AsyncMock(return_value="ok")

        adapter = _MCPToolAdapter("test_tool", "desc", {}, conn)
        result = await adapter.execute()

        conn.reconnect.assert_not_awaited()
        assert result == "ok"


class TestMCPConcurrentReconnect:
    """Phase 1 v1.4: verify reconnect lock prevents concurrent reconnect races."""

    @pytest.mark.asyncio
    async def test_concurrent_reconnect_only_connects_once(self):
        """5 concurrent reconnect() calls should only trigger 1 disconnect+connect."""
        conn = MCPServerConnection("dummy")
        conn._process = None  # appear dead

        connect_count = 0
        original_is_alive = conn.is_alive

        async def mock_connect():
            nonlocal connect_count
            connect_count += 1
            # After connect, make it appear alive
            conn._process = MagicMock()
            conn._process.returncode = None
            conn._reader_task = MagicMock()
            conn._reader_task.done.return_value = False

        conn.disconnect = AsyncMock()
        conn.connect = AsyncMock(side_effect=mock_connect)

        # Fire 5 concurrent reconnect calls
        await asyncio.gather(*[conn.reconnect() for _ in range(5)])

        # Only 1 should actually connect (others see is_alive()=True after first)
        assert connect_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls_during_reconnect(self):
        """Multiple adapters calling execute concurrently on dead server all get results."""
        conn = MCPServerConnection("dummy")
        conn._process = None  # dead

        async def mock_reconnect():
            async with conn._reconnect_lock:
                if conn.is_alive():
                    return
                await asyncio.sleep(0.01)  # simulate reconnect delay
                conn._process = MagicMock()
                conn._process.returncode = None
                conn._reader_task = MagicMock()
                conn._reader_task.done.return_value = False

        conn.reconnect = mock_reconnect
        conn.call_tool = AsyncMock(return_value="ok")

        adapters = [_MCPToolAdapter(f"tool{i}", "desc", {}, conn) for i in range(3)]
        results = await asyncio.gather(*[a.execute(x="1") for a in adapters])

        assert all(r == "ok" for r in results)

    @pytest.mark.asyncio
    async def test_reconnect_skips_when_already_alive(self):
        """reconnect() with live server is a no-op (double-check pattern)."""
        conn = MCPServerConnection("dummy")
        conn._process = MagicMock()
        conn._process.returncode = None
        conn._reader_task = MagicMock()
        conn._reader_task.done.return_value = False
        conn.disconnect = AsyncMock()
        conn.connect = AsyncMock()

        await conn.reconnect()

        conn.disconnect.assert_not_awaited()
        conn.connect.assert_not_awaited()
