"""MCP (Model Context Protocol) bridge -- connect external MCP servers as NiBot tools."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from nibot.log import logger
from nibot.registry import Tool


class _MCPToolAdapter(Tool):
    """Adapts a single MCP tool to the NiBot Tool interface."""

    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        tool_schema: dict[str, Any],
        server: "MCPServerConnection",
    ) -> None:
        self._name = f"mcp_{tool_name}"
        self._original_name = tool_name
        self._description = tool_description
        self._schema = tool_schema
        self._server = server

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"[MCP] {self._description}"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._schema

    async def execute(self, **kwargs: Any) -> str:
        try:
            result = await self._server.call_tool(self._original_name, kwargs)
            return result
        except Exception as e:
            return f"MCP tool error: {e}"


class MCPServerConnection:
    """Manage a connection to an MCP server via stdio."""

    def __init__(self, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None) -> None:
        self.command = command
        self.args = args or []
        self.env = env
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Start MCP server process and initialize."""
        import os
        merged_env = {**os.environ, **(self.env or {})}
        self._process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        # Send initialize
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "nibot", "version": "1.0.0"},
        })

    async def disconnect(self) -> None:
        """Stop the MCP server."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()

    async def list_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions from the MCP server."""
        result = await self._send_request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        # MCP returns content as list of content blocks
        content_blocks = result.get("content", [])
        parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                parts.append(f"[image: {block.get('mimeType', 'unknown')}]")
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else json.dumps(result)

    async def _send_request(self, method: str, params: dict[str, Any]) -> Any:
        """Send a JSON-RPC request and wait for response."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP server not connected")

        self._request_id += 1
        req_id = self._request_id
        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        data = json.dumps(msg) + "\n"
        # Register waiter BEFORE sending to avoid race with fast responses
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise RuntimeError(f"MCP request '{method}' timed out")

    async def _read_loop(self) -> None:
        """Read JSON-RPC responses from stdout."""
        if not self._process or not self._process.stdout:
            return
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue
                req_id = msg.get("id")
                if req_id is not None and req_id in self._pending:
                    future = self._pending.pop(req_id)
                    if "error" in msg:
                        future.set_exception(RuntimeError(
                            f"MCP error: {msg['error'].get('message', 'unknown')}"
                        ))
                    else:
                        future.set_result(msg.get("result", {}))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"MCP reader error: {e}")


class MCPBridgeTool(Tool):
    """Meta-tool that connects to an MCP server and registers its tools.

    This is not directly registered as a tool -- instead, it creates
    _MCPToolAdapter instances for each tool the MCP server provides.
    """

    def __init__(self, server_name: str, command: str,
                 args: list[str] | None = None,
                 env: dict[str, str] | None = None) -> None:
        self._server_name = server_name
        self._connection = MCPServerConnection(command, args, env)
        self._adapters: list[_MCPToolAdapter] = []

    @property
    def name(self) -> str:
        return f"mcp_bridge_{self._server_name}"

    @property
    def description(self) -> str:
        return f"MCP bridge to {self._server_name}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return f"MCP bridge '{self._server_name}' has {len(self._adapters)} tools"

    async def connect_and_discover(self) -> list[_MCPToolAdapter]:
        """Connect to server, discover tools, return adapter list."""
        await self._connection.connect()
        raw_tools = await self._connection.list_tools()
        self._adapters = []
        for t in raw_tools:
            adapter = _MCPToolAdapter(
                tool_name=t.get("name", "unknown"),
                tool_description=t.get("description", ""),
                tool_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
                server=self._connection,
            )
            self._adapters.append(adapter)
        logger.info(f"MCP '{self._server_name}': discovered {len(self._adapters)} tools")
        return self._adapters

    async def disconnect(self) -> None:
        await self._connection.disconnect()
