"""Tool ABC and ToolRegistry."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from nibot.types import ToolContext, ToolResult

if TYPE_CHECKING:
    from nibot.event_log import EventLog


class Tool(ABC):
    """Base class for all tools. Implement this to add capabilities."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema format parameter definition."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool. Always returns a string (including errors)."""
        ...

    def receive_context(self, ctx: ToolContext) -> None:
        """Called before execute with request context. Override if needed."""
        pass

    def to_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Dynamic tool registration and execution."""

    def __init__(self, event_log: EventLog | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._event_log = event_log

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_definitions(
        self, deny: list[str] | None = None, allow: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get tool definitions. allow (whitelist) takes priority over deny (blacklist)."""
        if allow is not None:
            allow_set = set(allow)
            return [t.to_schema() for t in self._tools.values() if t.name in allow_set]
        deny_set = set(deny or [])
        return [t.to_schema() for t in self._tools.values() if t.name not in deny_set]

    async def execute(
        self, name: str, arguments: dict[str, Any], call_id: str = "", ctx: ToolContext | None = None,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(call_id=call_id, name=name, content=f"Unknown tool: {name}", is_error=True)
        t0 = time.monotonic()
        try:
            if ctx:
                tool.receive_context(ctx)
            result = await tool.execute(**arguments)
            duration_ms = (time.monotonic() - t0) * 1000
            if self._event_log:
                self._event_log.log_tool_call(tool=name, duration_ms=duration_ms, success=True)
            return ToolResult(call_id=call_id, name=name, content=result)
        except Exception as e:
            duration_ms = (time.monotonic() - t0) * 1000
            if self._event_log:
                self._event_log.log_tool_call(
                    tool=name, duration_ms=duration_ms, success=False, error=str(e)[:200],
                )
            return ToolResult(call_id=call_id, name=name, content=f"Error: {e}", is_error=True)

    def has(self, name: str) -> bool:
        return name in self._tools
