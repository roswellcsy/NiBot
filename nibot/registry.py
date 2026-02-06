"""Tool ABC and ToolRegistry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nibot.types import ToolResult


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

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_definitions(self, deny: list[str] | None = None) -> list[dict[str, Any]]:
        """Get tool definitions in OpenAI format. Supports deny-list filtering."""
        deny_set = set(deny or [])
        return [t.to_schema() for t in self._tools.values() if t.name not in deny_set]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(call_id="", name=name, content=f"Unknown tool: {name}", is_error=True)
        try:
            result = await tool.execute(**arguments)
            return ToolResult(call_id="", name=name, content=result)
        except Exception as e:
            return ToolResult(call_id="", name=name, content=f"Error: {e}", is_error=True)

    def has(self, name: str) -> bool:
        return name in self._tools
