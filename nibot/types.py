"""Core data structures -- the foundation of NiBot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Envelope:
    """Message envelope between Channel and Bus. Unified for inbound/outbound."""

    channel: str
    chat_id: str
    sender_id: str
    content: str
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolCall:
    """Tool invocation request from LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Tool execution result."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class LLMResponse:
    """LLM chat completion response."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class SkillSpec:
    """Parsed skill specification from SKILL.md."""

    name: str
    description: str
    body: str
    path: str
    always: bool = False
    requires_bins: list[str] = field(default_factory=list)
    requires_env: list[str] = field(default_factory=list)
