"""NiBot - Lightweight multi-channel AI agent framework."""

from nibot.types import Envelope, ToolCall, ToolResult, ToolContext, LLMResponse, SkillSpec
from nibot.registry import Tool, ToolRegistry
from nibot.channel import BaseChannel
from nibot.bus import MessageBus
from nibot.app import NiBot

__all__ = [
    "NiBot",
    "Tool",
    "ToolRegistry",
    "BaseChannel",
    "MessageBus",
    "Envelope",
    "ToolCall",
    "ToolResult",
    "ToolContext",
    "LLMResponse",
    "SkillSpec",
]
