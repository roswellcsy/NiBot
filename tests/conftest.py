"""Shared test fixtures for NiBot test suite.

These fixtures are additive -- existing tests define their own helpers inline
and continue to work unchanged. New tests can import these for convenience.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from nibot.bus import MessageBus
from nibot.channel import BaseChannel
from nibot.config import NiBotConfig
from nibot.context import ContextBuilder
from nibot.memory import MemoryStore
from nibot.provider import LLMProvider
from nibot.registry import Tool, ToolRegistry
from nibot.session import Session, SessionManager
from nibot.types import Envelope, LLMResponse, SkillSpec, ToolCall


# ---- Fakes ----


class FakeProvider(LLMProvider):
    """LLM provider that returns pre-configured responses in order."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses: list[LLMResponse] = responses or []
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            return LLMResponse(content="(no more responses)")
        return self.responses.pop(0)


class EchoTool(Tool):
    """Simple tool that echoes its input."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo the input"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return f"echo: {kwargs.get('text', '')}"


class FailTool(Tool):
    """Tool that always raises an exception."""

    @property
    def name(self) -> str:
        return "fail_tool"

    @property
    def description(self) -> str:
        return "Always fails"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("intentional failure")


class FakeContextBuilder:
    """Minimal context builder for tests."""

    def build(self, session: Session, current: Envelope) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": "You are a test bot."}]
        for m in session.messages[-10:]:
            msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        msgs.append({"role": "user", "content": current.content})
        return msgs


class FakeChannel(BaseChannel):
    """Channel that captures sent messages."""

    name = "fake"

    def __init__(self) -> None:
        self.sent: list[Envelope] = []
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send(self, envelope: Envelope) -> None:
        self.sent.append(envelope)


class FakeSkills:
    """Minimal skills loader stub."""

    def __init__(self) -> None:
        self._always: list[SkillSpec] = []

    def get_always_skills(self) -> list[SkillSpec]:
        return self._always

    def build_summary(self) -> str:
        return ""


# ---- Fixtures ----


@pytest.fixture
def message_bus() -> MessageBus:
    return MessageBus()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def session_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(tmp_path / "sessions")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def fake_channel() -> FakeChannel:
    return FakeChannel()
