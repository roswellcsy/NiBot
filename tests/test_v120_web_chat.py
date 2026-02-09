"""v1.2 Web chat: SSE streaming, chat send/stream/sessions/history routes."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nibot.web.routes import handle_route
from nibot.web.server import SSEResponse


# ---- Stub objects (reuse pattern from test_v120_web_panel) ----


class _StubBus:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish_inbound(self, envelope: Any) -> None:
        self.published.append(envelope)


class _StubSessions:
    def __init__(self, sessions: list[dict] | None = None,
                 messages: list[dict] | None = None) -> None:
        self._cache: dict[str, Any] = {}
        self._sessions = sessions or []
        self._messages = messages or []

    def query_recent(self, limit: int = 50) -> list[dict]:
        return self._sessions[:limit]

    def get_session_messages(self, key: str, limit: int = 50) -> list[dict]:
        return self._messages[:limit]


class _StubAgent:
    _running = True
    _tasks: list[Any] = []


class _StubAgentConfig:
    model: str = "test-model"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_iterations: int = 25


class _StubToolsConfig:
    sandbox_enabled: bool = True
    sandbox_memory_mb: int = 512
    exec_timeout: int = 60


class _StubConfig:
    def __init__(self) -> None:
        self.agent = _StubAgentConfig()
        self.tools = _StubToolsConfig()
        self.agents: dict[str, Any] = {}


class _StubSubagents:
    def list_tasks(self, limit: int = 20) -> list[Any]:
        return []


class _StubSkills:
    def get_all(self) -> list[Any]:
        return []


class _StubChannel:
    name: str = "test"


class _StubApp:
    """Minimal app stub with bus and _web_streams for chat route testing."""

    def __init__(self) -> None:
        self.agent = _StubAgent()
        self.config = _StubConfig()
        self._channels = [_StubChannel()]
        self.sessions = _StubSessions()
        self.skills = _StubSkills()
        self.subagents = _StubSubagents()
        self.bus = _StubBus()
        self._web_streams: dict[str, asyncio.Queue[Any]] = {}


def _app(**overrides: Any) -> _StubApp:
    app = _StubApp()
    for k, v in overrides.items():
        setattr(app, k, v)
    return app


# ---- /api/chat/send ----


class TestChatSend:

    @pytest.mark.asyncio
    async def test_send_returns_stream_id_and_chat_id(self, tmp_path: Path) -> None:
        app = _app()
        body = json.dumps({"content": "hello"}).encode()
        result = await handle_route(app, "POST", "/api/chat/send", body, tmp_path)
        assert "stream_id" in result
        assert "chat_id" in result
        assert len(result["stream_id"]) == 12
        assert result["chat_id"].startswith("web_")

    @pytest.mark.asyncio
    async def test_send_with_existing_chat_id(self, tmp_path: Path) -> None:
        app = _app()
        body = json.dumps({"content": "hi", "chat_id": "my-session"}).encode()
        result = await handle_route(app, "POST", "/api/chat/send", body, tmp_path)
        assert result["chat_id"] == "my-session"

    @pytest.mark.asyncio
    async def test_send_empty_content_rejected(self, tmp_path: Path) -> None:
        app = _app()
        body = json.dumps({"content": ""}).encode()
        result = await handle_route(app, "POST", "/api/chat/send", body, tmp_path)
        assert "error" in result
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_send_whitespace_only_rejected(self, tmp_path: Path) -> None:
        app = _app()
        body = json.dumps({"content": "   "}).encode()
        result = await handle_route(app, "POST", "/api/chat/send", body, tmp_path)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_send_registers_stream_queue(self, tmp_path: Path) -> None:
        app = _app()
        body = json.dumps({"content": "test"}).encode()
        result = await handle_route(app, "POST", "/api/chat/send", body, tmp_path)
        stream_id = result["stream_id"]
        assert stream_id in app._web_streams
        assert isinstance(app._web_streams[stream_id], asyncio.Queue)

    @pytest.mark.asyncio
    async def test_send_publishes_envelope(self, tmp_path: Path) -> None:
        app = _app()
        body = json.dumps({"content": "hello bot"}).encode()
        result = await handle_route(app, "POST", "/api/chat/send", body, tmp_path)
        assert len(app.bus.published) == 1
        env = app.bus.published[0]
        assert env.channel == "web"
        assert env.content == "hello bot"
        assert env.sender_id == "web_user"
        assert env.metadata["stream_id"] == result["stream_id"]

    @pytest.mark.asyncio
    async def test_send_no_body(self, tmp_path: Path) -> None:
        app = _app()
        result = await handle_route(app, "POST", "/api/chat/send", b"", tmp_path)
        assert "error" in result


# ---- /api/chat/stream ----


class TestChatStream:

    @pytest.mark.asyncio
    async def test_stream_missing_id(self, tmp_path: Path) -> None:
        app = _app()
        result = await handle_route(app, "GET", "/api/chat/stream", b"", tmp_path)
        assert "error" in result
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_stream_not_found(self, tmp_path: Path) -> None:
        app = _app()
        result = await handle_route(
            app, "GET", "/api/chat/stream?id=nonexistent", b"", tmp_path,
        )
        assert "error" in result
        assert result.get("status") == 404

    @pytest.mark.asyncio
    async def test_stream_returns_sse_response(self, tmp_path: Path) -> None:
        app = _app()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        app._web_streams["abc123"] = queue
        result = await handle_route(
            app, "GET", "/api/chat/stream?id=abc123", b"", tmp_path,
        )
        assert isinstance(result, SSEResponse)
        assert callable(result.handler)


# ---- /api/chat/sessions ----


class TestChatSessions:

    @pytest.mark.asyncio
    async def test_sessions_filters_web_prefix(self, tmp_path: Path) -> None:
        sessions = [
            {"key": "web:chat1", "messages": 5},
            {"key": "discord:abc", "messages": 3},
            {"key": "web:chat2", "messages": 8},
            {"key": "api:xyz", "messages": 1},
        ]
        app = _app(sessions=_StubSessions(sessions=sessions))
        result = await handle_route(app, "GET", "/api/chat/sessions", b"", tmp_path)
        assert len(result["sessions"]) == 2
        assert all(s["key"].startswith("web:") for s in result["sessions"])

    @pytest.mark.asyncio
    async def test_sessions_empty(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/api/chat/sessions", b"", tmp_path)
        assert result["sessions"] == []

    @pytest.mark.asyncio
    async def test_sessions_no_web_sessions(self, tmp_path: Path) -> None:
        sessions = [{"key": "discord:abc", "messages": 3}]
        app = _app(sessions=_StubSessions(sessions=sessions))
        result = await handle_route(app, "GET", "/api/chat/sessions", b"", tmp_path)
        assert result["sessions"] == []


# ---- /api/chat/history ----


class TestChatHistory:

    @pytest.mark.asyncio
    async def test_history_missing_chat_id(self, tmp_path: Path) -> None:
        result = await handle_route(
            _app(), "GET", "/api/chat/history", b"", tmp_path,
        )
        assert "error" in result
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_history_returns_messages(self, tmp_path: Path) -> None:
        msgs = [
            {"role": "user", "content": "hello", "timestamp": "2026-01-01T00:00:00"},
            {"role": "assistant", "content": "hi there", "timestamp": "2026-01-01T00:00:01"},
        ]
        app = _app(sessions=_StubSessions(messages=msgs))
        result = await handle_route(
            app, "GET", "/api/chat/history?chat_id=web_abc", b"", tmp_path,
        )
        assert result["chat_id"] == "web_abc"
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["content"] == "hi there"

    @pytest.mark.asyncio
    async def test_history_filters_tool_messages(self, tmp_path: Path) -> None:
        msgs = [
            {"role": "user", "content": "run code", "timestamp": "2026-01-01"},
            {"role": "tool", "content": "tool output", "timestamp": "2026-01-01"},
            {"role": "assistant", "content": "done", "timestamp": "2026-01-01"},
        ]
        app = _app(sessions=_StubSessions(messages=msgs))
        result = await handle_route(
            app, "GET", "/api/chat/history?chat_id=c1", b"", tmp_path,
        )
        roles = [m["role"] for m in result["messages"]]
        assert "tool" not in roles
        assert len(result["messages"]) == 2

    @pytest.mark.asyncio
    async def test_history_content_truncation(self, tmp_path: Path) -> None:
        long_content = "x" * 10000
        msgs = [{"role": "assistant", "content": long_content, "timestamp": "2026-01-01"}]
        app = _app(sessions=_StubSessions(messages=msgs))
        result = await handle_route(
            app, "GET", "/api/chat/history?chat_id=c1", b"", tmp_path,
        )
        assert len(result["messages"][0]["content"]) <= 4000

    @pytest.mark.asyncio
    async def test_history_prefixes_web_key(self, tmp_path: Path) -> None:
        """chat_id without web: prefix should be auto-prefixed for session lookup."""
        msgs = [{"role": "user", "content": "hi", "timestamp": "2026-01-01"}]
        app = _app(sessions=_StubSessions(messages=msgs))
        result = await handle_route(
            app, "GET", "/api/chat/history?chat_id=abc123", b"", tmp_path,
        )
        assert result["chat_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_history_already_prefixed(self, tmp_path: Path) -> None:
        """chat_id already starting with web: should not be double-prefixed."""
        msgs = [{"role": "user", "content": "hi", "timestamp": "2026-01-01"}]
        app = _app(sessions=_StubSessions(messages=msgs))
        result = await handle_route(
            app, "GET", "/api/chat/history?chat_id=web:abc123", b"", tmp_path,
        )
        assert result["chat_id"] == "web:abc123"


# ---- SSEResponse type ----


class TestSSEResponseType:

    def test_sse_response_is_dataclass(self) -> None:
        handler = AsyncMock()
        sse = SSEResponse(handler=handler)
        assert sse.handler is handler

    def test_sse_response_callable(self) -> None:
        async def noop(writer: Any) -> None:
            pass
        sse = SSEResponse(handler=noop)
        assert callable(sse.handler)
