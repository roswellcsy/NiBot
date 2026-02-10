"""Web chat API tests -- route handlers, SSE, auth, rate limiting."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nibot.web.routes import (
    _MAX_CONCURRENT_STREAMS,
    _chat_history,
    _chat_send,
    _chat_sessions,
    _chat_stream,
    _health,
    handle_route,
)


def _mock_app(tmp_path: Path | None = None) -> MagicMock:
    """Create a minimal mock app with required attributes."""
    app = MagicMock()
    app._web_streams = {}
    app._web_chat_secrets = {}
    app._web_stream_cleanups = {}
    app.bus = MagicMock()
    app.bus.publish_inbound = AsyncMock()
    app.agent = MagicMock()
    app.agent._running = True
    app.agent._tasks = set()
    app.config = MagicMock()
    app.config.agent.model = "test-model"
    app._channels = []
    app.sessions = MagicMock()
    app.sessions._cache = {}
    return app


# ---------------------------------------------------------------------------
# handle_route dispatcher
# ---------------------------------------------------------------------------

class TestRouteDispatcher:

    @pytest.mark.asyncio
    async def test_index_route(self, tmp_path: Path) -> None:
        app = _mock_app()
        static_dir = tmp_path
        (tmp_path / "index.html").write_text("<html>dashboard</html>", encoding="utf-8")
        result = await handle_route(app, "GET", "/", b"", static_dir)
        assert isinstance(result, bytes)
        assert b"dashboard" in result

    @pytest.mark.asyncio
    async def test_not_found_route(self, tmp_path: Path) -> None:
        app = _mock_app()
        result = await handle_route(app, "GET", "/api/nonexistent", b"", tmp_path)
        assert result.get("error") == "not found"

    @pytest.mark.asyncio
    async def test_health_route(self, tmp_path: Path) -> None:
        app = _mock_app()
        result = await handle_route(app, "GET", "/api/health", b"", tmp_path)
        assert result["status"] == "ok"
        assert result["model"] == "test-model"


# ---------------------------------------------------------------------------
# POST /api/chat/send
# ---------------------------------------------------------------------------

class TestChatSend:

    @pytest.mark.asyncio
    async def test_send_returns_stream_id(self) -> None:
        app = _mock_app()
        body = json.dumps({"content": "hello", "chat_id": "c1"}).encode()
        result = await _chat_send(app, body)
        assert "stream_id" in result
        assert result["chat_id"] == "c1"
        assert result["stream_id"] in app._web_streams

    @pytest.mark.asyncio
    async def test_send_empty_content_rejected(self) -> None:
        app = _mock_app()
        body = json.dumps({"content": "", "chat_id": "c1"}).encode()
        result = await _chat_send(app, body)
        assert "error" in result
        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_send_auto_generates_chat_id(self) -> None:
        app = _mock_app()
        body = json.dumps({"content": "hello"}).encode()
        result = await _chat_send(app, body)
        assert result["chat_id"].startswith("web_")

    @pytest.mark.asyncio
    async def test_send_publishes_to_bus(self) -> None:
        app = _mock_app()
        body = json.dumps({"content": "test msg", "chat_id": "c1"}).encode()
        await _chat_send(app, body)
        app.bus.publish_inbound.assert_called_once()
        envelope = app.bus.publish_inbound.call_args[0][0]
        assert envelope.content == "test msg"
        assert envelope.channel == "web"

    @pytest.mark.asyncio
    async def test_send_generates_session_secret(self) -> None:
        app = _mock_app()
        body = json.dumps({"content": "hi", "chat_id": "c1"}).encode()
        result = await _chat_send(app, body)
        assert "secret" in result
        assert len(result["secret"]) > 0
        assert app._web_chat_secrets["c1"] == result["secret"]


# ---------------------------------------------------------------------------
# GET /api/chat/stream
# ---------------------------------------------------------------------------

class TestChatStream:

    def test_stream_missing_id(self) -> None:
        app = _mock_app()
        result = _chat_stream(app, "")
        assert result["error"] == "id parameter required"

    def test_stream_not_found(self) -> None:
        app = _mock_app()
        result = _chat_stream(app, "nonexistent")
        assert result["error"] == "stream not found"
        assert result["status"] == 404

    def test_stream_returns_sse_response(self) -> None:
        from nibot.web.server import SSEResponse
        app = _mock_app()
        queue: asyncio.Queue = asyncio.Queue()
        app._web_streams["sid123"] = queue
        result = _chat_stream(app, "sid123")
        assert isinstance(result, SSEResponse)

    def test_stream_max_concurrent_limit(self) -> None:
        app = _mock_app()
        # Fill up to max
        for i in range(_MAX_CONCURRENT_STREAMS + 1):
            app._web_streams[f"s{i}"] = asyncio.Queue()
        result = _chat_stream(app, "snew")
        assert result["status"] == 503


# ---------------------------------------------------------------------------
# GET /api/chat/sessions
# ---------------------------------------------------------------------------

class TestChatSessions:

    def test_lists_web_sessions_only(self) -> None:
        app = _mock_app()
        app.sessions.query_recent.return_value = [
            {"key": "web:c1", "messages": 5},
            {"key": "telegram:t1", "messages": 3},
            {"key": "web:c2", "messages": 10},
        ]
        result = _chat_sessions(app)
        assert len(result["sessions"]) == 2
        assert all(s["key"].startswith("web:") for s in result["sessions"])


# ---------------------------------------------------------------------------
# GET /api/chat/history
# ---------------------------------------------------------------------------

class TestChatHistory:

    def test_missing_chat_id(self) -> None:
        app = _mock_app()
        result = _chat_history(app, "", 50)
        assert result["status"] == 400

    def test_wrong_secret_forbidden(self) -> None:
        app = _mock_app()
        app._web_chat_secrets["c1"] = "correct_secret"
        result = _chat_history(app, "c1", 50, secret="wrong")
        assert result["status"] == 403

    def test_correct_secret_returns_messages(self) -> None:
        app = _mock_app()
        app._web_chat_secrets["c1"] = "mysecret"
        app.sessions.get_session_messages.return_value = [
            {"role": "user", "content": "hello", "timestamp": "t1"},
            {"role": "assistant", "content": "hi there", "timestamp": "t2"},
            {"role": "tool", "content": "result", "timestamp": "t3"},
        ]
        result = _chat_history(app, "c1", 50, secret="mysecret")
        # Only user and assistant messages returned
        assert len(result["messages"]) == 2

    def test_no_secret_required_when_not_set(self) -> None:
        app = _mock_app()
        app.sessions.get_session_messages.return_value = [
            {"role": "user", "content": "hi", "timestamp": "t"},
        ]
        result = _chat_history(app, "c1", 50)
        assert "messages" in result


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_running(self) -> None:
        app = _mock_app()
        result = _health(app)
        assert result["status"] == "ok"

    def test_health_stopped(self) -> None:
        app = _mock_app()
        app.agent._running = False
        result = _health(app)
        assert result["status"] == "stopped"
