"""v1.2 Web panel: route handlers, session messages, skill management, config."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from nibot.web.routes import handle_route


# ---- Stub objects ----


@dataclass
class _StubChannel:
    name: str = "test"


@dataclass
class _StubAgentConfig:
    model: str = "anthropic/claude-opus-4-6"
    temperature: float = 1.0
    max_tokens: int = 16384
    max_iterations: int = 25


@dataclass
class _StubToolsConfig:
    sandbox_enabled: bool = True
    sandbox_memory_mb: int = 512
    exec_timeout: int = 60


@dataclass
class _StubAgentTypeConfig:
    tools: list[str] = field(default_factory=lambda: ["file_read"])
    model: str = ""


@dataclass
class _StubSkillSpec:
    name: str = "test_skill"
    description: str = "A test skill"
    always: bool = False
    version: int = 1
    created_by: str = "user"
    executable: bool = False


@dataclass
class _StubTask:
    task_id: str = "t-001"
    agent_type: str = "coder"
    label: str = "test task"
    status: str = "completed"
    created_at: datetime = field(default_factory=datetime.now)


class _StubSkills:
    def __init__(self, skills: list[_StubSkillSpec] | None = None) -> None:
        self._skills = skills or []
        self.skills_dirs: list[Path] = []
        self._reloaded = False

    def get_all(self) -> list[_StubSkillSpec]:
        return self._skills

    def reload(self) -> None:
        self._reloaded = True


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

    def iter_recent_from_disk(self, limit: int = 50) -> list[Any]:
        return []


class _StubSubagents:
    def __init__(self, tasks: list[_StubTask] | None = None) -> None:
        self._tasks_list = tasks or []

    def list_tasks(self, limit: int = 20) -> list[_StubTask]:
        return self._tasks_list[:limit]


class _StubAgent:
    def __init__(self) -> None:
        self._running = True
        self._tasks: list[Any] = []


class _StubConfig:
    def __init__(self) -> None:
        self.agent = _StubAgentConfig()
        self.tools = _StubToolsConfig()
        self.agents: dict[str, _StubAgentTypeConfig] = {
            "coder": _StubAgentTypeConfig(tools=["file_read", "write_file", "exec"]),
        }


class _StubApp:
    """Minimal app stub for route testing."""

    def __init__(self) -> None:
        self.agent = _StubAgent()
        self.config = _StubConfig()
        self._channels = [_StubChannel("discord"), _StubChannel("api")]
        self.sessions = _StubSessions()
        self.skills = _StubSkills()
        self.subagents = _StubSubagents()


def _app(**overrides: Any) -> _StubApp:
    app = _StubApp()
    for k, v in overrides.items():
        setattr(app, k, v)
    return app


# ---- Static file serving ----


class TestStaticRoutes:

    @pytest.mark.asyncio
    async def test_root_serves_index(self, tmp_path: Path) -> None:
        index = tmp_path / "index.html"
        index.write_bytes(b"<html>dashboard</html>")
        result = await handle_route(_app(), "GET", "/", b"", tmp_path)
        assert isinstance(result, bytes)
        assert b"dashboard" in result

    @pytest.mark.asyncio
    async def test_index_html_path(self, tmp_path: Path) -> None:
        index = tmp_path / "index.html"
        index.write_bytes(b"<html>ok</html>")
        result = await handle_route(_app(), "GET", "/index.html", b"", tmp_path)
        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_missing_index(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/", b"", tmp_path)
        assert isinstance(result, dict)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/nonexistent", b"", tmp_path)
        assert isinstance(result, dict)
        assert result.get("status") == 404


# ---- Health endpoint ----


class TestHealthRoute:

    @pytest.mark.asyncio
    async def test_health_ok(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/api/health", b"", tmp_path)
        assert result["status"] == "ok"
        assert result["model"] == "anthropic/claude-opus-4-6"
        assert result["active_sessions"] == 0
        assert len(result["channels"]) == 2

    @pytest.mark.asyncio
    async def test_health_stopped(self, tmp_path: Path) -> None:
        app = _app()
        app.agent._running = False
        result = await handle_route(app, "GET", "/api/health", b"", tmp_path)
        assert result["status"] == "stopped"


# ---- Sessions endpoint ----


class TestSessionsRoute:

    @pytest.mark.asyncio
    async def test_sessions_list(self, tmp_path: Path) -> None:
        sessions_data = [
            {"key": "s1", "messages": 5, "tool_calls": 2, "errors": 0, "updated_at": "2026-01-01"},
            {"key": "s2", "messages": 3, "tool_calls": 1, "errors": 1, "updated_at": "2026-01-02"},
        ]
        app = _app(sessions=_StubSessions(sessions=sessions_data))
        result = await handle_route(app, "GET", "/api/sessions", b"", tmp_path)
        assert result["total"] == 2
        assert len(result["sessions"]) == 2

    @pytest.mark.asyncio
    async def test_sessions_empty(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/api/sessions", b"", tmp_path)
        assert result["total"] == 0


# ---- Session messages endpoint ----


class TestSessionMessagesRoute:

    @pytest.mark.asyncio
    async def test_messages_with_key(self, tmp_path: Path) -> None:
        msgs = [
            {"role": "user", "content": "hello", "timestamp": "2026-01-01T00:00:00"},
            {"role": "assistant", "content": "hi there", "timestamp": "2026-01-01T00:00:01"},
        ]
        app = _app(sessions=_StubSessions(messages=msgs))
        result = await handle_route(
            app, "GET", "/api/sessions/messages?key=test-session", b"", tmp_path,
        )
        assert result["key"] == "test-session"
        assert result["total"] == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["content"] == "hi there"

    @pytest.mark.asyncio
    async def test_messages_missing_key(self, tmp_path: Path) -> None:
        result = await handle_route(
            _app(), "GET", "/api/sessions/messages", b"", tmp_path,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_messages_content_truncation(self, tmp_path: Path) -> None:
        long_content = "x" * 5000
        msgs = [{"role": "user", "content": long_content, "timestamp": "2026-01-01"}]
        app = _app(sessions=_StubSessions(messages=msgs))
        result = await handle_route(
            app, "GET", "/api/sessions/messages?key=s1", b"", tmp_path,
        )
        assert len(result["messages"][0]["content"]) <= 2000

    @pytest.mark.asyncio
    async def test_messages_with_limit(self, tmp_path: Path) -> None:
        msgs = [
            {"role": "user", "content": f"msg{i}", "timestamp": f"2026-01-01T00:00:{i:02d}"}
            for i in range(10)
        ]
        app = _app(sessions=_StubSessions(messages=msgs))
        result = await handle_route(
            app, "GET", "/api/sessions/messages?key=s1&limit=3", b"", tmp_path,
        )
        assert result["total"] == 3


# ---- Skills endpoint ----


class TestSkillsRoute:

    @pytest.mark.asyncio
    async def test_skills_list(self, tmp_path: Path) -> None:
        skills = [
            _StubSkillSpec(name="greet", description="Greeting skill", executable=False),
            _StubSkillSpec(name="deploy", description="Deploy skill", executable=True),
        ]
        app = _app(skills=_StubSkills(skills))
        result = await handle_route(app, "GET", "/api/skills", b"", tmp_path)
        assert len(result["skills"]) == 2
        assert result["skills"][0]["name"] == "greet"
        assert result["skills"][0]["executable"] is False
        assert result["skills"][1]["executable"] is True

    @pytest.mark.asyncio
    async def test_skills_empty(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/api/skills", b"", tmp_path)
        assert result["skills"] == []

    @pytest.mark.asyncio
    async def test_skills_reload(self, tmp_path: Path) -> None:
        skills_loader = _StubSkills([_StubSkillSpec()])
        app = _app(skills=skills_loader)
        result = await handle_route(app, "POST", "/api/skills/reload", b"", tmp_path)
        assert result["status"] == "reloaded"
        assert skills_loader._reloaded is True

    @pytest.mark.asyncio
    async def test_skills_delete_missing_name(self, tmp_path: Path) -> None:
        result = await handle_route(
            _app(), "DELETE", "/api/skills", json.dumps({}).encode(), tmp_path,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_skills_delete_not_found(self, tmp_path: Path) -> None:
        skills_loader = _StubSkills()
        skills_loader.skills_dirs = [tmp_path / "skills"]
        (tmp_path / "skills").mkdir()
        app = _app(skills=skills_loader)
        body = json.dumps({"name": "nonexistent"}).encode()
        result = await handle_route(app, "DELETE", "/api/skills", body, tmp_path)
        assert "not found" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_skills_delete_path_traversal_blocked(self, tmp_path: Path) -> None:
        skills_loader = _StubSkills()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skills_loader.skills_dirs = [skills_dir]
        app = _app(skills=skills_loader)
        body = json.dumps({"name": "../../etc"}).encode()
        result = await handle_route(app, "DELETE", "/api/skills", body, tmp_path)
        assert "invalid" in result.get("error", "")


# ---- Config endpoint ----


class TestConfigRoute:

    @pytest.mark.asyncio
    async def test_config_get(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/api/config", b"", tmp_path)
        assert result["agent"]["model"] == "anthropic/claude-opus-4-6"
        assert result["agent"]["temperature"] == 1.0
        assert result["agent"]["max_tokens"] == 16384
        assert result["agent"]["max_iterations"] == 25

    @pytest.mark.asyncio
    async def test_config_tools_section(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/api/config", b"", tmp_path)
        assert result["tools"]["sandbox_enabled"] is True
        assert result["tools"]["sandbox_memory_mb"] == 512
        assert result["tools"]["exec_timeout"] == 60

    @pytest.mark.asyncio
    async def test_config_agents_section(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/api/config", b"", tmp_path)
        assert "coder" in result["agents"]
        assert "file_read" in result["agents"]["coder"]["tools"]


# ---- Tasks endpoint ----


class TestTasksRoute:

    @pytest.mark.asyncio
    async def test_tasks_list(self, tmp_path: Path) -> None:
        tasks = [
            _StubTask(task_id="t-001", agent_type="coder", label="fix bug", status="completed"),
            _StubTask(task_id="t-002", agent_type="researcher", label="analyze", status="running"),
        ]
        app = _app(subagents=_StubSubagents(tasks))
        result = await handle_route(app, "GET", "/api/tasks", b"", tmp_path)
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["task_id"] == "t-001"
        assert result["tasks"][1]["status"] == "running"

    @pytest.mark.asyncio
    async def test_tasks_empty(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/api/tasks", b"", tmp_path)
        assert result["tasks"] == []


# ---- Analytics endpoint ----


class TestAnalyticsRoute:

    @pytest.mark.asyncio
    async def test_analytics_no_sessions(self, tmp_path: Path) -> None:
        result = await handle_route(_app(), "GET", "/api/analytics", b"", tmp_path)
        assert result.get("sessions") == 0 or "session_count" in result
