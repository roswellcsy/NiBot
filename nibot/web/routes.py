"""Web panel route handlers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


async def handle_route(
    app: Any, method: str, path: str, body: bytes, static_dir: Path,
) -> dict[str, Any] | bytes:
    """Route dispatcher. Returns dict (JSON) or bytes (static file)."""

    # Parse query string
    parsed = urlparse(path)
    clean_path = parsed.path
    query = parse_qs(parsed.query)

    # Static files
    if clean_path == "/" or clean_path == "/index.html":
        index = static_dir / "index.html"
        if index.exists():
            return index.read_bytes()
        return {"error": "dashboard not found"}

    # API routes
    if clean_path == "/api/health":
        return _health(app)
    if clean_path == "/api/sessions":
        return _sessions(app)
    if clean_path == "/api/sessions/messages":
        key = query.get("key", [""])[0]
        limit = int(query.get("limit", ["50"])[0])
        return _session_messages(app, key, limit)
    if clean_path == "/api/skills":
        if method == "GET":
            return _skills_list(app)
        if method == "DELETE":
            data = json.loads(body) if body else {}
            return await _skills_delete(app, data.get("name", ""))
    if clean_path == "/api/skills/reload":
        if method == "POST":
            return _skills_reload(app)
    if clean_path == "/api/config":
        if method == "GET":
            return _config_get(app)
    if clean_path == "/api/analytics":
        return _analytics(app)
    if clean_path == "/api/tasks":
        return _tasks(app)

    return {"error": "not found", "status": 404}


def _health(app: Any) -> dict[str, Any]:
    import time
    return {
        "status": "ok" if app.agent._running else "stopped",
        "model": app.config.agent.model,
        "channels": [ch.name for ch in app._channels],
        "active_sessions": len(app.sessions._cache),
        "active_tasks": len(app.agent._tasks),
    }


def _sessions(app: Any) -> dict[str, Any]:
    sessions = app.sessions.query_recent(limit=50)
    return {"sessions": sessions, "total": len(sessions)}


def _session_messages(app: Any, key: str, limit: int = 50) -> dict[str, Any]:
    if not key:
        return {"error": "key parameter required"}
    messages = app.sessions.get_session_messages(key, limit=limit)
    return {
        "key": key,
        "messages": [
            {
                "role": m.get("role", ""),
                "content": (m.get("content") or "")[:2000],
                "timestamp": m.get("timestamp", ""),
            }
            for m in messages
        ],
        "total": len(messages),
    }


def _skills_list(app: Any) -> dict[str, Any]:
    skills = app.skills.get_all()
    return {
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "always": s.always,
                "version": s.version,
                "created_by": s.created_by,
                "executable": s.executable,
            }
            for s in skills
        ]
    }


async def _skills_delete(app: Any, name: str) -> dict[str, Any]:
    if not name:
        return {"error": "name required"}
    import shutil
    for d in app.skills.skills_dirs:
        candidate = (d / name).resolve()
        if not candidate.is_relative_to(d.resolve()):
            return {"error": "invalid skill name"}
        if candidate.is_dir():
            shutil.rmtree(candidate)
            app.skills.reload()
            return {"status": "deleted", "name": name}
    return {"error": f"skill '{name}' not found"}


def _skills_reload(app: Any) -> dict[str, Any]:
    app.skills.reload()
    return {"status": "reloaded", "count": len(app.skills.get_all())}


def _config_get(app: Any) -> dict[str, Any]:
    return {
        "agent": {
            "model": app.config.agent.model,
            "temperature": app.config.agent.temperature,
            "max_tokens": app.config.agent.max_tokens,
            "max_iterations": app.config.agent.max_iterations,
        },
        "agents": {k: {"tools": v.tools, "model": v.model} for k, v in (app.config.agents or {}).items()},
        "tools": {
            "sandbox_enabled": app.config.tools.sandbox_enabled,
            "sandbox_memory_mb": app.config.tools.sandbox_memory_mb,
            "exec_timeout": app.config.tools.exec_timeout,
        },
    }


def _analytics(app: Any) -> dict[str, Any]:
    from nibot.metrics import aggregate_metrics, compute_session_metrics
    sessions = app.sessions.iter_recent_from_disk(limit=50)
    if not sessions:
        return {"sessions": 0}
    per_session = [compute_session_metrics(s.messages) for s in sessions]
    agg = aggregate_metrics(per_session)
    return agg.to_dict()


def _tasks(app: Any) -> dict[str, Any]:
    tasks = app.subagents.list_tasks(limit=20)
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "agent_type": t.agent_type,
                "label": t.label,
                "status": t.status,
                "created_at": t.created_at.isoformat(),
            }
            for t in tasks
        ]
    }
