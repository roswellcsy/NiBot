"""Web panel route handlers."""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from nibot.log import logger
from nibot.types import Envelope


async def handle_route(
    app: Any, method: str, path: str, body: bytes, static_dir: Path,
) -> Any:
    """Route dispatcher. Returns dict (JSON), bytes (static file), or SSEResponse."""

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

    # Chat API routes
    if clean_path == "/api/chat/send" and method == "POST":
        return await _chat_send(app, body)
    if clean_path == "/api/chat/stream":
        stream_id = query.get("id", [""])[0]
        return _chat_stream(app, stream_id)
    if clean_path == "/api/chat/sessions":
        return _chat_sessions(app)
    if clean_path == "/api/chat/history":
        chat_id = query.get("chat_id", [""])[0]
        limit = int(query.get("limit", ["50"])[0])
        return _chat_history(app, chat_id, limit)

    # Management API routes
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


# ---- Web Chat API ----


async def _chat_send(app: Any, body: bytes) -> dict[str, Any]:
    """Send a chat message and return a stream_id for SSE consumption."""
    data = json.loads(body) if body else {}
    content = data.get("content", "").strip()
    chat_id = data.get("chat_id", "")

    if not content:
        return {"error": "empty content", "status": 400}

    if not chat_id:
        chat_id = f"web_{uuid.uuid4().hex[:8]}"

    stream_id = uuid.uuid4().hex[:12]
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    streams: dict[str, asyncio.Queue[Any]] = getattr(app, "_web_streams", {})
    streams[stream_id] = queue

    await app.bus.publish_inbound(Envelope(
        channel="web",
        chat_id=chat_id,
        sender_id="web_user",
        content=content,
        metadata={"stream_id": stream_id},
    ))

    # Fallback cleanup if SSE never connects
    async def _cleanup() -> None:
        await asyncio.sleep(60.0)
        streams.pop(stream_id, None)

    cleanup_task = asyncio.create_task(_cleanup())
    cleanups: dict[str, asyncio.Task[None]] = getattr(app, "_web_stream_cleanups", {})
    if not hasattr(app, "_web_stream_cleanups"):
        app._web_stream_cleanups = cleanups
    cleanups[stream_id] = cleanup_task

    return {"stream_id": stream_id, "chat_id": chat_id}


def _chat_stream(app: Any, stream_id: str) -> Any:
    """Return an SSEResponse that streams agent output to the client."""
    from nibot.web.server import SSEResponse

    if not stream_id:
        return {"error": "id parameter required", "status": 400}

    streams: dict[str, asyncio.Queue[Any]] = getattr(app, "_web_streams", {})
    queue = streams.get(stream_id)
    if not queue:
        return {"error": "stream not found", "status": 404}

    async def _sse_handler(writer: asyncio.StreamWriter) -> None:
        headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n"
        )
        writer.write(headers.encode())
        await writer.drain()
        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=120.0)
                if item is None:
                    writer.write(b"data: [DONE]\n\n")
                    await writer.drain()
                    break
                event = json.dumps(item, ensure_ascii=False)
                writer.write(f"data: {event}\n\n".encode())
                await writer.drain()
        except asyncio.TimeoutError:
            logger.info(f"SSE {stream_id} closed: idle timeout")
        except (ConnectionError, OSError) as e:
            logger.info(f"SSE {stream_id} closed: client disconnected ({e})")
        finally:
            streams.pop(stream_id, None)
            # Cancel fallback cleanup timer
            cleanups: dict[str, asyncio.Task[None]] = getattr(app, "_web_stream_cleanups", {})
            task = cleanups.pop(stream_id, None)
            if task and not task.done():
                task.cancel()
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    return SSEResponse(handler=_sse_handler)


def _chat_sessions(app: Any) -> dict[str, Any]:
    """List web chat sessions (key starts with 'web:')."""
    all_sessions = app.sessions.query_recent(limit=50)
    web_sessions = [s for s in all_sessions if s["key"].startswith("web:")]
    return {"sessions": web_sessions}


def _chat_history(app: Any, chat_id: str, limit: int = 50) -> dict[str, Any]:
    """Get message history for a web chat session."""
    if not chat_id:
        return {"error": "chat_id parameter required", "status": 400}
    key = f"web:{chat_id}" if not chat_id.startswith("web:") else chat_id
    messages = app.sessions.get_session_messages(key, limit=limit)
    return {
        "chat_id": chat_id,
        "messages": [
            {
                "role": m.get("role", ""),
                "content": (m.get("content") or "")[:4000],
                "timestamp": m.get("timestamp", ""),
            }
            for m in messages
            if m.get("role") in ("user", "assistant")
        ],
    }
