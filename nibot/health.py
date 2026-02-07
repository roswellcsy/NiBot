"""Minimal health-check HTTP server -- zero external dependencies.

Uses asyncio.start_server (stdlib) to serve a single JSON endpoint:
  GET /health -> {"status": "ok", "uptime_seconds": ..., ...}

Design: raw TCP, minimal HTTP parsing, Connection: close.
Internal monitoring only -- not a public API.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from nibot.log import logger

if TYPE_CHECKING:
    from nibot.app import NiBot

_START_TIME: float = 0.0


_MAX_REQUEST_LINE = 8192   # 8KB max for request line
_MAX_HEADER_LINES = 64     # refuse more than 64 header lines


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    app: NiBot,
) -> None:
    """Handle one HTTP connection. Parse request line, return JSON, close."""
    try:
        # Read request line with size limit to prevent memory exhaustion
        request_data = await asyncio.wait_for(
            reader.readuntil(b"\n"), timeout=5.0,
        )
        if len(request_data) > _MAX_REQUEST_LINE:
            return  # oversized request, just close
        parts = request_data.decode("utf-8", errors="replace").split()
        path = parts[1] if len(parts) >= 2 else "/"

        # Drain remaining headers with line count limit
        for _ in range(_MAX_HEADER_LINES):
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if line in (b"\r\n", b"\n", b""):
                break
        else:
            return  # too many headers, close

        if path == "/health":
            body: dict[str, Any] = _build_health(app)
            status = "200 OK"
        else:
            body = {"error": "not found"}
            status = "404 Not Found"

        payload = json.dumps(body, ensure_ascii=False)
        response = (
            f"HTTP/1.1 {status}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload.encode())}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{payload}"
        )
        writer.write(response.encode())
        await writer.drain()
    except (asyncio.TimeoutError, ConnectionError, OSError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


def _build_health(app: NiBot) -> dict[str, Any]:
    """Build health response payload. Pure data collection, no I/O."""
    uptime = time.monotonic() - _START_TIME
    return {
        "status": "ok" if app.agent._running else "degraded",
        "uptime_seconds": round(uptime, 1),
        "model": app.config.agent.model,
        "channels": [ch.name for ch in app._channels],
        "active_sessions": len(app.sessions._cache),
        "active_tasks": len(app.agent._tasks),
        "scheduler_jobs": len(app.scheduler._jobs),
    }


async def start_health_server(app: NiBot) -> asyncio.Server | None:
    """Start health HTTP server if enabled in config. Returns Server for cleanup."""
    cfg = app.config.health
    if not cfg.enabled:
        return None

    global _START_TIME
    _START_TIME = time.monotonic()

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _handle_connection(r, w, app)

    server = await asyncio.start_server(handler, cfg.host, cfg.port)
    logger.info(f"Health server listening on {cfg.host}:{cfg.port}")
    return server
