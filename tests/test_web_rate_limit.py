"""Web panel rate limiting tests (Phase 7)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from nibot.web.server import WebPanel


# ---- Minimal stub app ----

@dataclass
class _StubApp:
    """Just enough for WebPanel to handle /api/ routes."""
    bus: Any = None
    channels: list[Any] = field(default_factory=list)
    _web_chat_queues: dict[str, Any] = field(default_factory=dict)


async def _http_get(host: str, port: int, path: str) -> tuple[int, str]:
    """Send a minimal HTTP GET and return (status_code, body)."""
    reader, writer = await asyncio.open_connection(host, port)
    request = f"GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n"
    writer.write(request.encode())
    await writer.drain()

    # Read response
    data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass

    text = data.decode("utf-8", errors="replace")
    # Parse status code from first line: "HTTP/1.1 429 Too Many Requests\r\n..."
    first_line = text.split("\r\n", 1)[0]
    parts = first_line.split(" ", 2)
    status = int(parts[1]) if len(parts) >= 2 else 0
    # Body is after \r\n\r\n
    body = text.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in text else ""
    return status, body


@pytest.mark.asyncio
async def test_web_panel_rate_limit_blocks():
    """rpm=2: third API request within 60s window returns 429."""
    panel = WebPanel(_StubApp(), host="127.0.0.1", port=0, rate_limit_rpm=2)
    # port=0 → OS picks a free port
    await panel.start()
    assert panel._server is not None
    # Get the actual bound port
    sockets = panel._server.sockets
    port = sockets[0].getsockname()[1]

    try:
        # First two requests should get through (404 from missing route is fine)
        s1, _ = await _http_get("127.0.0.1", port, "/api/status")
        s2, _ = await _http_get("127.0.0.1", port, "/api/status")
        assert s1 != 429, "first request should not be rate limited"
        assert s2 != 429, "second request should not be rate limited"

        # Third request should be rate limited
        s3, body3 = await _http_get("127.0.0.1", port, "/api/status")
        assert s3 == 429
        assert "rate limit" in body3.lower()
    finally:
        await panel.stop()


@pytest.mark.asyncio
async def test_web_panel_rate_limit_zero_disables():
    """rpm=0: rate limiting disabled, all requests pass."""
    panel = WebPanel(_StubApp(), host="127.0.0.1", port=0, rate_limit_rpm=0)
    await panel.start()
    port = panel._server.sockets[0].getsockname()[1]

    try:
        for _ in range(5):
            status, _ = await _http_get("127.0.0.1", port, "/api/status")
            assert status != 429, "should not rate limit when rpm=0"
    finally:
        await panel.stop()
