"""Web panel HTTP server -- asyncio-based management interface."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from nibot.log import logger

if TYPE_CHECKING:
    from nibot.app import NiBot


_MAX_BODY = 1_048_576  # 1 MB


@dataclass
class SSEResponse:
    """Sentinel: route handler wants to stream SSE to the client."""
    handler: Callable[[asyncio.StreamWriter], Awaitable[None]]


class WebPanel:
    """Lightweight management web panel.

    Uses asyncio.start_server for zero-dependency operation.
    Serves REST API + static HTML dashboard.
    """

    def __init__(self, app: Any, host: str = "127.0.0.1", port: int = 9200,
                 auth_token: str = "") -> None:
        self._app = app
        self._host = host
        self._port = port
        self._auth_token = auth_token
        self._server: asyncio.Server | None = None
        self._static_dir = Path(__file__).parent / "static"

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, self._host, self._port,
        )
        logger.info(f"Web panel at http://{self._host}:{self._port}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        sse_handled = False
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            parts = line.decode("utf-8", errors="replace").split()
            method = parts[0] if parts else "GET"
            path = parts[1] if len(parts) >= 2 else "/"

            headers: dict[str, str] = {}
            content_length = 0
            while True:
                h = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if h in (b"\r\n", b"\n", b""):
                    break
                decoded = h.decode("utf-8", errors="replace").strip()
                if ":" in decoded:
                    k, v = decoded.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
                    if k.strip().lower() == "content-length":
                        content_length = int(v.strip())

            body = b""
            if content_length > _MAX_BODY:
                await self._respond(writer, {"error": "payload too large"}, status=413)
                return
            if content_length > 0:
                body = await asyncio.wait_for(reader.readexactly(content_length), timeout=10.0)

            # Auth check for API routes
            if path.startswith("/api/") and self._auth_token:
                auth = headers.get("authorization", "")
                token = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else ""
                # Fallback: query param ?token=xxx (EventSource can't set headers)
                if not token:
                    qs = parse_qs(urlparse(path).query)
                    token = qs.get("token", [""])[0]
                if token != self._auth_token:
                    await self._respond(writer, {"error": "unauthorized"}, status=401)
                    return

            from nibot.web.routes import handle_route
            result = await handle_route(self._app, method, path, body, self._static_dir)

            if isinstance(result, SSEResponse):
                sse_handled = True
                await result.handler(writer)
            elif isinstance(result, bytes):
                # Static file
                content_type = "text/html" if path.endswith(".html") or path == "/" else "application/octet-stream"
                resp = (f"HTTP/1.1 200 OK\r\n"
                       f"Content-Type: {content_type}\r\n"
                       f"Content-Length: {len(result)}\r\n"
                       f"Connection: close\r\n\r\n")
                writer.write(resp.encode() + result)
            else:
                status_code = result.pop("status", 200) if isinstance(result.get("status"), int) else 200
                await self._respond(writer, result, status=status_code)
            if not sse_handled:
                await writer.drain()
        except Exception as e:
            logger.debug(f"Web panel error: {e}")
        finally:
            if not sse_handled:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

    async def _respond(self, writer: asyncio.StreamWriter, data: dict[str, Any],
                      status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        status_text = {200: "OK", 400: "Bad Request", 401: "Unauthorized",
                      404: "Not Found", 500: "Error"}.get(status, "OK")
        resp = (f"HTTP/1.1 {status} {status_text}\r\n"
               f"Content-Type: application/json\r\n"
               f"Access-Control-Allow-Origin: *\r\n"
               f"Content-Length: {len(payload.encode())}\r\n"
               f"Connection: close\r\n\r\n{payload}")
        writer.write(resp.encode())
