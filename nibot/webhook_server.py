"""Unified webhook/API HTTP server -- asyncio-based, zero dependencies."""
from __future__ import annotations

import asyncio
import json
from typing import Any, TYPE_CHECKING

from nibot.log import logger

if TYPE_CHECKING:
    from nibot.channels.api import APIChannel
    from nibot.channels.wecom import WeComChannel


class WebhookServer:
    """Lightweight HTTP server for webhook callbacks and API requests.

    Uses asyncio.start_server for zero-dependency operation.
    Handles: POST /webhook/wecom, POST /api/chat
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        wecom_channel: "WeComChannel | None" = None,
        api_channel: "APIChannel | None" = None,
    ) -> None:
        self._host = host
        self._port = port
        self._wecom = wecom_channel
        self._api = api_channel
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._host,
            self._port,
        )
        logger.info(f"Webhook server listening on {self._host}:{self._port}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            # Read request line
            request_line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            parts = request_line.decode("utf-8", errors="replace").split()
            method = parts[0] if parts else "GET"
            path = parts[1] if len(parts) >= 2 else "/"

            # Read headers
            headers: dict[str, str] = {}
            content_length = 0
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line in (b"\r\n", b"\n", b""):
                    break
                decoded = line.decode("utf-8", errors="replace").strip()
                if ":" in decoded:
                    key, value = decoded.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
                    if key.strip().lower() == "content-length":
                        content_length = int(value.strip())

            # Read body
            body = b""
            if content_length > 0:
                body = await asyncio.wait_for(
                    reader.readexactly(content_length), timeout=10.0
                )

            # Parse query params
            query_params: dict[str, str] = {}
            if "?" in path:
                path, qs = path.split("?", 1)
                for pair in qs.split("&"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        query_params[k] = v

            # Route
            result = await self._route(method, path, body, headers, query_params)
            await self._send_response(writer, result)
        except (asyncio.TimeoutError, ConnectionError, OSError):
            pass
        except Exception as e:
            logger.error(f"Webhook handler error: {e}")
            try:
                await self._send_response(writer, {"error": str(e)}, status=500)
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _route(
        self,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
        query_params: dict[str, str],
    ) -> dict[str, Any]:
        if path == "/webhook/wecom" and self._wecom:
            return await self._wecom.handle_webhook(body, query_params)

        if path == "/api/chat" and method == "POST" and self._api:
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                return {"error": "invalid JSON", "status": 400}

            auth_header = headers.get("authorization", "")
            token = (
                auth_header.replace("Bearer ", "")
                if auth_header.startswith("Bearer ")
                else ""
            )

            return await self._api.handle_request(
                content=data.get("content", ""),
                sender_id=data.get("sender_id", "api"),
                chat_id=data.get("chat_id", ""),
                auth_token=token,
                timeout=float(data.get("timeout", 60)),
            )

        return {"error": "not found", "status": 404}

    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        data: dict[str, Any],
        status: int = 0,
    ) -> None:
        status_code = data.pop("status", None) or status or 200
        status_text = {
            200: "OK",
            400: "Bad Request",
            401: "Unauthorized",
            404: "Not Found",
            500: "Internal Server Error",
            504: "Gateway Timeout",
        }.get(status_code, "OK")
        payload = json.dumps(data, ensure_ascii=False)
        response = (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload.encode())}\r\n"
            f"Connection: close\r\n\r\n{payload}"
        )
        writer.write(response.encode())
        await writer.drain()
