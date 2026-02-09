"""Web tools -- search and fetch."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

from nibot.log import logger
from nibot.registry import Tool


def _is_private_url(url: str) -> bool:
    """Block requests to private/reserved IP ranges (SSRF protection)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if not hostname:
            return True
        addrs = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canon, sockaddr in addrs:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return True
    except (socket.gaierror, ValueError):
        return True
    return False


class WebSearchTool(Tool):
    """HA web search: Anthropic server-side search (primary) → Brave (fallback)."""

    def __init__(self, api_key: str = "", anthropic_api_key: str = "") -> None:
        self._brave_api_key = api_key
        self._anthropic_api_key = anthropic_api_key

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs["query"]
        count = kwargs.get("count", 5)

        # Primary: Anthropic built-in web search (server-side, via Haiku)
        if self._anthropic_api_key:
            try:
                result = await self._anthropic_search(query, count)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Anthropic web search failed: {e}, falling back to Brave")

        # Fallback: Brave Search API
        if self._brave_api_key:
            try:
                return await self._brave_search(query, count)
            except Exception as e:
                logger.warning(f"Brave web search failed: {e}")
                return f"Web search error: {e}"

        return "Web search not configured (missing API key)."

    async def _anthropic_search(self, query: str, count: int) -> str:
        """Use Anthropic Messages API with built-in web_search tool (Haiku for cost)."""
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1024,
                    "messages": [{
                        "role": "user",
                        "content": (
                            f"Search the web for: {query}\n"
                            f"Return the top {count} results. For each result, provide:\n"
                            f"- Title (bold)\n- URL\n- Brief description\n"
                            f"Be concise. Use the web search tool."
                        ),
                    }],
                    "tools": [{
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract text blocks from response content
        texts = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    texts.append(text)
        return "\n".join(texts)

    async def _brave_search(self, query: str, count: int) -> str:
        """Brave Search API fallback."""
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={"X-Subscription-Token": self._brave_api_key, "Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        results = data.get("web", {}).get("results", [])
        if not results:
            return "No results found."
        lines = []
        for r in results[:count]:
            lines.append(f"**{r.get('title', '')}**")
            lines.append(r.get("url", ""))
            lines.append(r.get("description", ""))
            lines.append("")
        return "\n".join(lines)


class WebFetchTool(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch a URL and return its text content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "max_length": {"type": "integer", "description": "Max chars", "default": 20000},
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        import httpx

        url = kwargs["url"]
        if _is_private_url(url):
            return "Error: URL points to a private/reserved address. Blocked for security."
        max_length = kwargs.get("max_length", 20000)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            # Check final URL after redirects (SSRF: redirect to private IP)
            final_url = str(resp.url)
            if final_url != url and _is_private_url(final_url):
                return "Error: URL redirected to a private/reserved address. Blocked for security."
        text = resp.text
        # Try to extract readable content if lxml is available
        try:
            from readability import Document

            doc = Document(text)
            from lxml.html import fromstring, tostring

            tree = fromstring(doc.summary())
            text = tree.text_content()
        except ImportError:
            pass
        text = text.strip()
        if len(text) > max_length:
            text = text[:max_length] + "\n... (truncated)"
        return text
