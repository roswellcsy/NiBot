"""Web tools -- search and fetch."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

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
    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web using Brave Search API."

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
        if not self._api_key:
            return "Web search not configured (missing API key)."
        import httpx

        query = kwargs["query"]
        count = kwargs.get("count", 5)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={"X-Subscription-Token": self._api_key, "Accept": "application/json"},
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
