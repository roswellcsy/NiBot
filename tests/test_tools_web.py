"""Web tools tests -- SSRF protection, HA fallback, truncation."""
from __future__ import annotations

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.tools.web_tools import WebFetchTool, WebSearchTool, _is_private_url


# ---------------------------------------------------------------------------
# _is_private_url  SSRF protection
# ---------------------------------------------------------------------------

class TestIsPrivateUrl:
    """SSRF guard: _is_private_url must block private/reserved addresses."""

    @patch("nibot.tools.web_tools.socket.getaddrinfo")
    def test_blocks_localhost_ipv4(self, mock_dns: MagicMock) -> None:
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]
        assert _is_private_url("http://localhost/secret") is True

    @patch("nibot.tools.web_tools.socket.getaddrinfo")
    def test_blocks_127_x(self, mock_dns: MagicMock) -> None:
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.2", 0))]
        assert _is_private_url("http://something/") is True

    @patch("nibot.tools.web_tools.socket.getaddrinfo")
    def test_blocks_10_x(self, mock_dns: MagicMock) -> None:
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))]
        assert _is_private_url("http://internal.corp/") is True

    @patch("nibot.tools.web_tools.socket.getaddrinfo")
    def test_blocks_172_16_x(self, mock_dns: MagicMock) -> None:
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("172.16.0.1", 0))]
        assert _is_private_url("http://host/") is True

    @patch("nibot.tools.web_tools.socket.getaddrinfo")
    def test_blocks_172_31_x(self, mock_dns: MagicMock) -> None:
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("172.31.255.255", 0))]
        assert _is_private_url("http://host/") is True

    @patch("nibot.tools.web_tools.socket.getaddrinfo")
    def test_blocks_192_168_x(self, mock_dns: MagicMock) -> None:
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 0))]
        assert _is_private_url("http://router/") is True

    @patch("nibot.tools.web_tools.socket.getaddrinfo")
    def test_blocks_ipv6_loopback(self, mock_dns: MagicMock) -> None:
        mock_dns.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 0, 0, 0))]
        assert _is_private_url("http://[::1]/") is True

    @patch("nibot.tools.web_tools.socket.getaddrinfo")
    def test_blocks_link_local(self, mock_dns: MagicMock) -> None:
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 0))]
        assert _is_private_url("http://metadata.google.internal/") is True

    @patch("nibot.tools.web_tools.socket.getaddrinfo")
    def test_allows_public_ip(self, mock_dns: MagicMock) -> None:
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
        assert _is_private_url("http://example.com/") is False

    def test_blocks_empty_url(self) -> None:
        assert _is_private_url("") is True

    def test_blocks_no_hostname(self) -> None:
        assert _is_private_url("file:///etc/passwd") is True

    @patch("nibot.tools.web_tools.socket.getaddrinfo", side_effect=socket.gaierror("DNS fail"))
    def test_blocks_dns_failure(self, _: MagicMock) -> None:
        assert _is_private_url("http://nonexistent.invalid/") is True

    @patch("nibot.tools.web_tools.socket.getaddrinfo")
    def test_dns_rebinding_any_private_blocks(self, mock_dns: MagicMock) -> None:
        """If DNS returns multiple IPs and ANY is private, block."""
        mock_dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
        ]
        assert _is_private_url("http://evil.com/") is True


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------

class TestWebSearchTool:

    @pytest.mark.asyncio
    async def test_no_keys_returns_not_configured(self) -> None:
        tool = WebSearchTool(api_key="", anthropic_api_key="")
        result = await tool.execute(query="test")
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_anthropic_primary_success(self) -> None:
        tool = WebSearchTool(api_key="brave_key", anthropic_api_key="ant_key")
        with patch.object(tool, "_anthropic_search", new_callable=AsyncMock, return_value="Anthropic result"):
            result = await tool.execute(query="test")
        assert result == "Anthropic result"

    @pytest.mark.asyncio
    async def test_anthropic_fail_brave_fallback(self) -> None:
        tool = WebSearchTool(api_key="brave_key", anthropic_api_key="ant_key")
        with (
            patch.object(tool, "_anthropic_search", new_callable=AsyncMock, side_effect=RuntimeError("down")),
            patch.object(tool, "_brave_search", new_callable=AsyncMock, return_value="Brave result"),
        ):
            result = await tool.execute(query="test")
        assert result == "Brave result"

    @pytest.mark.asyncio
    async def test_anthropic_empty_falls_through(self) -> None:
        """Anthropic returns empty string -> fall through to Brave."""
        tool = WebSearchTool(api_key="brave_key", anthropic_api_key="ant_key")
        with (
            patch.object(tool, "_anthropic_search", new_callable=AsyncMock, return_value=""),
            patch.object(tool, "_brave_search", new_callable=AsyncMock, return_value="Brave result"),
        ):
            result = await tool.execute(query="test")
        assert result == "Brave result"

    @pytest.mark.asyncio
    async def test_both_fail_returns_error(self) -> None:
        tool = WebSearchTool(api_key="brave_key", anthropic_api_key="ant_key")
        with (
            patch.object(tool, "_anthropic_search", new_callable=AsyncMock, side_effect=RuntimeError("a")),
            patch.object(tool, "_brave_search", new_callable=AsyncMock, side_effect=RuntimeError("b")),
        ):
            result = await tool.execute(query="test")
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_brave_only_key(self) -> None:
        """Only Brave key configured, Anthropic skipped."""
        tool = WebSearchTool(api_key="brave_key", anthropic_api_key="")
        with patch.object(tool, "_brave_search", new_callable=AsyncMock, return_value="Brave only"):
            result = await tool.execute(query="test")
        assert result == "Brave only"


# ---------------------------------------------------------------------------
# WebFetchTool
# ---------------------------------------------------------------------------

class TestWebFetchTool:

    @pytest.mark.asyncio
    async def test_blocks_private_url(self) -> None:
        tool = WebFetchTool()
        with patch("nibot.tools.web_tools._is_private_url", return_value=True):
            result = await tool.execute(url="http://127.0.0.1/secret")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_truncates_long_response(self) -> None:
        import httpx as _httpx

        tool = WebFetchTool()
        big_text = "x" * 30000

        mock_resp = MagicMock()
        mock_resp.text = big_text
        mock_resp.url = "http://example.com/"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("nibot.tools.web_tools._is_private_url", return_value=False),
            patch.object(_httpx, "AsyncClient", return_value=mock_client),
        ):
            result = await tool.execute(url="http://example.com/big", max_length=100)
        assert len(result) <= 120  # 100 + "... (truncated)" overhead
        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_blocks_redirect_to_private(self) -> None:
        """Public URL redirects to private IP -> blocked."""
        import httpx as _httpx

        tool = WebFetchTool()

        mock_resp = MagicMock()
        mock_resp.text = "secret data"
        mock_resp.url = "http://127.0.0.1/internal"  # final URL after redirect
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        def side_effect(url: str) -> bool:
            if "127.0.0.1" in url:
                return True
            return False

        with (
            patch("nibot.tools.web_tools._is_private_url", side_effect=side_effect),
            patch.object(_httpx, "AsyncClient", return_value=mock_client),
        ):
            result = await tool.execute(url="http://evil.com/redirect")
        assert "Blocked" in result
