"""Provider pool rate limit detection tests (Phase 2 v1.4)."""
from __future__ import annotations

import pytest

from nibot.provider_pool import ProviderPool


class TestIsRateLimitError:
    """_is_rate_limit_error must detect true 429s and reject false positives."""

    def test_rate_limit_error_class(self):
        """LiteLLM RateLimitError type detected by class name."""
        class RateLimitError(Exception):
            pass
        assert ProviderPool._is_rate_limit_error(RateLimitError("too many"))

    def test_status_code_429(self):
        assert ProviderPool._is_rate_limit_error(Exception("status_code=429"))
        assert ProviderPool._is_rate_limit_error(Exception("status: 429"))

    def test_word_boundary_429(self):
        assert ProviderPool._is_rate_limit_error(Exception("HTTP 429 Too Many Requests"))

    def test_false_positive_42900(self):
        """'Error 42900' contains '429' but is NOT a rate limit."""
        assert not ProviderPool._is_rate_limit_error(Exception("Error 42900: SQL syntax"))

    def test_false_positive_disk_rate(self):
        """'disk rate limit' is not an HTTP rate limit."""
        assert not ProviderPool._is_rate_limit_error(Exception("migration rate limited by disk I/O"))

    def test_false_positive_storage_quota(self):
        """'storage quota' is not a provider rate limit."""
        assert not ProviderPool._is_rate_limit_error(Exception("insufficient storage quota"))

    def test_connection_error_not_rate_limit(self):
        assert not ProviderPool._is_rate_limit_error(ConnectionError("connection refused"))

    def test_generic_error_not_rate_limit(self):
        assert not ProviderPool._is_rate_limit_error(ValueError("invalid input"))
