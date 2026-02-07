"""Tests for v1.0.0: multi-user isolation + rate limiting."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from nibot.rate_limiter import RateLimitConfig, SlidingWindowRateLimiter
from nibot.types import ToolContext


class TestToolContextSenderId:
    """ToolContext should include sender_id."""

    def test_tool_context_has_sender_id(self) -> None:
        ctx = ToolContext(channel="tg", chat_id="123", session_key="tg:123", sender_id="user1")
        assert ctx.sender_id == "user1"

    def test_tool_context_sender_id_default_empty(self) -> None:
        ctx = ToolContext()
        assert ctx.sender_id == ""

    def test_agent_process_propagates_sender_id(self) -> None:
        """Verify AgentLoop._process creates ToolContext with sender_id."""
        import inspect
        from nibot.agent import AgentLoop
        source = inspect.getsource(AgentLoop._process)
        assert "sender_id" in source, "AgentLoop._process must set sender_id on ToolContext"


class TestSlidingWindowRateLimiter:
    """Rate limiter correctness."""

    def test_disabled_always_allows(self) -> None:
        rl = SlidingWindowRateLimiter(RateLimitConfig(enabled=False))
        for _ in range(1000):
            allowed, _ = rl.check("user1", "ch1")
            assert allowed

    def test_per_user_limit(self) -> None:
        rl = SlidingWindowRateLimiter(RateLimitConfig(enabled=True, per_user_rpm=3, per_channel_rpm=100))
        for _ in range(3):
            allowed, _ = rl.check("user1", "ch1")
            assert allowed
        allowed, reason = rl.check("user1", "ch1")
        assert not allowed
        assert "user" in reason.lower()

    def test_different_users_independent(self) -> None:
        rl = SlidingWindowRateLimiter(RateLimitConfig(enabled=True, per_user_rpm=2, per_channel_rpm=100))
        rl.check("user1", "ch1")
        rl.check("user1", "ch1")
        allowed, _ = rl.check("user1", "ch1")
        assert not allowed
        # user2 should still be allowed
        allowed, _ = rl.check("user2", "ch1")
        assert allowed

    def test_per_channel_limit(self) -> None:
        rl = SlidingWindowRateLimiter(RateLimitConfig(enabled=True, per_user_rpm=100, per_channel_rpm=2))
        rl.check("user1", "ch1")
        rl.check("user2", "ch1")
        allowed, reason = rl.check("user3", "ch1")
        assert not allowed
        assert "channel" in reason.lower()

    def test_window_expiry(self) -> None:
        rl = SlidingWindowRateLimiter(RateLimitConfig(enabled=True, per_user_rpm=1, per_channel_rpm=100))
        rl.check("user1", "ch1")
        allowed, _ = rl.check("user1", "ch1")
        assert not allowed

        # Simulate time passing by manipulating the deque
        if "user1" in rl._user_windows:
            rl._user_windows["user1"].clear()
        allowed, _ = rl.check("user1", "ch1")
        assert allowed

    def test_reset_user(self) -> None:
        rl = SlidingWindowRateLimiter(RateLimitConfig(enabled=True, per_user_rpm=1, per_channel_rpm=100))
        rl.check("user1", "ch1")
        allowed, _ = rl.check("user1", "ch1")
        assert not allowed
        rl.reset(user_key="user1")
        allowed, _ = rl.check("user1", "ch1")
        assert allowed

    def test_reset_all(self) -> None:
        rl = SlidingWindowRateLimiter(RateLimitConfig(enabled=True, per_user_rpm=1, per_channel_rpm=100))
        rl.check("user1", "ch1")
        rl.check("user2", "ch2")
        rl.reset()
        assert rl.stats() == {"tracked_users": 0, "tracked_channels": 0}

    def test_stats(self) -> None:
        rl = SlidingWindowRateLimiter(RateLimitConfig(enabled=True, per_user_rpm=10, per_channel_rpm=100))
        rl.check("user1", "ch1")
        rl.check("user2", "ch2")
        stats = rl.stats()
        assert stats["tracked_users"] == 2
        assert stats["tracked_channels"] == 2

    def test_empty_keys_skip_check(self) -> None:
        rl = SlidingWindowRateLimiter(RateLimitConfig(enabled=True, per_user_rpm=1, per_channel_rpm=1))
        # Empty keys should be skipped
        allowed, _ = rl.check("", "")
        assert allowed
