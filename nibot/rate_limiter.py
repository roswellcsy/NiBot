"""Sliding window rate limiter -- zero external dependencies."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    per_user_rpm: int = 30      # requests per minute per user
    per_channel_rpm: int = 100  # requests per minute per channel
    enabled: bool = False


class SlidingWindowRateLimiter:
    """Token-bucket style rate limiter using sliding windows.

    Uses collections.deque for O(1) append and efficient cleanup.
    Thread-safe within single asyncio event loop (no locks needed).
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self._config = config or RateLimitConfig()
        self._user_windows: dict[str, deque[float]] = {}
        self._channel_windows: dict[str, deque[float]] = {}

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def check(self, user_key: str, channel_key: str) -> tuple[bool, str]:
        """Check if request is allowed.

        Returns (allowed, reason). If not allowed, reason explains why.
        """
        if not self._config.enabled:
            return True, ""

        now = time.monotonic()
        window = 60.0  # 1 minute window

        # Check per-user limit
        if self._config.per_user_rpm > 0 and user_key:
            allowed, reason = self._check_window(
                self._user_windows, user_key, now, window,
                self._config.per_user_rpm, "user"
            )
            if not allowed:
                return False, reason

        # Check per-channel limit
        if self._config.per_channel_rpm > 0 and channel_key:
            allowed, reason = self._check_window(
                self._channel_windows, channel_key, now, window,
                self._config.per_channel_rpm, "channel"
            )
            if not allowed:
                return False, reason

        # Record the request
        if user_key:
            self._record(self._user_windows, user_key, now)
        if channel_key:
            self._record(self._channel_windows, channel_key, now)

        return True, ""

    def _check_window(
        self,
        windows: dict[str, deque[float]],
        key: str,
        now: float,
        window_size: float,
        limit: int,
        label: str,
    ) -> tuple[bool, str]:
        """Check if key is within rate limit."""
        if key not in windows:
            return True, ""
        dq = windows[key]
        # Purge expired entries
        cutoff = now - window_size
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            return False, f"Rate limit exceeded for {label} '{key}': {limit} requests per minute"
        return True, ""

    def _record(self, windows: dict[str, deque[float]], key: str, now: float) -> None:
        """Record a request timestamp."""
        if key not in windows:
            windows[key] = deque()
        windows[key].append(now)

    def reset(self, user_key: str = "", channel_key: str = "") -> None:
        """Reset rate limit counters for a specific key or all."""
        if user_key:
            self._user_windows.pop(user_key, None)
        if channel_key:
            self._channel_windows.pop(channel_key, None)
        if not user_key and not channel_key:
            self._user_windows.clear()
            self._channel_windows.clear()

    def stats(self) -> dict[str, int]:
        """Return current tracked keys count."""
        return {
            "tracked_users": len(self._user_windows),
            "tracked_channels": len(self._channel_windows),
        }
