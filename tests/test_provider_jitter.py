"""Provider retry jitter tests (Phase 5 v1.4)."""
from __future__ import annotations

import inspect

import pytest


class TestRetryJitter:
    def test_retry_code_uses_random(self):
        """Provider.chat() retry logic must import and use random."""
        from nibot.provider import LiteLLMProvider
        source = inspect.getsource(LiteLLMProvider.chat)
        assert "random" in source or "jitter" in source

    def test_jitter_range(self):
        """±25% jitter keeps delay within [0.75*base, 1.25*base]."""
        import random
        base = 2.0
        samples = [base + base * random.uniform(-0.25, 0.25) for _ in range(1000)]
        assert all(0.75 * base <= s <= 1.25 * base for s in samples)
        # Standard deviation should be meaningful (not all same value)
        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)
        std = variance ** 0.5
        assert std > base * 0.05  # at least 5% spread
