"""Provider pool -- lazy-create LiteLLMProvider instances per provider config."""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Any

from nibot.config import ProviderQuotaConfig, ProvidersConfig
from nibot.log import logger
from nibot.provider import LiteLLMProvider, LLMProvider
from nibot.types import LLMResponse


class ProviderQuota:
    """Three-layer quota tracking: config limits + response header calibration + 429 fallback.

    Layer 1 (config): User declares RPM/TPM based on their tier. NiBot counts internally.
    Layer 2 (headers): Calibrate remaining capacity from x-ratelimit-remaining-* headers.
    Layer 3 (429): On rate limit error, mark provider exhausted for Retry-After seconds.
    """

    def __init__(self, name: str, rpm_limit: int = 0, tpm_limit: int = 0) -> None:
        self.name = name
        self.rpm_limit = rpm_limit  # 0 = unlimited
        self.tpm_limit = tpm_limit
        self._minute_requests: deque[float] = deque()
        self._minute_tokens: deque[tuple[float, int]] = deque()
        self._exhausted_until: float = 0.0
        # Header calibration (from x-ratelimit-remaining-* etc.)
        self._header_remaining_requests: int | None = None
        self._header_remaining_tokens: int | None = None
        self._header_updated_at: float = 0.0  # monotonic timestamp of last header update

    def record_usage(self, tokens: int = 0) -> None:
        """Record one request's usage (self-counting layer)."""
        now = time.monotonic()
        self._minute_requests.append(now)
        if tokens:
            self._minute_tokens.append((now, tokens))

    def update_from_headers(self, remaining_requests: int | None, remaining_tokens: int | None) -> None:
        """Calibrate remaining quota from LLM response headers (header layer)."""
        if remaining_requests is not None:
            self._header_remaining_requests = remaining_requests
            self._header_updated_at = time.monotonic()
        if remaining_tokens is not None:
            self._header_remaining_tokens = remaining_tokens
            self._header_updated_at = time.monotonic()

    def record_rate_limit(self, retry_after: float = 60.0) -> None:
        """Mark provider temporarily unavailable (429 fallback layer)."""
        self._exhausted_until = time.monotonic() + retry_after
        logger.warning(f"Provider '{self.name}' marked exhausted for {retry_after:.0f}s")

    def is_available(self) -> bool:
        """Check all three layers to determine if provider is available."""
        now = time.monotonic()

        # Layer 3: 429 exhaustion check
        if now < self._exhausted_until:
            return False

        # Layer 2: header remaining check (0 means exhausted, expires after 60s)
        header_age = now - self._header_updated_at
        if header_age < 60.0:
            if self._header_remaining_requests is not None and self._header_remaining_requests <= 0:
                return False
            if self._header_remaining_tokens is not None and self._header_remaining_tokens <= 0:
                return False
        else:
            # Stale headers: reset so provider gets a fresh chance
            self._header_remaining_requests = None
            self._header_remaining_tokens = None

        # Layer 1: self-counting RPM check
        if self.rpm_limit:
            self._prune_old(now)
            if len(self._minute_requests) >= self.rpm_limit:
                return False

        # Layer 1: self-counting TPM check
        if self.tpm_limit:
            self._prune_old(now)
            total = sum(t for _, t in self._minute_tokens)
            if total >= self.tpm_limit:
                return False

        return True

    def _prune_old(self, now: float) -> None:
        """Remove entries older than 60 seconds from sliding windows."""
        cutoff = now - 60.0
        while self._minute_requests and self._minute_requests[0] < cutoff:
            self._minute_requests.popleft()
        while self._minute_tokens and self._minute_tokens[0][0] < cutoff:
            self._minute_tokens.popleft()


class ProviderPool:
    """Manage multiple LLM providers with lazy instantiation and caching.

    Each named provider gets its own LiteLLMProvider instance (own env vars, own model).
    Unknown or unconfigured names fall back to the default provider.
    """

    def __init__(
        self,
        providers_config: ProvidersConfig,
        default: LLMProvider,
        quota_configs: dict[str, ProviderQuotaConfig] | None = None,
    ) -> None:
        self._config = providers_config
        self._default = default
        self._cache: dict[str, LiteLLMProvider] = {}
        self._quotas: dict[str, ProviderQuota] = {}
        for name, qc in (quota_configs or {}).items():
            self._quotas[name] = ProviderQuota(name, rpm_limit=qc.rpm, tpm_limit=qc.tpm)

    def get(self, name: str = "") -> LLMProvider:
        """Get provider by name. Empty name or unknown returns default."""
        if not name:
            return self._default
        if name in self._cache:
            return self._cache[name]
        pc = self._config.get(name)
        if not pc or not pc.api_key:
            return self._default
        # Fallback to default provider's model if provider config has no model
        default_model = getattr(self._default, "model", "")
        provider = LiteLLMProvider(
            model=pc.model or default_model, api_key=pc.api_key, api_base=pc.api_base,
        )
        self._cache[name] = provider
        return provider

    def has(self, name: str) -> bool:
        """Check if a named provider is configured with an API key."""
        pc = self._config.get(name)
        return pc is not None and bool(pc.api_key)

    def get_quota(self, name: str) -> ProviderQuota | None:
        """Get quota tracker for a named provider."""
        return self._quotas.get(name)

    async def chat_with_fallback(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        chain: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Try providers in chain order, skipping quota-exhausted ones.

        Args:
            messages: Chat messages.
            tools: Tool definitions.
            chain: Ordered list of provider names to try. Empty = just default.
            **kwargs: Passed through to provider.chat().

        Returns:
            LLMResponse from first successful provider, or error LLMResponse if all fail.
        """
        providers_to_try: list[tuple[str, LLMProvider]] = []
        skipped: list[str] = []
        for name in (chain or []):
            quota = self._quotas.get(name)
            if quota and not quota.is_available():
                skipped.append(name)
                continue
            p = self.get(name)
            if p is not self._default or name == "":
                providers_to_try.append((name, p))
        if skipped:
            logger.debug(f"Quota-exhausted providers skipped: {skipped}")
        if not providers_to_try:
            providers_to_try = [("default", self._default)]

        errors: list[str] = []
        for name, provider in providers_to_try:
            try:
                result = await provider.chat(messages=messages, tools=tools, **kwargs)
                if result.finish_reason != "error":
                    self._record_success(name, result)
                    return result
                errors.append(f"{name}: {result.content}")
                logger.warning(f"Provider '{name}' returned error, trying next...")
            except Exception as e:
                self._record_error(name, e)
                errors.append(f"{name}: {e}")
                logger.warning(f"Provider '{name}' failed: {e}, trying next...")

        error_detail = "; ".join(errors)
        logger.error(f"All providers in chain failed: {error_detail}")
        return LLMResponse(
            content=f"All providers failed: {error_detail}",
            finish_reason="error",
        )

    def _record_success(self, name: str, result: LLMResponse) -> None:
        """Record usage and calibrate quota from successful response."""
        quota = self._quotas.get(name)
        if not quota:
            return
        quota.record_usage(result.usage.get("total_tokens", 0) if result.usage else 0)
        # Calibrate from response headers (use explicit None check -- 0 is a valid value)
        rl = result.ratelimit_info
        if rl:
            remaining_req = rl.get("x-ratelimit-remaining-requests")
            if remaining_req is None:
                remaining_req = rl.get("anthropic-ratelimit-requests-remaining")
            remaining_tok = rl.get("x-ratelimit-remaining-tokens")
            if remaining_tok is None:
                remaining_tok = rl.get("anthropic-ratelimit-tokens-remaining")
            quota.update_from_headers(remaining_req, remaining_tok)

    def _record_error(self, name: str, error: Exception) -> None:
        """Detect rate limit errors and mark provider exhausted."""
        quota = self._quotas.get(name)
        if not quota:
            return
        error_str = str(error).lower()
        if "429" in error_str or "rate" in error_str or "quota" in error_str:
            retry_after = self._parse_retry_after(error)
            quota.record_rate_limit(retry_after)

    @staticmethod
    def _parse_retry_after(error: Exception) -> float:
        """Extract Retry-After from error message, default 60s."""
        match = re.search(r"retry.?after.?(\d+)", str(error), re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 60.0
