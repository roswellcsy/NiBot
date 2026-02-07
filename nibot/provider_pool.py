"""Provider pool -- lazy-create LiteLLMProvider instances per provider config."""

from __future__ import annotations

from typing import Any

from nibot.config import ProvidersConfig
from nibot.provider import LiteLLMProvider, LLMProvider


class ProviderPool:
    """Manage multiple LLM providers with lazy instantiation and caching.

    Each named provider gets its own LiteLLMProvider instance (own env vars, own model).
    Unknown or unconfigured names fall back to the default provider.
    """

    def __init__(self, providers_config: ProvidersConfig, default: LLMProvider) -> None:
        self._config = providers_config
        self._default = default
        self._cache: dict[str, LiteLLMProvider] = {}

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

    async def chat_with_fallback(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        chain: list[str] | None = None,
        **kwargs: Any,
    ) -> "LLMResponse":
        """Try providers in chain order. First success wins.

        Args:
            messages: Chat messages.
            tools: Tool definitions.
            chain: Ordered list of provider names to try. Empty = just default.
            **kwargs: Passed through to provider.chat().

        Returns:
            LLMResponse from first successful provider, or error LLMResponse if all fail.
        """
        from nibot.log import logger
        from nibot.types import LLMResponse

        providers_to_try: list[tuple[str, LLMProvider]] = []
        for name in (chain or []):
            p = self.get(name)
            if p is not self._default or name == "":
                providers_to_try.append((name, p))
        if not providers_to_try:
            providers_to_try = [("default", self._default)]

        errors: list[str] = []
        for name, provider in providers_to_try:
            try:
                result = await provider.chat(messages=messages, tools=tools, **kwargs)
                if result.finish_reason != "error":
                    return result
                errors.append(f"{name}: {result.content}")
                logger.warning(f"Provider '{name}' returned error, trying next...")
            except Exception as e:
                errors.append(f"{name}: {e}")
                logger.warning(f"Provider '{name}' failed: {e}, trying next...")

        error_detail = "; ".join(errors)
        logger.error(f"All providers in chain failed: {error_detail}")
        return LLMResponse(
            content=f"All providers failed: {error_detail}",
            finish_reason="error",
        )
