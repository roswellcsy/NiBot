"""LLM Provider abstraction -- LiteLLM backend."""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from nibot.log import logger
from nibot.types import LLMResponse, ToolCall


class LLMProvider(ABC):
    """Abstract LLM provider. Implement for custom backends."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse: ...

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str | LLMResponse]:
        """Stream text chunks. Yields str for text, LLMResponse for tool calls.

        Default falls back to non-streaming chat().
        """
        resp = await self.chat(messages, tools, model, max_tokens, temperature)
        if resp.has_tool_calls:
            yield resp
        elif resp.content:
            yield resp.content


class LiteLLMProvider(LLMProvider):
    """LiteLLM-backed provider supporting 100+ LLM APIs."""

    def __init__(
        self,
        model: str,
        api_key: str = "",
        api_base: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        # Still set env vars for the default provider (backward compat with litellm auto-detect)
        if api_base:
            os.environ.setdefault("OPENAI_API_BASE", api_base)
        if api_key:
            self._configure_env_key(api_key, model)

    # Model prefix -> env var name for litellm auto-detection.
    _ENV_KEY_MAP: dict[str, str] = {
        "anthropic": "ANTHROPIC_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }

    def _configure_env_key(self, api_key: str, model: str) -> None:
        """Set env vars for litellm auto-detection (default provider only)."""
        if api_key.startswith("sk-or-"):
            os.environ.setdefault("OPENROUTER_API_KEY", api_key)
            return
        model_lower = model.lower()
        for prefix, env_var in self._ENV_KEY_MAP.items():
            if prefix in model_lower:
                os.environ.setdefault(env_var, api_key)
                return
        os.environ.setdefault("OPENAI_API_KEY", api_key)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
        max_tokens: int = 0,
        temperature: float = -1.0,
    ) -> LLMResponse:
        from litellm import acompletion

        model = model or self.model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature >= 0 else self.temperature,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if tools:
            kwargs["tools"] = tools
        max_attempts = max(1, self.max_retries)
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                resp = await acompletion(**kwargs)
                return self._parse(resp)
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_attempts}): {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
        logger.error(f"LLM call failed after {max_attempts} attempts: {last_error}")
        return LLMResponse(content=f"LLM error: {type(last_error).__name__}", finish_reason="error")

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "",
        max_tokens: int = 0,
        temperature: float = -1.0,
    ) -> AsyncIterator[str | LLMResponse]:
        """Stream text chunks, then yield final LLMResponse.

        When tools are present, falls back to non-streaming chat().
        Yields LLMResponse with tool_calls if LLM chose to call tools.
        """
        if tools:
            resp = await self.chat(messages, tools, model, max_tokens, temperature)
            if resp.has_tool_calls:
                yield resp
                return
            if resp.content:
                yield resp.content
            yield resp
            return

        from litellm import acompletion

        model = model or self.model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature >= 0 else self.temperature,
            "stream": True,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        try:
            resp = await acompletion(**kwargs)
            full: list[str] = []
            async for chunk in resp:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    full.append(delta.content)
                    yield delta.content
            yield LLMResponse(content="".join(full))
        except Exception as e:
            logger.error(f"LLM stream failed: {e}")
            yield LLMResponse(content=f"LLM error: {type(e).__name__}", finish_reason="error")

    def _parse(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        # Extract rate limit headers from LiteLLM response (if available)
        ratelimit_info: dict[str, int] = {}
        hidden = getattr(resp, "_hidden_params", None) or {}
        headers = hidden.get("additional_headers", None) or {}
        _RL_KEYS = (
            "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
            "anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-tokens-remaining",
        )
        for key in _RL_KEYS:
            val = headers.get(key)
            if val is not None:
                try:
                    ratelimit_info[key] = int(val)
                except (ValueError, TypeError):
                    pass
        return LLMResponse(
            content=getattr(msg, "content", None),
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=dict(resp.usage) if resp.usage else {},
            ratelimit_info=ratelimit_info,
        )
