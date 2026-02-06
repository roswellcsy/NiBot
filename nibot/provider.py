"""LLM Provider abstraction -- LiteLLM backend."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
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


class LiteLLMProvider(LLMProvider):
    """LiteLLM-backed provider supporting 100+ LLM APIs."""

    def __init__(
        self,
        model: str,
        api_key: str = "",
        api_base: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if api_base:
            os.environ.setdefault("OPENAI_API_BASE", api_base)
        if api_key:
            self._configure_api_key(api_key, model)

    def _configure_api_key(self, api_key: str, model: str) -> None:
        if api_key.startswith("sk-or-"):
            os.environ.setdefault("OPENROUTER_API_KEY", api_key)
        elif "anthropic" in model or "claude" in model:
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        elif "deepseek" in model:
            os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
        else:
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
        if tools:
            kwargs["tools"] = tools
        try:
            resp = await acompletion(**kwargs)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return LLMResponse(content=f"LLM error: {e}", finish_reason="error")
        return self._parse(resp)

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
        return LLMResponse(
            content=getattr(msg, "content", None),
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=dict(resp.usage) if resp.usage else {},
        )
