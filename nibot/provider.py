"""LLM Provider abstraction -- LiteLLM backend."""

from __future__ import annotations

import asyncio
import json
import os
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from nibot.log import logger
from nibot.types import LLMResponse, ToolCall, ToolCallDelta


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
        "openai": "OPENAI_API_KEY",
        "gpt": "OPENAI_API_KEY",
        "o1": "OPENAI_API_KEY",
        "o3": "OPENAI_API_KEY",
        "o4": "OPENAI_API_KEY",
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
                    base_delay = self.retry_base_delay * (2 ** attempt)
                    jitter = base_delay * random.uniform(-0.25, 0.25)
                    delay = base_delay + jitter
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_attempts}): {e}, retrying in {delay:.2f}s")
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
    ) -> AsyncIterator[str | ToolCallDelta | LLMResponse]:
        """Stream text/tool-call chunks, then yield final LLMResponse.

        Bypasses LiteLLM's stream_chunk_builder for tool calls (known bugs).
        Self-accumulates tool_call deltas from raw stream. Falls back to
        non-streaming chat() if streaming with tools fails.
        """
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
        if tools:
            kwargs["tools"] = tools

        try:
            resp = await acompletion(**kwargs)
            full_content: list[str] = []
            tc_acc: dict[int, dict[str, Any]] = {}  # index -> {id, name, args_parts}

            async for chunk in resp:
                delta = chunk.choices[0].delta

                # Text delta
                if hasattr(delta, "content") and delta.content:
                    full_content.append(delta.content)
                    yield delta.content

                # Tool call delta (self-accumulated)
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_acc:
                            tc_acc[idx] = {
                                "id": getattr(tc_delta, "id", "") or "",
                                "name": "",
                                "args_parts": [],
                            }
                        acc = tc_acc[idx]
                        if acc["id"] == "" and getattr(tc_delta, "id", ""):
                            acc["id"] = tc_delta.id
                        fn = getattr(tc_delta, "function", None)
                        if fn:
                            if getattr(fn, "name", None):
                                acc["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                acc["args_parts"].append(fn.arguments)
                        yield ToolCallDelta(
                            index=idx,
                            name=acc["name"],
                            partial_args="".join(acc["args_parts"]),
                        )

            # Assemble final tool calls
            tool_calls: list[ToolCall] = []
            for idx in sorted(tc_acc):
                acc = tc_acc[idx]
                args_str = "".join(acc["args_parts"])
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {"raw": args_str} if args_str else {}
                tool_calls.append(ToolCall(
                    id=acc["id"], name=acc["name"], arguments=args,
                ))

            yield LLMResponse(
                content="".join(full_content) or None,
                tool_calls=tool_calls,
            )

        except Exception as e:
            if tools:
                # Fallback: streaming with tools failed, retry non-streaming
                logger.warning(f"Stream+tools failed, falling back to non-stream: {e}")
                resp = await self.chat(messages, tools, model, max_tokens, temperature)
                if resp.has_tool_calls:
                    yield resp
                    return
                if resp.content:
                    yield resp.content
                yield resp
            else:
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
