"""Subagent manager -- background task execution with tool isolation."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from nibot.bus import MessageBus
from nibot.log import logger
from nibot.provider import LLMProvider
from nibot.registry import ToolRegistry
from nibot.types import Envelope

SUBAGENT_TOOL_DENY = ["message", "spawn"]


class SubagentManager:
    """Spawn isolated background agents that report results via the bus."""

    def __init__(self, provider: LLMProvider, registry: ToolRegistry, bus: MessageBus) -> None:
        self.provider = provider
        self.registry = registry
        self.bus = bus
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        task: str,
        label: str,
        origin_channel: str,
        origin_chat_id: str,
        max_iterations: int = 15,
    ) -> str:
        task_id = uuid.uuid4().hex[:8]
        bg = asyncio.create_task(
            self._run(task_id, task, label, origin_channel, origin_chat_id, max_iterations)
        )
        self._tasks[task_id] = bg
        bg.add_done_callback(lambda _: self._tasks.pop(task_id, None))
        return task_id

    async def _run(
        self,
        task_id: str,
        task: str,
        label: str,
        channel: str,
        chat_id: str,
        max_iterations: int,
    ) -> None:
        tool_defs = self.registry.get_definitions(deny=SUBAGENT_TOOL_DENY)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"You are a subagent. Task ID: {task_id}. Task: {task}"},
            {"role": "user", "content": task},
        ]
        final = ""
        try:
            for _ in range(max_iterations):
                resp = await self.provider.chat(messages=messages, tools=tool_defs or None)
                if not resp.has_tool_calls:
                    final = resp.content or ""
                    break
                tc_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in resp.tool_calls
                ]
                messages.append({"role": "assistant", "content": resp.content, "tool_calls": tc_dicts})
                for tc in resp.tool_calls:
                    result = await self.registry.execute(tc.name, tc.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result.content,
                    })
        except Exception as e:
            logger.error(f"Subagent {task_id} error: {e}")
            final = f"Subagent error: {e}"

        await self.bus.publish_inbound(
            Envelope(
                channel="system",
                sender_id="subagent",
                chat_id=f"{channel}:{chat_id}",
                content=f"[Subagent '{label}' completed]\nResult: {final}",
            )
        )

    def list_active(self) -> list[str]:
        return list(self._tasks.keys())
