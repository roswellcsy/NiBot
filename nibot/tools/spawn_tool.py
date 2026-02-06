"""Subagent spawn tool."""

from __future__ import annotations

from typing import Any

from nibot.registry import Tool
from nibot.subagent import SubagentManager


class SpawnTool(Tool):
    def __init__(self, subagents: SubagentManager) -> None:
        self._subagents = subagents
        self._origin_channel = ""
        self._origin_chat_id = ""

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return "Spawn a background subagent to handle a task."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task description for the subagent"},
                "label": {"type": "string", "description": "Short label for tracking"},
            },
            "required": ["task", "label"],
        }

    def set_origin(self, channel: str, chat_id: str) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    async def execute(self, **kwargs: Any) -> str:
        task_id = await self._subagents.spawn(
            task=kwargs["task"],
            label=kwargs["label"],
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )
        return f"Subagent spawned: id={task_id}, label={kwargs['label']}"
