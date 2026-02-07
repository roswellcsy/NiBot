"""Delegate tool -- typed sub-agent dispatch with task monitoring."""

from __future__ import annotations

from typing import Any

from nibot.config import AgentTypeConfig
from nibot.registry import Tool
from nibot.subagent import SubagentManager
from nibot.types import ToolContext


class DelegateTool(Tool):
    def __init__(
        self, subagents: SubagentManager, agents_config: dict[str, AgentTypeConfig],
    ) -> None:
        self._subagents = subagents
        self._agents = agents_config
        self._ctx: ToolContext | None = None

    def receive_context(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "delegate"

    @property
    def description(self) -> str:
        types = ", ".join(self._agents.keys()) if self._agents else "none"
        return (
            f"Delegate tasks to specialized agents, or query/list task status. "
            f"Available types: {types}."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["spawn", "query", "list"],
                    "description": "spawn: delegate a task, query: get task status, list: show tasks",
                },
                "agent_type": {
                    "type": "string",
                    "description": "Type of agent (e.g. coder, researcher, system, evolution)",
                },
                "task": {"type": "string", "description": "Detailed task description"},
                "label": {"type": "string", "description": "Short tracking label"},
                "task_id": {"type": "string", "description": "Task ID (for query action)"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        ctx = self._ctx
        self._ctx = None  # consume once
        action = kwargs.get("action", "spawn")

        if action == "list":
            return self._list_tasks()

        if action == "query":
            return self._query_task(kwargs.get("task_id", ""))

        # Default: spawn
        agent_type = kwargs.get("agent_type", "")
        task_desc = kwargs.get("task", "")
        if not agent_type or not task_desc:
            return "Error: 'agent_type' and 'task' are required for spawn."
        agent_config = self._agents.get(agent_type)
        if not agent_config:
            available = ", ".join(self._agents.keys())
            return f"Unknown agent type '{agent_type}'. Available: {available}"
        task_id = await self._subagents.spawn(
            task=task_desc,
            label=kwargs.get("label", agent_type),
            origin_channel=ctx.channel if ctx else "",
            origin_chat_id=ctx.chat_id if ctx else "",
            agent_type=agent_type,
            agent_config=agent_config,
        )
        mode_info = ""
        if agent_config.workspace_mode == "worktree":
            mode_info = f" [worktree: task/{task_id}]"
        return f"Delegated to {agent_type}: task_id={task_id}{mode_info}"

    def _list_tasks(self) -> str:
        tasks = self._subagents.list_tasks(limit=20)
        if not tasks:
            return "No tasks found."
        lines = []
        for t in tasks:
            elapsed = ""
            if t.finished_at:
                secs = (t.finished_at - t.created_at).total_seconds()
                elapsed = f" ({secs:.0f}s)"
            lines.append(f"  {t.task_id} [{t.status}] {t.agent_type}/{t.label}{elapsed}")
        return "Tasks:\n" + "\n".join(lines)

    def _query_task(self, task_id: str) -> str:
        if not task_id:
            return "Error: 'task_id' is required for query."
        info = self._subagents.get_task_info(task_id)
        if not info:
            return f"Task '{task_id}' not found."
        result_preview = info.result[:500] if info.result else "(no result yet)"
        return (
            f"Task: {info.task_id}\n"
            f"Type: {info.agent_type}\n"
            f"Label: {info.label}\n"
            f"Status: {info.status}\n"
            f"Created: {info.created_at.isoformat()}\n"
            f"Result: {result_preview}"
        )
