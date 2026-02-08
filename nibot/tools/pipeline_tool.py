"""Pipeline tool -- DAG-based multi-agent orchestration."""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nibot.log import logger
from nibot.registry import Tool
from nibot.subagent import SubagentManager
from nibot.types import ToolContext


@dataclass
class PipelineStep:
    """A single step in a pipeline."""

    id: str
    agent_type: str
    task: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class StepExecution:
    """Runtime state of a pipeline step."""

    step: PipelineStep
    status: str = "pending"  # pending | running | completed | failed | skipped
    task_id: str = ""
    result: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass
class PipelineExecution:
    """Runtime state of a complete pipeline."""

    pipeline_id: str
    steps: dict[str, StepExecution] = field(default_factory=dict)
    status: str = "running"  # running | completed | failed | cancelled
    created_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    origin_channel: str = ""
    origin_chat_id: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class PipelineEngine:
    """DAG scheduler for multi-agent pipelines."""

    def __init__(self, subagents: SubagentManager, agents_config: dict[str, Any]) -> None:
        self._subagents = subagents
        self._agents_config = agents_config
        self._pipelines: dict[str, PipelineExecution] = {}
        self._max_pipelines = 50

    async def create(
        self,
        steps: list[PipelineStep],
        origin_channel: str = "",
        origin_chat_id: str = "",
    ) -> str:
        """Create and start a pipeline from step definitions."""
        self._prune()

        pipeline_id = uuid.uuid4().hex[:8]
        execution = PipelineExecution(
            pipeline_id=pipeline_id,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
        )
        for step in steps:
            execution.steps[step.id] = StepExecution(step=step)

        # Validate dependency references
        step_ids = set(execution.steps.keys())
        for se in execution.steps.values():
            for dep in se.step.depends_on:
                if dep not in step_ids:
                    raise ValueError(f"Step '{se.step.id}' depends on unknown step '{dep}'")

        self._pipelines[pipeline_id] = execution

        # Start scheduling
        asyncio.create_task(self._schedule(pipeline_id))

        return pipeline_id

    async def _schedule(self, pipeline_id: str) -> None:
        """Main scheduling loop: find ready steps and dispatch them."""
        execution = self._pipelines.get(pipeline_id)
        if not execution:
            return

        try:
            while execution.status == "running":
                async with execution.lock:
                    ready = self._find_ready_steps(execution)

                    if not ready and not self._has_running_steps(execution):
                        all_completed = all(
                            se.status in ("completed", "skipped")
                            for se in execution.steps.values()
                        )
                        execution.status = "completed" if all_completed else "failed"
                        execution.finished_at = datetime.now()
                        break

                    tasks = []
                    for step_exec in ready:
                        tasks.append(self._dispatch_step(execution, step_exec))

                if tasks:
                    await asyncio.gather(*tasks)
                else:
                    await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} scheduler error: {e}")
            execution.status = "failed"
            execution.finished_at = datetime.now()

    def _find_ready_steps(self, execution: PipelineExecution) -> list[StepExecution]:
        """Find steps whose dependencies are all completed."""
        ready = []
        for se in execution.steps.values():
            if se.status != "pending":
                continue
            deps_failed = any(
                execution.steps[dep].status in ("failed", "skipped")
                for dep in se.step.depends_on
                if dep in execution.steps
            )
            if deps_failed:
                se.status = "skipped"
                se.result = "Skipped: upstream dependency failed"
                se.finished_at = datetime.now()
                continue
            deps_met = all(
                execution.steps[dep].status == "completed"
                for dep in se.step.depends_on
                if dep in execution.steps
            )
            if deps_met:
                ready.append(se)
        return ready

    def _has_running_steps(self, execution: PipelineExecution) -> bool:
        return any(se.status == "running" for se in execution.steps.values())

    async def _dispatch_step(self, execution: PipelineExecution, step_exec: StepExecution) -> None:
        """Spawn a subagent for a pipeline step."""
        step = step_exec.step
        agent_config = self._agents_config.get(step.agent_type)

        step_exec.status = "running"
        step_exec.started_at = datetime.now()

        async def on_step_complete(task_id: str, result: str) -> None:
            step_exec.result = result
            step_exec.finished_at = datetime.now()
            step_exec.status = "completed"

        try:
            task_id = await self._subagents.spawn(
                task=step.task,
                label=f"pipeline-{execution.pipeline_id}-{step.id}",
                origin_channel=execution.origin_channel,
                origin_chat_id=execution.origin_chat_id,
                agent_type=step.agent_type,
                agent_config=agent_config,
                on_complete=on_step_complete,
            )
            step_exec.task_id = task_id
        except Exception as e:
            step_exec.status = "failed"
            step_exec.result = f"Spawn error: {e}"
            step_exec.finished_at = datetime.now()

    def get_status(self, pipeline_id: str) -> dict[str, Any] | None:
        """Get pipeline status."""
        execution = self._pipelines.get(pipeline_id)
        if not execution:
            return None
        steps = {}
        for sid, se in execution.steps.items():
            steps[sid] = {
                "agent_type": se.step.agent_type,
                "status": se.status,
                "task_id": se.task_id,
                "result": (se.result or "")[:200],
                "depends_on": se.step.depends_on,
            }
        return {
            "pipeline_id": pipeline_id,
            "status": execution.status,
            "created_at": execution.created_at.isoformat(),
            "finished_at": execution.finished_at.isoformat() if execution.finished_at else None,
            "steps": steps,
        }

    def cancel(self, pipeline_id: str) -> bool:
        """Cancel a running pipeline."""
        execution = self._pipelines.get(pipeline_id)
        if not execution or execution.status != "running":
            return False
        execution.status = "cancelled"
        execution.finished_at = datetime.now()
        for se in execution.steps.values():
            if se.status in ("pending", "running"):
                se.status = "skipped"
                se.result = "Cancelled"
        return True

    def list_pipelines(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent pipelines."""
        pipelines = sorted(
            self._pipelines.values(),
            key=lambda p: p.created_at,
            reverse=True,
        )[:limit]
        return [
            {
                "pipeline_id": p.pipeline_id,
                "status": p.status,
                "steps": len(p.steps),
                "created_at": p.created_at.isoformat(),
            }
            for p in pipelines
        ]

    def _prune(self) -> None:
        """Remove old completed pipelines."""
        if len(self._pipelines) <= self._max_pipelines:
            return
        completed = sorted(
            [(pid, p) for pid, p in self._pipelines.items() if p.status != "running"],
            key=lambda x: x[1].created_at,
        )
        while len(self._pipelines) > self._max_pipelines and completed:
            pid, _ = completed.pop(0)
            del self._pipelines[pid]


class PipelineTool(Tool):
    """Create and manage multi-agent pipelines through conversation."""

    def __init__(self, engine: PipelineEngine) -> None:
        self._engine = engine
        self._ctx: ToolContext | None = None

    def receive_context(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "pipeline"

    @property
    def description(self) -> str:
        return (
            "Create and manage multi-agent pipelines. "
            "Steps run in dependency order; independent steps run in parallel."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "status", "cancel", "list"],
                },
                "steps": {
                    "type": "array",
                    "description": "Pipeline steps (for create). Each: {id, agent_type, task, depends_on?}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "agent_type": {"type": "string"},
                            "task": {"type": "string"},
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "pipeline_id": {"type": "string", "description": "Pipeline ID (for status/cancel)"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        ctx = self._ctx
        self._ctx = None
        action = kwargs["action"]

        if action == "list":
            pipelines = self._engine.list_pipelines()
            if not pipelines:
                return "No pipelines found."
            lines = [
                f"  {p['pipeline_id']} [{p['status']}] {p['steps']} steps ({p['created_at']})"
                for p in pipelines
            ]
            return "Pipelines:\n" + "\n".join(lines)

        if action == "status":
            pid = kwargs.get("pipeline_id", "")
            if not pid:
                return "Error: 'pipeline_id' required for status."
            info = self._engine.get_status(pid)
            if not info:
                return f"Pipeline '{pid}' not found."
            return json.dumps(info, indent=2)

        if action == "cancel":
            pid = kwargs.get("pipeline_id", "")
            if not pid:
                return "Error: 'pipeline_id' required for cancel."
            if self._engine.cancel(pid):
                return f"Pipeline '{pid}' cancelled."
            return f"Pipeline '{pid}' not found or not running."

        if action == "create":
            raw_steps = kwargs.get("steps", [])
            if not raw_steps:
                return "Error: 'steps' required for create."
            steps = []
            for s in raw_steps:
                steps.append(PipelineStep(
                    id=s.get("id", ""),
                    agent_type=s.get("agent_type", ""),
                    task=s.get("task", ""),
                    depends_on=s.get("depends_on", []),
                ))
            try:
                pid = await self._engine.create(
                    steps,
                    origin_channel=ctx.channel if ctx else "",
                    origin_chat_id=ctx.chat_id if ctx else "",
                )
            except ValueError as e:
                return f"Error: {e}"
            return f"Pipeline created: {pid} ({len(steps)} steps)"

        return f"Unknown action: {action}"
