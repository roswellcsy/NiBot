"""Subagent manager -- background task execution with tool isolation."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from nibot.bus import MessageBus
from nibot.config import AgentTypeConfig
from nibot.log import logger
from nibot.provider import LLMProvider
from nibot.registry import Tool, ToolRegistry
from nibot.types import Envelope

SUBAGENT_TOOL_DENY = ["message", "delegate"]


@dataclass
class TaskInfo:
    """Metadata for a spawned subagent task."""

    task_id: str
    agent_type: str
    label: str
    status: str = "running"  # running | completed | error
    result: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None


class _WriteThoughtTool(Tool):
    """Write a thought file to workspace/thoughts/ for cross-agent sharing."""

    def __init__(self, workspace: Path) -> None:
        self._thoughts_dir = workspace / "thoughts"

    @property
    def name(self) -> str:
        return "write_thought"

    @property
    def description(self) -> str:
        return "Write a shared thought/note to thoughts/ directory for other agents to read."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename without extension (e.g. 'plan')"},
                "content": {"type": "string", "description": "Markdown content to write"},
            },
            "required": ["filename", "content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        filename = kwargs["filename"].replace("/", "_").replace("\\", "_")
        if not filename.endswith(".md"):
            filename += ".md"
        self._thoughts_dir.mkdir(parents=True, exist_ok=True)
        path = self._thoughts_dir / filename
        path.write_text(kwargs["content"], encoding="utf-8")
        return f"Thought written: {path.name}"


class SubagentManager:
    """Spawn isolated background agents that report results via the bus."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        bus: MessageBus,
        provider_pool: Any | None = None,
        worktree_mgr: Any | None = None,
        workspace: Path | None = None,
        sessions: Any | None = None,
        skills: Any | None = None,
        evolution_log: Any | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.bus = bus
        self.provider_pool = provider_pool
        self.worktree_mgr = worktree_mgr
        self.workspace = workspace
        self._sessions = sessions
        self._skills = skills
        self._evolution_log = evolution_log
        self._max_task_history = 200
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._task_info: dict[str, TaskInfo] = {}

    def _prune_task_info(self) -> None:
        """Remove completed tasks older than 1 hour, keep at most max_task_history."""
        if len(self._task_info) <= self._max_task_history:
            return
        now = datetime.now()
        cutoff = timedelta(hours=1)
        to_remove = [
            tid for tid, info in self._task_info.items()
            if info.status != "running" and info.finished_at and (now - info.finished_at) > cutoff
        ]
        for tid in to_remove:
            del self._task_info[tid]
        # If still over limit, drop oldest completed entries
        if len(self._task_info) > self._max_task_history:
            completed = sorted(
                [(tid, info) for tid, info in self._task_info.items() if info.status != "running"],
                key=lambda x: x[1].created_at,
            )
            excess = len(self._task_info) - self._max_task_history
            for tid, _ in completed[:excess]:
                del self._task_info[tid]

    async def spawn(
        self,
        task: str,
        label: str,
        origin_channel: str,
        origin_chat_id: str,
        agent_type: str = "",
        agent_config: AgentTypeConfig | None = None,
        max_iterations: int = 15,
        on_complete: Any | None = None,
    ) -> str:
        self._prune_task_info()
        task_id = uuid.uuid4().hex[:8]
        self._task_info[task_id] = TaskInfo(
            task_id=task_id, agent_type=agent_type, label=label,
        )
        bg = asyncio.create_task(
            self._run(task_id, task, label, origin_channel, origin_chat_id,
                      agent_type, agent_config, max_iterations, on_complete)
        )
        self._tasks[task_id] = bg
        bg.add_done_callback(lambda t, tid=task_id: self._task_done(t, tid))
        return task_id

    async def _run(
        self,
        task_id: str,
        task: str,
        label: str,
        channel: str,
        chat_id: str,
        agent_type: str,
        agent_config: AgentTypeConfig | None,
        max_iterations: int,
        on_complete: Any | None = None,
    ) -> None:
        # Worktree isolation for coding agents
        wt_path: Path | None = None
        if agent_config and agent_config.workspace_mode == "worktree" and self.worktree_mgr:
            try:
                await self.worktree_mgr.ensure_repo()
                wt_path = await self.worktree_mgr.create(task_id)
            except Exception as e:
                logger.warning(f"Worktree creation failed for {task_id}: {e}")

        # Tool registry: isolated (worktree) or shared
        if wt_path and agent_config is not None:
            tool_registry = self._create_isolated_registry(wt_path, agent_config)
            # Always add write_thought for worktree agents
            if self.workspace and not tool_registry.has("write_thought"):
                tool_registry.register(_WriteThoughtTool(self.workspace))
            allow = list(agent_config.tools or []) + ["write_thought"]
            tool_defs = tool_registry.get_definitions(allow=allow)
        elif agent_config is not None:
            tool_registry = self.registry
            tool_defs = self.registry.get_definitions(allow=agent_config.tools)
        else:
            tool_registry = self.registry
            tool_defs = self.registry.get_definitions(deny=SUBAGENT_TOOL_DENY)

        system_content = f"You are a specialized {agent_type or 'subagent'}. Task ID: {task_id}."
        if agent_config and agent_config.system_prompt:
            system_content += f"\n\n{agent_config.system_prompt}"
        if wt_path:
            system_content += f"\n\nWorking directory (isolated worktree): {wt_path}"
        # Inject evolution context for evolution agents
        if agent_type == "evolution" and self._sessions and self._skills:
            system_content += f"\n\n{build_evolution_context(self._sessions, self._skills, self._evolution_log)}"
        # Inject shared thoughts context
        thoughts = self._read_thoughts()
        if thoughts:
            system_content += f"\n\n## Shared Context (thoughts/)\n{thoughts}"
        system_content += f"\n\nTask: {task}"

        model_override = agent_config.model if agent_config and agent_config.model else ""
        max_iter = agent_config.max_iterations if agent_config else max_iterations

        # Provider selection: fallback_chain -> named provider -> default
        use_fallback = False
        if agent_config and agent_config.fallback_chain and self.provider_pool:
            use_fallback = True
        elif agent_config and agent_config.provider and self.provider_pool:
            provider = self.provider_pool.get(agent_config.provider)
        else:
            provider = self.provider

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": task},
        ]
        timeout = agent_config.timeout_seconds if agent_config else 300
        final = ""

        async def _agent_loop() -> str:
            result_text = ""
            for _ in range(max_iter):
                if use_fallback:
                    resp = await self.provider_pool.chat_with_fallback(
                        messages=messages, tools=tool_defs or None,
                        chain=agent_config.fallback_chain, model=model_override,
                    )
                else:
                    resp = await provider.chat(
                        messages=messages, tools=tool_defs or None, model=model_override,
                    )
                if not resp.has_tool_calls:
                    result_text = resp.content or ""
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
                    result = await tool_registry.execute(tc.name, tc.arguments, call_id=tc.id)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "name": tc.name,
                        "content": result.content,
                    })
            return result_text

        try:
            final = await asyncio.wait_for(_agent_loop(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Subagent {task_id} timed out after {timeout}s")
            final = f"Subagent timed out after {timeout}s"
            if task_id in self._task_info:
                self._task_info[task_id].status = "error"
        except Exception as e:
            logger.error(f"Subagent {task_id} error: {e}")
            final = f"Subagent error: {e}"
            if task_id in self._task_info:
                self._task_info[task_id].status = "error"

        # Auto-commit worktree changes
        worktree_info = ""
        if wt_path and self.worktree_mgr:
            try:
                diff = await self.worktree_mgr.diff(task_id)
                if diff.strip():
                    await self.worktree_mgr.commit(
                        task_id, f"[{agent_type}] {label}: {task[:50]}"
                    )
                    worktree_info = f"\n\n[Worktree: task/{task_id}]\n{diff}"
            except Exception as e:
                logger.warning(f"Worktree commit failed for {task_id}: {e}")

        # Update task info
        if task_id in self._task_info:
            info = self._task_info[task_id]
            if info.status == "running":
                info.status = "completed"
            info.result = (final + worktree_info)[:2000]
            info.finished_at = datetime.now()

        # Fire completion callback
        if on_complete:
            try:
                await on_complete(task_id, self._task_info[task_id].result if task_id in self._task_info else final)
            except Exception as e:
                logger.warning(f"on_complete callback failed for {task_id}: {e}")

        await self.bus.publish_outbound(
            Envelope(
                channel=channel,
                sender_id="subagent",
                chat_id=chat_id,
                content=f"[Subagent '{label}' completed]\nResult: {final}{worktree_info}",
            )
        )

    def _task_done(self, task: asyncio.Task[None], task_id: str) -> None:
        """Cleanup callback for completed subagent tasks."""
        self._tasks.pop(task_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"Subagent {task_id} crashed: {exc!r}")
            if task_id in self._task_info:
                info = self._task_info[task_id]
                info.status = "error"
                info.result = f"Crash: {exc}"[:500]
                info.finished_at = datetime.now()

    def _create_isolated_registry(self, wt_path: Path, config: AgentTypeConfig) -> ToolRegistry:
        """Create a ToolRegistry with file tools scoped to worktree path."""
        from nibot.tools.code_review_tool import CodeReviewTool
        from nibot.tools.exec_tool import ExecTool
        from nibot.tools.file_tools import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
        from nibot.tools.git_tool import GitTool
        from nibot.tools.test_runner_tool import TestRunnerTool

        # Extract task_id from worktree path (last component)
        task_id = wt_path.name

        reg = ToolRegistry()
        tools: list[Tool] = [
            ReadFileTool(wt_path, restrict=True),
            WriteFileTool(wt_path, restrict=True),
            EditFileTool(wt_path, restrict=True),
            ListDirTool(wt_path, restrict=True),
            ExecTool(wt_path),
            CodeReviewTool(wt_path, worktree_mgr=self.worktree_mgr),
            TestRunnerTool(wt_path),
        ]
        # Add isolated GitTool if worktree manager is available
        if self.worktree_mgr:
            tools.append(GitTool(self.worktree_mgr, allowed_task_id=task_id))
        for tool in tools:
            reg.register(tool)
        # Inherit non-file tools from main registry (skip 'git' -- already isolated above)
        for name in (config.tools or []):
            if name not in reg._tools and name in self.registry._tools:
                reg.register(self.registry._tools[name])
        return reg

    def list_active(self) -> list[str]:
        return list(self._tasks.keys())

    def get_task_info(self, task_id: str) -> TaskInfo | None:
        return self._task_info.get(task_id)

    def list_tasks(self, limit: int = 20) -> list[TaskInfo]:
        """Return recent tasks sorted by creation time (newest first)."""
        tasks = sorted(self._task_info.values(), key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    def _read_thoughts(self) -> str:  # noqa: C901
        """Read workspace/thoughts/*.md for injection into subagent system prompt."""
        if not self.workspace:
            return ""
        thoughts_dir = self.workspace / "thoughts"
        if not thoughts_dir.exists():
            return ""
        files = sorted(thoughts_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
        parts: list[str] = []
        budget = 4000
        for f in files[:6]:
            try:
                content = f.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if len(content) > budget:
                content = content[:budget] + "...(truncated)"
            parts.append(f"### {f.stem}\n{content}")
            budget -= len(content)
            if budget <= 0:
                break
        return "\n\n".join(parts)


def build_evolution_context(
    sessions: Any,
    skills: Any,
    evolution_log: Any | None = None,
    limit: int = 20,
) -> str:
    """Compose metrics snapshot + skill inventory for evolution agent system prompt.

    Pure-ish: reads data from sessions/skills but produces a string. No mutations.
    """
    from nibot.metrics import aggregate_metrics, compute_session_metrics

    # Aggregate recent session metrics
    recent = sessions.iter_recent_from_disk(limit=limit)
    recent.sort(key=lambda s: s.updated_at, reverse=True)
    per_session = [compute_session_metrics(s.messages) for s in recent]
    agg = aggregate_metrics(per_session)
    metrics_json = json.dumps(agg.to_dict(), indent=2)

    # Skill inventory
    skill_lines = [
        f"- {s.name}: {s.description} (v{s.version}, by={s.created_by or 'manual'}, always={s.always})"
        for s in skills.get_all()
    ]
    skill_list = "\n".join(skill_lines) or "(no skills)"

    parts = [
        f"## Current System State\n\n### Aggregate Metrics (last {len(recent)} sessions)\n"
        f"```json\n{metrics_json}\n```\n\n### Skill Inventory\n{skill_list}",
    ]

    # Recent evolution decisions
    if evolution_log:
        decision_summary = evolution_log.summary(5)
        parts.append(f"\n\n### Recent Evolution Decisions\n{decision_summary}")

    return "".join(parts)
