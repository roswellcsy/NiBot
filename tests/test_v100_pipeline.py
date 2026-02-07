"""Tests for v1.0.0: Pipeline orchestration."""
from __future__ import annotations

import asyncio

import pytest

from nibot.tools.pipeline_tool import PipelineEngine, PipelineStep, PipelineTool
from nibot.types import ToolContext


class FakeSubagentManager:
    """Fake SubagentManager that completes tasks immediately."""

    def __init__(self, fail_types: set[str] | None = None) -> None:
        self._counter = 0
        self._fail_types = fail_types or set()
        self.spawned: list[dict] = []

    async def spawn(self, task="", label="", origin_channel="", origin_chat_id="",
                    agent_type="", agent_config=None, max_iterations=15,
                    on_complete=None) -> str:
        self._counter += 1
        task_id = f"fake_{self._counter}"
        self.spawned.append({"task_id": task_id, "agent_type": agent_type, "task": task})

        # Simulate async completion
        async def complete():
            await asyncio.sleep(0.05)
            if agent_type in self._fail_types:
                # For failed steps, don't call on_complete to simulate error
                pass
            elif on_complete:
                await on_complete(task_id, f"result from {agent_type}")

        asyncio.create_task(complete())
        return task_id


class TestPipelineEngine:
    """PipelineEngine DAG scheduling."""

    @pytest.mark.asyncio
    async def test_linear_pipeline(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {"coder": None, "researcher": None})

        steps = [
            PipelineStep(id="A", agent_type="researcher", task="research"),
            PipelineStep(id="B", agent_type="coder", task="code", depends_on=["A"]),
            PipelineStep(id="C", agent_type="coder", task="test", depends_on=["B"]),
        ]
        pid = await engine.create(steps)

        # Wait for pipeline to complete
        for _ in range(20):
            await asyncio.sleep(0.1)
            status = engine.get_status(pid)
            if status and status["status"] != "running":
                break

        status = engine.get_status(pid)
        assert status is not None
        assert status["status"] == "completed"
        assert len(mgr.spawned) == 3

    @pytest.mark.asyncio
    async def test_parallel_steps(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {"coder": None, "researcher": None})

        steps = [
            PipelineStep(id="A", agent_type="researcher", task="research"),
            PipelineStep(id="B", agent_type="coder", task="code1"),  # no deps = parallel with A
            PipelineStep(id="C", agent_type="coder", task="final", depends_on=["A", "B"]),
        ]
        pid = await engine.create(steps)

        for _ in range(20):
            await asyncio.sleep(0.1)
            status = engine.get_status(pid)
            if status and status["status"] != "running":
                break

        status = engine.get_status(pid)
        assert status is not None
        assert status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_cancel_pipeline(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {"coder": None})

        steps = [
            PipelineStep(id="A", agent_type="coder", task="long task"),
        ]
        pid = await engine.create(steps)
        assert engine.cancel(pid)

        status = engine.get_status(pid)
        assert status["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_list_pipelines(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {"coder": None})

        await engine.create([PipelineStep(id="A", agent_type="coder", task="t1")])
        await engine.create([PipelineStep(id="A", agent_type="coder", task="t2")])

        pipelines = engine.list_pipelines()
        assert len(pipelines) == 2

    @pytest.mark.asyncio
    async def test_get_status_not_found(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {})
        assert engine.get_status("nonexistent") is None


class TestPipelineTool:
    """PipelineTool wraps PipelineEngine for conversation use."""

    @pytest.mark.asyncio
    async def test_create_action(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {"coder": None})
        tool = PipelineTool(engine)

        result = await tool.execute(
            action="create",
            steps=[{"id": "A", "agent_type": "coder", "task": "do stuff"}],
        )
        assert "created" in result.lower()

    @pytest.mark.asyncio
    async def test_list_action_empty(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {})
        tool = PipelineTool(engine)

        result = await tool.execute(action="list")
        assert "no pipeline" in result.lower()

    @pytest.mark.asyncio
    async def test_status_not_found(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {})
        tool = PipelineTool(engine)

        result = await tool.execute(action="status", pipeline_id="nonexist")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_cancel_action(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {"coder": None})
        tool = PipelineTool(engine)

        create_result = await tool.execute(
            action="create",
            steps=[{"id": "A", "agent_type": "coder", "task": "task"}],
        )
        pid = create_result.split(":")[1].strip().split()[0]

        cancel_result = await tool.execute(action="cancel", pipeline_id=pid)
        assert "cancelled" in cancel_result.lower()

    @pytest.mark.asyncio
    async def test_create_needs_steps(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {})
        tool = PipelineTool(engine)

        result = await tool.execute(action="create")
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_receive_context(self) -> None:
        mgr = FakeSubagentManager()
        engine = PipelineEngine(mgr, {"coder": None})
        tool = PipelineTool(engine)
        ctx = ToolContext(channel="tg", chat_id="123")
        tool.receive_context(ctx)

        result = await tool.execute(
            action="create",
            steps=[{"id": "A", "agent_type": "coder", "task": "test"}],
        )
        assert "created" in result.lower()
