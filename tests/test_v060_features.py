"""Tests for v0.6.0 features: merge, thoughts injection, task monitoring, analyze, skill create, evolution schedule."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.bus import MessageBus
from nibot.config import (
    AgentTypeConfig,
    NiBotConfig,
    ProviderConfig,
    ProvidersConfig,
    ScheduledJob,
    default_evolution_schedule,
)
from nibot.memory import MemoryStore
from nibot.registry import Tool, ToolRegistry
from nibot.session import Session, SessionManager
from nibot.skills import SkillsLoader
from nibot.subagent import SubagentManager, TaskInfo, _WriteThoughtTool
from nibot.tools.analyze_tool import AnalyzeTool
from nibot.tools.git_tool import GitTool
from nibot.tools.spawn_tool import DelegateTool
from nibot.types import LLMResponse
from nibot.worktree import WorktreeManager


# ---- Helpers ----

class FakeProvider:
    async def chat(self, messages=None, tools=None, model="", **kwargs):
        return LLMResponse(content="done", finish_reason="stop")


class FakeTool(Tool):
    def __init__(self, tool_name: str) -> None:
        self._name = tool_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


# ---- A1: WorktreeManager.merge() + branch_info() ----

class TestWorktreeMerge:
    @pytest.mark.asyncio
    async def test_merge_success(self, tmp_path: Path) -> None:
        mgr = WorktreeManager(tmp_path)
        await mgr.ensure_repo()
        wt_path = await mgr.create("m1")
        (wt_path / "new.txt").write_text("data")
        await mgr.commit("m1", "add new.txt")
        result = await mgr.merge("m1")
        assert "Merge" in result or "merge" in result.lower()
        await mgr.remove("m1")

    @pytest.mark.asyncio
    async def test_merge_no_base_branch(self, tmp_path: Path) -> None:
        mgr = WorktreeManager(tmp_path)
        await mgr.ensure_repo()
        wt_path = await mgr.create("m2")
        (wt_path / "x.txt").write_text("x")
        await mgr.commit("m2", "add x")
        # Merge with nonexistent base -- should fallback to HEAD
        result = await mgr.merge("m2", base_branch="nonexistent")
        assert "failed" not in result.lower() or "Merge" in result
        await mgr.remove("m2")

    @pytest.mark.asyncio
    async def test_branch_info(self, tmp_path: Path) -> None:
        mgr = WorktreeManager(tmp_path)
        await mgr.ensure_repo()
        wt_path = await mgr.create("bi1")
        (wt_path / "f.txt").write_text("content")
        await mgr.commit("bi1", "add f")
        info = await mgr.branch_info("bi1")
        assert info["branch"] == "task/bi1"
        assert int(info["commits"]) >= 1
        assert info["last_message"] == "add f"
        await mgr.remove("bi1")


# ---- A2: GitTool merge/branch_info actions ----

class TestGitToolMerge:
    @pytest.mark.asyncio
    async def test_merge_action(self, tmp_path: Path) -> None:
        wt_mgr = WorktreeManager(tmp_path)
        await wt_mgr.ensure_repo()
        wt_path = await wt_mgr.create("gt1")
        (wt_path / "a.txt").write_text("a")
        await wt_mgr.commit("gt1", "add a")
        tool = GitTool(wt_mgr)
        result = await tool.execute(action="merge", task_id="gt1")
        assert "Merge" in result or "merge" in result.lower()
        await wt_mgr.remove("gt1")

    @pytest.mark.asyncio
    async def test_branch_info_action(self, tmp_path: Path) -> None:
        wt_mgr = WorktreeManager(tmp_path)
        await wt_mgr.ensure_repo()
        wt_path = await wt_mgr.create("gt2")
        (wt_path / "b.txt").write_text("b")
        await wt_mgr.commit("gt2", "add b")
        tool = GitTool(wt_mgr)
        result = await tool.execute(action="branch_info", task_id="gt2")
        assert "task/gt2" in result
        assert "add b" in result
        await wt_mgr.remove("gt2")

    @pytest.mark.asyncio
    async def test_merge_no_task_id(self, tmp_path: Path) -> None:
        tool = GitTool(WorktreeManager(tmp_path))
        result = await tool.execute(action="merge")
        assert "Error" in result


# ---- A3-A4: Thoughts injection + WriteThoughtTool ----

class TestThoughtsInjection:
    def test_read_thoughts_in_subagent(self, tmp_path: Path) -> None:
        thoughts = tmp_path / "thoughts"
        thoughts.mkdir()
        (thoughts / "plan.md").write_text("# Plan\nBuild feature X")
        reg = ToolRegistry()
        bus = MessageBus()
        mgr = SubagentManager(FakeProvider(), reg, bus, workspace=tmp_path)
        content = mgr._read_thoughts()
        assert "Plan" in content
        assert "Build feature X" in content

    def test_no_thoughts_returns_empty(self, tmp_path: Path) -> None:
        reg = ToolRegistry()
        bus = MessageBus()
        mgr = SubagentManager(FakeProvider(), reg, bus, workspace=tmp_path)
        assert mgr._read_thoughts() == ""

    def test_no_workspace_returns_empty(self) -> None:
        reg = ToolRegistry()
        bus = MessageBus()
        mgr = SubagentManager(FakeProvider(), reg, bus)
        assert mgr._read_thoughts() == ""


class TestWriteThoughtTool:
    @pytest.mark.asyncio
    async def test_write_thought(self, tmp_path: Path) -> None:
        tool = _WriteThoughtTool(tmp_path)
        result = await tool.execute(filename="test_note", content="# Test\nHello")
        assert "written" in result.lower()
        assert (tmp_path / "thoughts" / "test_note.md").exists()
        assert "Hello" in (tmp_path / "thoughts" / "test_note.md").read_text()

    @pytest.mark.asyncio
    async def test_write_thought_sanitizes_path(self, tmp_path: Path) -> None:
        tool = _WriteThoughtTool(tmp_path)
        await tool.execute(filename="../../evil", content="nope")
        assert (tmp_path / "thoughts" / ".._.._evil.md").exists()


# ---- A5: TaskInfo + get_task_info/list_tasks ----

class TestTaskInfo:
    @pytest.mark.asyncio
    async def test_spawn_creates_task_info(self) -> None:
        reg = ToolRegistry()
        bus = MessageBus()
        mgr = SubagentManager(FakeProvider(), reg, bus)
        config = AgentTypeConfig(tools=[], max_iterations=1)
        task_id = await mgr.spawn(
            task="test", label="lbl", origin_channel="ch", origin_chat_id="1",
            agent_type="test", agent_config=config,
        )
        info = mgr.get_task_info(task_id)
        assert info is not None
        assert info.task_id == task_id
        assert info.agent_type == "test"
        assert info.label == "lbl"
        await asyncio.sleep(0.2)
        info2 = mgr.get_task_info(task_id)
        assert info2 is not None
        assert info2.status in ("completed", "error")

    @pytest.mark.asyncio
    async def test_list_tasks(self) -> None:
        reg = ToolRegistry()
        bus = MessageBus()
        mgr = SubagentManager(FakeProvider(), reg, bus)
        config = AgentTypeConfig(tools=[], max_iterations=1)
        await mgr.spawn(
            task="t1", label="l1", origin_channel="ch", origin_chat_id="1",
            agent_type="a", agent_config=config,
        )
        await mgr.spawn(
            task="t2", label="l2", origin_channel="ch", origin_chat_id="1",
            agent_type="b", agent_config=config,
        )
        await asyncio.sleep(0.2)
        tasks = mgr.list_tasks()
        assert len(tasks) >= 2

    def test_get_nonexistent(self) -> None:
        reg = ToolRegistry()
        bus = MessageBus()
        mgr = SubagentManager(FakeProvider(), reg, bus)
        assert mgr.get_task_info("nope") is None


# ---- A6: DelegateTool query/list actions ----

class TestDelegateActions:
    @pytest.mark.asyncio
    async def test_list_action(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        mock_mgr.list_tasks.return_value = [
            TaskInfo(task_id="abc", agent_type="coder", label="fix", status="completed",
                     finished_at=datetime.now()),
        ]
        agents = {"coder": AgentTypeConfig(tools=["exec"])}
        tool = DelegateTool(mock_mgr, agents)
        result = await tool.execute(action="list")
        assert "abc" in result
        assert "coder" in result

    @pytest.mark.asyncio
    async def test_query_action(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        mock_mgr.get_task_info.return_value = TaskInfo(
            task_id="xyz", agent_type="researcher", label="search", status="completed",
            result="Found 3 results",
        )
        agents = {"researcher": AgentTypeConfig(tools=[])}
        tool = DelegateTool(mock_mgr, agents)
        result = await tool.execute(action="query", task_id="xyz")
        assert "xyz" in result
        assert "Found 3 results" in result

    @pytest.mark.asyncio
    async def test_query_not_found(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        mock_mgr.get_task_info.return_value = None
        tool = DelegateTool(mock_mgr, {"coder": AgentTypeConfig(tools=[])})
        result = await tool.execute(action="query", task_id="nope")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_spawn_action_default(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        mock_mgr.spawn = AsyncMock(return_value="t123")
        agents = {"coder": AgentTypeConfig(tools=["exec"], workspace_mode="worktree")}
        tool = DelegateTool(mock_mgr, agents)
        from nibot.types import ToolContext
        ctx = ToolContext(channel="tg", chat_id="1", session_key="tg:1")
        result = await tool.execute(action="spawn", agent_type="coder", task="build", _tool_ctx=ctx)
        assert "t123" in result
        assert "worktree" in result.lower()

    @pytest.mark.asyncio
    async def test_list_empty(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        mock_mgr.list_tasks.return_value = []
        tool = DelegateTool(mock_mgr, {})
        result = await tool.execute(action="list")
        assert "No tasks" in result


# ---- B1: SessionManager.query_recent() ----

class TestSessionQuery:
    def test_query_recent(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        s1 = mgr.get_or_create("ch:1")
        s1.add_message("user", "hello")
        s1.add_message("assistant", "hi")
        mgr.save(s1)
        s2 = mgr.get_or_create("ch:2")
        s2.add_message("user", "help")
        s2.add_message("tool", "Error: not found")
        mgr.save(s2)
        results = mgr.query_recent(limit=10)
        assert len(results) == 2
        keys = {r["key"] for r in results}
        assert "ch:1" in keys
        assert "ch:2" in keys

    def test_query_recent_empty(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        assert mgr.query_recent() == []

    def test_get_session_messages(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        s = mgr.get_or_create("ch:1")
        s.add_message("user", "msg1")
        s.add_message("assistant", "msg2")
        mgr.save(s)
        msgs = mgr.get_session_messages("ch:1", limit=5)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "msg1"

    def test_get_session_messages_not_found(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        assert mgr.get_session_messages("nope") == []


# ---- B2: AnalyzeTool ----

class TestAnalyzeTool:
    @pytest.mark.asyncio
    async def test_summary(self, tmp_path: Path) -> None:
        sessions = SessionManager(tmp_path)
        s = sessions.get_or_create("ch:1")
        s.add_message("user", "hello")
        s.add_message("assistant", "hi")
        sessions.save(s)
        tool = AnalyzeTool(sessions)
        result = await tool.execute(action="summary")
        assert "Sessions:" in result
        assert "ch:1" in result

    @pytest.mark.asyncio
    async def test_errors(self, tmp_path: Path) -> None:
        sessions = SessionManager(tmp_path)
        s = sessions.get_or_create("ch:1")
        s.add_message("user", "do something")
        s.add_message("tool", "Error: file not found")
        sessions.save(s)
        tool = AnalyzeTool(sessions)
        result = await tool.execute(action="errors")
        assert "Error: file not found" in result

    @pytest.mark.asyncio
    async def test_errors_none(self, tmp_path: Path) -> None:
        sessions = SessionManager(tmp_path)
        s = sessions.get_or_create("ch:1")
        s.add_message("user", "hi")
        s.add_message("assistant", "hello")
        sessions.save(s)
        tool = AnalyzeTool(sessions)
        result = await tool.execute(action="errors")
        assert "No errors" in result

    @pytest.mark.asyncio
    async def test_session_detail(self, tmp_path: Path) -> None:
        sessions = SessionManager(tmp_path)
        s = sessions.get_or_create("ch:1")
        s.add_message("user", "what is 2+2?")
        s.add_message("assistant", "4")
        sessions.save(s)
        tool = AnalyzeTool(sessions)
        result = await tool.execute(action="session_detail", session_key="ch:1")
        assert "what is 2+2?" in result

    @pytest.mark.asyncio
    async def test_session_detail_not_found(self, tmp_path: Path) -> None:
        sessions = SessionManager(tmp_path)
        tool = AnalyzeTool(sessions)
        result = await tool.execute(action="session_detail", session_key="nope")
        assert "not found" in result.lower()


# ---- B3: SkillTool create action ----

class TestSkillCreate:
    @pytest.mark.asyncio
    async def test_create_skill(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        loader = SkillsLoader([skills_dir])
        from nibot.tools.admin_tools import SkillTool
        tool = SkillTool(loader)
        result = await tool.execute(
            action="create", name="greet", description="Greeting skill",
            body="Always say hello first.",
        )
        assert "created" in result.lower()
        assert (skills_dir / "greet" / "SKILL.md").exists()
        content = (skills_dir / "greet" / "SKILL.md").read_text()
        assert "greet" in content
        assert "Always say hello first" in content

    @pytest.mark.asyncio
    async def test_create_skill_missing_fields(self, tmp_path: Path) -> None:
        loader = SkillsLoader([tmp_path / "skills"])
        from nibot.tools.admin_tools import SkillTool
        tool = SkillTool(loader)
        result = await tool.execute(action="create", name="", body="")
        assert "Error" in result


# ---- B4: Evolution schedule + auto_evolution config ----

class TestEvolutionSchedule:
    def test_default_evolution_schedule(self) -> None:
        sched = default_evolution_schedule()
        assert sched.id == "evolution-daily"
        assert sched.cron == "0 3 * * *"
        assert sched.enabled is False
        assert "analyze" in sched.prompt

    def test_auto_evolution_config_default_false(self) -> None:
        from nibot.config import AgentConfig
        cfg = AgentConfig()
        assert cfg.auto_evolution is False

    def test_evolution_agent_has_analyze_tool(self) -> None:
        from nibot.config import DEFAULT_AGENT_TYPES
        evo = DEFAULT_AGENT_TYPES["evolution"]
        assert "analyze" in evo.tools
