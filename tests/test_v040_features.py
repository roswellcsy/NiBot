"""Tests for v0.4.0 features: gateway architecture, autonomous evolution."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.bus import MessageBus
from nibot.config import (
    DEFAULT_AGENT_TYPES,
    AgentTypeConfig,
    NiBotConfig,
    ScheduledJob,
)
from nibot.registry import Tool, ToolRegistry
from nibot.scheduler import SchedulerManager
from nibot.subagent import SUBAGENT_TOOL_DENY, SubagentManager
from nibot.tools.admin_tools import ConfigTool, ScheduleTool, SkillTool, _CONFIG_SAFE_FIELDS
from nibot.tools.spawn_tool import DelegateTool
from nibot.types import Envelope, LLMResponse, ToolContext


# ---- Helpers ----

class MinimalProvider:
    async def chat(self, messages=None, tools=None, model="", max_tokens=4096, temperature=0.7):
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


# ---- Phase 1: Config + Registry ----

class TestAgentTypeConfig:
    """DEFAULT_AGENT_TYPES and AgentTypeConfig correctness."""

    def test_default_types_exist(self) -> None:
        assert "coder" in DEFAULT_AGENT_TYPES
        assert "researcher" in DEFAULT_AGENT_TYPES
        assert "system" in DEFAULT_AGENT_TYPES
        assert "evolution" in DEFAULT_AGENT_TYPES

    def test_coder_fields_correct(self) -> None:
        coder = DEFAULT_AGENT_TYPES["coder"]
        assert "read_file" in coder.tools
        assert "exec" in coder.tools
        assert "git" in coder.tools
        assert coder.max_iterations == 25
        assert coder.workspace_mode == "worktree"

    def test_evolution_has_skill_tool_and_prompt(self) -> None:
        evo = DEFAULT_AGENT_TYPES["evolution"]
        assert "skill" in evo.tools
        assert evo.system_prompt != ""
        assert evo.max_iterations == 30


class TestRegistryAllowList:
    """Registry get_definitions() allow-list filtering."""

    def test_allow_filters_to_whitelist(self) -> None:
        reg = ToolRegistry()
        reg.register(FakeTool("alpha"))
        reg.register(FakeTool("beta"))
        reg.register(FakeTool("gamma"))
        defs = reg.get_definitions(allow=["alpha", "gamma"])
        names = {d["function"]["name"] for d in defs}
        assert names == {"alpha", "gamma"}

    def test_empty_allow_returns_nothing(self) -> None:
        reg = ToolRegistry()
        reg.register(FakeTool("alpha"))
        defs = reg.get_definitions(allow=[])
        assert defs == []

    def test_allow_takes_priority_over_deny(self) -> None:
        reg = ToolRegistry()
        reg.register(FakeTool("alpha"))
        reg.register(FakeTool("beta"))
        # When allow is set, deny is ignored
        defs = reg.get_definitions(allow=["alpha"], deny=["alpha"])
        names = {d["function"]["name"] for d in defs}
        assert names == {"alpha"}


# ---- Phase 2: DelegateTool + TypedSubagent ----

class TestDelegateToolV040:
    """DelegateTool dispatches to typed agents."""

    @pytest.mark.asyncio
    async def test_known_type_delegates(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        mock_mgr.spawn = AsyncMock(return_value="task123")
        agents = {"coder": AgentTypeConfig(tools=["exec"])}
        tool = DelegateTool(mock_mgr, agents)
        ctx = ToolContext(channel="tg", chat_id="1", session_key="tg:1")
        result = await tool.execute(agent_type="coder", task="build it", _tool_ctx=ctx)
        assert "task123" in result
        assert "coder" in result

    @pytest.mark.asyncio
    async def test_unknown_type_returns_error(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        agents = {"coder": AgentTypeConfig(tools=["exec"])}
        tool = DelegateTool(mock_mgr, agents)
        result = await tool.execute(agent_type="nonexistent", task="x")
        assert "Unknown agent type" in result
        assert "coder" in result  # lists available types

    def test_description_lists_available_types(self) -> None:
        mock_mgr = MagicMock(spec=SubagentManager)
        agents = {"coder": AgentTypeConfig(), "researcher": AgentTypeConfig()}
        tool = DelegateTool(mock_mgr, agents)
        assert "coder" in tool.description
        assert "researcher" in tool.description


class TestTypedSubagent:
    """SubagentManager uses agent_config for tool filtering and model override."""

    @pytest.mark.asyncio
    async def test_whitelist_mode_with_agent_config(self) -> None:
        reg = ToolRegistry()
        reg.register(FakeTool("read_file"))
        reg.register(FakeTool("exec"))
        reg.register(FakeTool("delegate"))

        provider = MinimalProvider()
        bus = MessageBus()
        mgr = SubagentManager(provider, reg, bus)

        config = AgentTypeConfig(tools=["read_file"], max_iterations=1)
        await mgr.spawn(
            task="test", label="t", origin_channel="ch", origin_chat_id="1",
            agent_type="coder", agent_config=config,
        )
        await asyncio.sleep(0.1)
        # The subagent ran and produced output on the bus
        assert bus._outbound.qsize() > 0

    @pytest.mark.asyncio
    async def test_empty_tools_list_grants_no_tools(self) -> None:
        """P1 fix: agent_config with tools=[] must NOT fall back to deny-list."""
        called_tools: list[list | None] = []

        class TrackingProvider:
            async def chat(self, messages=None, tools=None, model="", **kwargs):
                called_tools.append(tools)
                return LLMResponse(content="done", finish_reason="stop")

        reg = ToolRegistry()
        reg.register(FakeTool("read_file"))
        reg.register(FakeTool("exec"))
        bus = MessageBus()
        mgr = SubagentManager(TrackingProvider(), reg, bus)

        config = AgentTypeConfig(tools=[], max_iterations=1)
        await mgr.spawn(
            task="test", label="t", origin_channel="ch", origin_chat_id="1",
            agent_type="restricted", agent_config=config,
        )
        await asyncio.sleep(0.1)
        # Provider should have been called with empty tool list (no tools)
        assert called_tools[0] is None or called_tools[0] == []

    @pytest.mark.asyncio
    async def test_no_config_falls_back_to_deny_list(self) -> None:
        """Without agent_config, subagent uses SUBAGENT_TOOL_DENY blacklist."""
        reg = ToolRegistry()
        reg.register(FakeTool("read_file"))
        reg.register(FakeTool("delegate"))

        provider = MinimalProvider()
        bus = MessageBus()
        mgr = SubagentManager(provider, reg, bus)

        # No agent_config: should use deny list
        await mgr.spawn(
            task="test", label="t", origin_channel="ch", origin_chat_id="1",
            max_iterations=1,
        )
        await asyncio.sleep(0.1)
        assert bus._outbound.qsize() > 0

    @pytest.mark.asyncio
    async def test_model_override_passed_to_provider(self) -> None:
        """agent_config.model is forwarded to provider.chat(model=...)."""
        called_models: list[str] = []

        class TrackingProvider:
            async def chat(self, messages=None, tools=None, model="", **kwargs):
                called_models.append(model)
                return LLMResponse(content="done", finish_reason="stop")

        reg = ToolRegistry()
        bus = MessageBus()
        mgr = SubagentManager(TrackingProvider(), reg, bus)

        config = AgentTypeConfig(tools=[], model="custom/model-v2", max_iterations=1)
        await mgr.spawn(
            task="test", label="t", origin_channel="ch", origin_chat_id="1",
            agent_type="test", agent_config=config,
        )
        await asyncio.sleep(0.1)
        assert "custom/model-v2" in called_models


# ---- Phase 3: Gateway Mode ----

class TestGatewayMode:
    """AgentLoop gateway_tools limits visible tools."""

    def test_gateway_tools_config_default_empty(self) -> None:
        cfg = NiBotConfig()
        assert cfg.agent.gateway_tools == []

    def test_gateway_tools_settable(self) -> None:
        cfg = NiBotConfig(agent={"gateway_tools": ["delegate", "message"]})
        assert cfg.agent.gateway_tools == ["delegate", "message"]


# ---- Phase 4: Scheduler ----

class TestSchedulerManager:
    """SchedulerManager add/remove/list and cron trigger."""

    def test_add_and_list(self) -> None:
        bus = MessageBus()
        mgr = SchedulerManager(bus, [])
        job = ScheduledJob(id="j1", cron="0 9 * * *", prompt="hello")
        mgr.add(job)
        assert len(mgr.list_jobs()) == 1
        assert mgr.list_jobs()[0].id == "j1"

    def test_remove(self) -> None:
        bus = MessageBus()
        job = ScheduledJob(id="j1", cron="0 9 * * *", prompt="hello")
        mgr = SchedulerManager(bus, [job])
        assert mgr.remove("j1") is True
        assert mgr.remove("j1") is False
        assert len(mgr.list_jobs()) == 0

    def test_init_with_jobs(self) -> None:
        bus = MessageBus()
        jobs = [
            ScheduledJob(id="a", cron="0 9 * * *", prompt="morning"),
            ScheduledJob(id="b", cron="0 22 * * *", prompt="evening"),
        ]
        mgr = SchedulerManager(bus, jobs)
        assert len(mgr.list_jobs()) == 2

    @pytest.mark.asyncio
    async def test_fire_publishes_to_inbound(self) -> None:
        bus = MessageBus()
        job = ScheduledJob(id="test", cron="* * * * *", prompt="do stuff", channel="telegram", chat_id="42")
        mgr = SchedulerManager(bus, [job])
        await mgr._fire(job)
        msg = await bus._inbound.get()
        assert msg.channel == "telegram"
        assert msg.chat_id == "42"
        assert msg.content == "do stuff"
        assert msg.sender_id == "scheduler"
        assert msg.metadata["scheduled"] is True


# ---- Phase 5: Admin Tools ----

class TestConfigTool:
    """ConfigTool get/set/list with security boundary."""

    @pytest.mark.asyncio
    async def test_get_returns_value(self, tmp_path: Path) -> None:
        cfg = NiBotConfig()
        tool = ConfigTool(cfg, tmp_path)
        result = await tool.execute(action="get", key="agent.model")
        assert "agent.model" in result
        assert cfg.agent.model in result

    @pytest.mark.asyncio
    async def test_set_changes_config(self, tmp_path: Path) -> None:
        cfg = NiBotConfig()
        tool = ConfigTool(cfg, tmp_path)
        result = await tool.execute(action="set", key="agent.temperature", value="0.5")
        assert "0.5" in result
        assert cfg.agent.temperature == 0.5

    @pytest.mark.asyncio
    async def test_set_refuses_sensitive_field(self, tmp_path: Path) -> None:
        cfg = NiBotConfig()
        tool = ConfigTool(cfg, tmp_path)
        result = await tool.execute(action="set", key="providers.openai.api_key", value="sk-secret")
        assert "Refused" in result

    @pytest.mark.asyncio
    async def test_shared_reference_visible(self, tmp_path: Path) -> None:
        """ConfigTool modifies the live config object -- DelegateTool sees changes immediately."""
        cfg = NiBotConfig()
        tool = ConfigTool(cfg, tmp_path)
        await tool.execute(action="set", key="agent.model", value="new/model")
        # The same config object reflects the change
        assert cfg.agent.model == "new/model"


class TestScheduleTool:
    """ScheduleTool add/remove/list with persistence."""

    @pytest.mark.asyncio
    async def test_add_job(self, tmp_path: Path) -> None:
        bus = MessageBus()
        cfg = NiBotConfig()
        scheduler = SchedulerManager(bus, [])
        tool = ScheduleTool(scheduler, cfg, tmp_path)
        result = await tool.execute(action="add", id="morning", cron="0 9 * * *", prompt="Good morning")
        assert "Added" in result
        assert len(scheduler.list_jobs()) == 1

    @pytest.mark.asyncio
    async def test_list_jobs(self, tmp_path: Path) -> None:
        bus = MessageBus()
        cfg = NiBotConfig()
        job = ScheduledJob(id="j1", cron="0 9 * * *", prompt="hello world test")
        scheduler = SchedulerManager(bus, [job])
        tool = ScheduleTool(scheduler, cfg, tmp_path)
        result = await tool.execute(action="list")
        assert "j1" in result
        assert "0 9 * * *" in result

    @pytest.mark.asyncio
    async def test_persist_round_trip(self, tmp_path: Path) -> None:
        """Adding a job updates config.schedules (persistence target)."""
        bus = MessageBus()
        cfg = NiBotConfig()
        # workspace is tmp_path, so config.json lands at tmp_path.parent / "config.json"
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        scheduler = SchedulerManager(bus, [])
        tool = ScheduleTool(scheduler, cfg, workspace)
        await tool.execute(action="add", id="daily", cron="0 8 * * *", prompt="wake up")
        # config.schedules should reflect the new job
        assert len(cfg.schedules) == 1
        assert cfg.schedules[0].id == "daily"

    @pytest.mark.asyncio
    async def test_persist_writes_to_disk(self, tmp_path: Path) -> None:
        """P2 fix: ScheduleTool must write schedules to config.json on disk."""
        bus = MessageBus()
        cfg = NiBotConfig()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        config_file = tmp_path / "config.json"
        scheduler = SchedulerManager(bus, [])
        tool = ScheduleTool(scheduler, cfg, workspace)
        await tool.execute(action="add", id="nightly", cron="0 2 * * *", prompt="evolve")
        # config.json should exist and contain the job
        assert config_file.exists()
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert len(data["schedules"]) == 1
        assert data["schedules"][0]["id"] == "nightly"


class TestSkillTool:
    """SkillTool list and reload."""

    @pytest.mark.asyncio
    async def test_list_empty(self) -> None:
        skills = MagicMock()
        skills.get_all.return_value = []
        tool = SkillTool(skills)
        result = await tool.execute(action="list")
        assert "No skills" in result

    @pytest.mark.asyncio
    async def test_reload_calls_reload(self) -> None:
        skills = MagicMock()
        skills.reload = MagicMock()
        skills.get_all.return_value = []
        tool = SkillTool(skills)
        result = await tool.execute(action="reload")
        skills.reload.assert_called_once()
        assert "reloaded" in result.lower()


# ---- Phase 6: Bus ----

class TestBusNoSubscriber:
    """Messages to unknown channels log warning but don't crash."""

    @pytest.mark.asyncio
    async def test_no_subscriber_message_dropped_silently(self) -> None:
        bus = MessageBus()
        # Put a message to an unregistered channel on outbound
        await bus.publish_outbound(Envelope(
            channel="nonexistent", chat_id="1", sender_id="bot", content="hello",
        ))
        # Start dispatch, let it process
        task = asyncio.create_task(bus.dispatch_outbound())
        await asyncio.sleep(0.1)
        bus.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # No crash -- bus still functional
        assert bus._outbound.qsize() == 0
