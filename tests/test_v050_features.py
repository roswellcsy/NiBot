"""Tests for v0.5.0 features: multi-model coding agent, git worktree, thoughts context."""

from __future__ import annotations

import asyncio
import json
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
)
from nibot.context import ContextBuilder
from nibot.memory import MemoryStore
from nibot.provider_pool import ProviderPool
from nibot.registry import Tool, ToolRegistry
from nibot.skills import SkillsLoader
from nibot.subagent import SubagentManager
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


# ---- ProviderPool Tests ----

class TestProviderPoolBasics:
    """ProviderPool get/has with cache and fallback."""

    def test_get_default_when_empty_name(self) -> None:
        default = FakeProvider()
        pool = ProviderPool(ProvidersConfig(), default)
        assert pool.get("") is default

    def test_get_returns_default_for_unknown(self) -> None:
        default = FakeProvider()
        pool = ProviderPool(ProvidersConfig(), default)
        assert pool.get("nonexistent") is default

    def test_has_returns_false_without_key(self) -> None:
        pool = ProviderPool(ProvidersConfig(), FakeProvider())
        assert pool.has("openai") is False


class TestProviderPoolLazy:
    """ProviderPool lazy creation and caching."""

    def test_cache_reuses_instance(self) -> None:
        cfg = ProvidersConfig(
            extras={"moon": ProviderConfig(api_key="sk-test", api_base="http://test", model="m1")}
        )
        pool = ProviderPool(cfg, FakeProvider())
        with patch("nibot.provider_pool.LiteLLMProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            p1 = pool.get("moon")
            p2 = pool.get("moon")
            assert p1 is p2
            mock_cls.assert_called_once()

    def test_no_api_key_falls_back(self) -> None:
        cfg = ProvidersConfig(
            extras={"nokey": ProviderConfig(api_key="", api_base="http://test")}
        )
        default = FakeProvider()
        pool = ProviderPool(cfg, default)
        assert pool.get("nokey") is default


# ---- ProvidersConfig.extras + get() ----

class TestProvidersConfigExtras:
    """ProvidersConfig extras field and get() method."""

    def test_extras_field_default_empty(self) -> None:
        cfg = ProvidersConfig()
        assert cfg.extras == {}

    def test_get_builtin_provider(self) -> None:
        cfg = ProvidersConfig(openai=ProviderConfig(api_key="sk-oai"))
        pc = cfg.get("openai")
        assert pc is not None
        assert pc.api_key == "sk-oai"

    def test_get_extras_provider(self) -> None:
        cfg = ProvidersConfig(
            extras={"moonshot": ProviderConfig(api_key="sk-moon", api_base="http://moon")}
        )
        pc = cfg.get("moonshot")
        assert pc is not None
        assert pc.api_key == "sk-moon"


# ---- SubagentManager Provider Selection ----

class TestSubagentProviderSelection:
    """SubagentManager picks provider from pool when agent_config.provider is set."""

    @pytest.mark.asyncio
    async def test_provider_field_selects_from_pool(self) -> None:
        called_models: list[str] = []

        class TrackingProvider:
            async def chat(self, messages=None, tools=None, model="", **kwargs):
                called_models.append(model)
                return LLMResponse(content="done", finish_reason="stop")

        pool = MagicMock()
        tracking = TrackingProvider()
        pool.get.return_value = tracking

        reg = ToolRegistry()
        bus = MessageBus()
        mgr = SubagentManager(FakeProvider(), reg, bus, provider_pool=pool)
        config = AgentTypeConfig(tools=[], model="custom-v1", provider="moonshot", max_iterations=1)
        await mgr.spawn(
            task="test", label="t", origin_channel="ch", origin_chat_id="1",
            agent_type="coder", agent_config=config,
        )
        await asyncio.sleep(0.15)
        pool.get.assert_called_with("moonshot")
        assert "custom-v1" in called_models

    @pytest.mark.asyncio
    async def test_empty_provider_uses_default(self) -> None:
        called: list[bool] = []

        class DefaultProvider:
            async def chat(self, messages=None, tools=None, model="", **kwargs):
                called.append(True)
                return LLMResponse(content="done", finish_reason="stop")

        default = DefaultProvider()
        pool = MagicMock()
        reg = ToolRegistry()
        bus = MessageBus()
        mgr = SubagentManager(default, reg, bus, provider_pool=pool)
        config = AgentTypeConfig(tools=[], provider="", max_iterations=1)
        await mgr.spawn(
            task="test", label="t", origin_channel="ch", origin_chat_id="1",
            agent_type="test", agent_config=config,
        )
        await asyncio.sleep(0.15)
        pool.get.assert_not_called()
        assert called

    @pytest.mark.asyncio
    async def test_unknown_provider_falls_back(self) -> None:
        """Pool returns default when provider name unknown."""
        default = FakeProvider()
        pool = MagicMock()
        pool.get.return_value = default  # pool returns default for unknown

        reg = ToolRegistry()
        bus = MessageBus()
        mgr = SubagentManager(default, reg, bus, provider_pool=pool)
        config = AgentTypeConfig(tools=[], provider="unknown_provider", max_iterations=1)
        await mgr.spawn(
            task="test", label="t", origin_channel="ch", origin_chat_id="1",
            agent_type="test", agent_config=config,
        )
        await asyncio.sleep(0.15)
        pool.get.assert_called_with("unknown_provider")


# ---- WorktreeManager ----

class TestWorktreeManager:
    """WorktreeManager ensure_repo, create, remove, diff, commit."""

    @pytest.mark.asyncio
    async def test_ensure_repo_inits(self, tmp_path: Path) -> None:
        mgr = WorktreeManager(tmp_path)
        ok = await mgr.ensure_repo()
        assert ok is True
        assert (tmp_path / ".git").exists()

    @pytest.mark.asyncio
    async def test_create_and_remove(self, tmp_path: Path) -> None:
        mgr = WorktreeManager(tmp_path)
        await mgr.ensure_repo()
        wt_path = await mgr.create("abc123")
        assert wt_path.exists()
        assert "abc123" in str(wt_path)
        ok = await mgr.remove("abc123")
        assert ok is True

    @pytest.mark.asyncio
    async def test_commit_and_diff(self, tmp_path: Path) -> None:
        mgr = WorktreeManager(tmp_path)
        await mgr.ensure_repo()
        wt_path = await mgr.create("xyz789")
        # Write a file in worktree
        (wt_path / "test.txt").write_text("hello world")
        diff = await mgr.diff("xyz789")
        assert diff != ""
        out = await mgr.commit("xyz789", "test commit")
        assert "test commit" in out or "1 file" in out
        await mgr.remove("xyz789")

    @pytest.mark.asyncio
    async def test_list_worktrees(self, tmp_path: Path) -> None:
        mgr = WorktreeManager(tmp_path)
        await mgr.ensure_repo()
        await mgr.create("wt1")
        wts = await mgr.list_worktrees()
        # At least main worktree + wt1
        assert len(wts) >= 2
        await mgr.remove("wt1")


# ---- GitTool ----

class TestGitTool:
    """GitTool action dispatch."""

    @pytest.mark.asyncio
    async def test_worktree_create_action(self, tmp_path: Path) -> None:
        wt_mgr = WorktreeManager(tmp_path)
        tool = GitTool(wt_mgr)
        result = await tool.execute(action="worktree_create", task_id="t1")
        assert "Worktree created" in result
        assert "task/t1" in result
        await wt_mgr.remove("t1")

    @pytest.mark.asyncio
    async def test_worktree_list_action(self, tmp_path: Path) -> None:
        wt_mgr = WorktreeManager(tmp_path)
        await wt_mgr.ensure_repo()
        tool = GitTool(wt_mgr)
        result = await tool.execute(action="worktree_list")
        assert "Worktrees" in result or "No worktrees" in result


# ---- Isolated Registry ----

class TestIsolatedRegistry:
    """SubagentManager._create_isolated_registry scopes file tools."""

    def test_file_tools_point_to_worktree(self, tmp_path: Path) -> None:
        reg = ToolRegistry()
        reg.register(FakeTool("web_search"))
        reg.register(FakeTool("file_read"))
        bus = MessageBus()
        mgr = SubagentManager(FakeProvider(), reg, bus)

        config = AgentTypeConfig(tools=["file_read", "write_file", "exec", "web_search"])
        iso_reg = mgr._create_isolated_registry(tmp_path, config)

        # File tools should be real tool instances, not FakeTool
        assert iso_reg.has("file_read")
        assert iso_reg.has("write_file")
        assert iso_reg.has("exec")
        # web_search inherited from main registry
        assert iso_reg.has("web_search")

    def test_non_file_tools_inherited(self, tmp_path: Path) -> None:
        reg = ToolRegistry()
        reg.register(FakeTool("git"))
        bus = MessageBus()
        mgr = SubagentManager(FakeProvider(), reg, bus)

        config = AgentTypeConfig(tools=["file_read", "git"])
        iso_reg = mgr._create_isolated_registry(tmp_path, config)
        assert iso_reg.has("git")
        assert iso_reg.has("file_read")


# ---- Thoughts Context ----

class TestThoughtsContext:
    """ContextBuilder reads thoughts/ directory into system prompt."""

    def test_thoughts_injected_when_present(self, tmp_path: Path) -> None:
        thoughts = tmp_path / "thoughts"
        thoughts.mkdir()
        (thoughts / "plan.md").write_text("# Build Plan\nStep 1: setup")
        cfg = NiBotConfig()
        memory = MemoryStore(tmp_path / "memory")
        skills = SkillsLoader([tmp_path / "skills"])
        builder = ContextBuilder(config=cfg, memory=memory, skills=skills, workspace=tmp_path)
        prompt = builder._build_system_prompt()
        assert "Shared Context" in prompt
        assert "Build Plan" in prompt

    def test_no_thoughts_dir_no_section(self, tmp_path: Path) -> None:
        cfg = NiBotConfig()
        memory = MemoryStore(tmp_path / "memory")
        skills = SkillsLoader([tmp_path / "skills"])
        builder = ContextBuilder(config=cfg, memory=memory, skills=skills, workspace=tmp_path)
        prompt = builder._build_system_prompt()
        assert "Shared Context" not in prompt


# ---- DelegateTool Worktree Info ----

class TestDelegateWorktreeInfo:
    """DelegateTool returns worktree branch info when workspace_mode=worktree."""

    @pytest.mark.asyncio
    async def test_worktree_mode_shows_branch(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        mock_mgr.spawn = AsyncMock(return_value="abc123")
        agents = {
            "coder": AgentTypeConfig(
                tools=["exec"], workspace_mode="worktree",
            ),
        }
        tool = DelegateTool(mock_mgr, agents)
        from nibot.types import ToolContext
        ctx = ToolContext(channel="tg", chat_id="1", session_key="tg:1")
        result = await tool.execute(agent_type="coder", task="build it", _tool_ctx=ctx)
        assert "task/abc123" in result
        assert "worktree" in result.lower()

    @pytest.mark.asyncio
    async def test_no_worktree_no_branch_info(self) -> None:
        mock_mgr = AsyncMock(spec=SubagentManager)
        mock_mgr.spawn = AsyncMock(return_value="def456")
        agents = {
            "researcher": AgentTypeConfig(tools=["web_search"]),
        }
        tool = DelegateTool(mock_mgr, agents)
        result = await tool.execute(agent_type="researcher", task="search")
        assert "worktree" not in result.lower()
        assert "def456" in result
