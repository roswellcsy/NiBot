"""Admin tools tests -- SkillTool CRUD, ConfigTool, ScheduleTool."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nibot.bus import MessageBus
from nibot.config import NiBotConfig, ScheduledJob
from nibot.scheduler import SchedulerManager
from nibot.skills import SkillsLoader
from nibot.tools.admin_tools import ConfigTool, ScheduleTool, SkillTool


# ---------------------------------------------------------------------------
# SkillTool CRUD
# ---------------------------------------------------------------------------

def _make_skill_tool(tmp_path: Path) -> SkillTool:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    loader = SkillsLoader([skills_dir])
    return SkillTool(skills=loader)


class TestSkillToolCreate:

    @pytest.mark.asyncio
    async def test_create_skill_basic(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        result = await tool.execute(action="create", name="greet", body="Say hello warmly.", description="Greeting skill")
        assert "created" in result.lower()
        md = tmp_path / "skills" / "greet" / "SKILL.md"
        assert md.exists()
        content = md.read_text(encoding="utf-8")
        assert "name: greet" in content
        assert "Say hello warmly." in content

    @pytest.mark.asyncio
    async def test_create_skill_missing_name(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        result = await tool.execute(action="create", name="", body="stuff")
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_create_skill_missing_body(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        result = await tool.execute(action="create", name="test", body="")
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_create_skill_with_executable(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        result = await tool.execute(action="create", name="runner", body="Run things.", executable=True)
        assert "executable" in result.lower()
        run_py = tmp_path / "skills" / "runner" / "run.py"
        assert run_py.exists()

    @pytest.mark.asyncio
    async def test_create_skill_appears_in_list(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        await tool.execute(action="create", name="myskill", body="Do X.", description="X skill")
        result = await tool.execute(action="list")
        assert "myskill" in result


class TestSkillToolUpdate:

    @pytest.mark.asyncio
    async def test_update_increments_version(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        await tool.execute(action="create", name="vskill", body="v1 body", description="desc")
        result = await tool.execute(action="update", name="vskill", body="v2 body")
        assert "v2" in result
        md = tmp_path / "skills" / "vskill" / "SKILL.md"
        content = md.read_text(encoding="utf-8")
        assert "version: 2" in content
        assert "v2 body" in content

    @pytest.mark.asyncio
    async def test_update_nonexistent_skill(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        result = await tool.execute(action="update", name="nope", body="new")
        assert "not found" in result.lower()


class TestSkillToolDeleteDisableEnable:

    @pytest.mark.asyncio
    async def test_delete_skill(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        await tool.execute(action="create", name="todel", body="temp")
        result = await tool.execute(action="delete", name="todel")
        assert "deleted" in result.lower()
        assert not (tmp_path / "skills" / "todel").exists()

    @pytest.mark.asyncio
    async def test_disable_enable_cycle(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        await tool.execute(action="create", name="toggle", body="toggleable")
        # Disable
        result = await tool.execute(action="disable", name="toggle")
        assert "disabled" in result.lower()
        assert (tmp_path / "skills" / "toggle" / "SKILL.md.disabled").exists()
        assert not (tmp_path / "skills" / "toggle" / "SKILL.md").exists()
        # Enable
        result = await tool.execute(action="enable", name="toggle")
        assert "enabled" in result.lower()
        assert (tmp_path / "skills" / "toggle" / "SKILL.md").exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        result = await tool.execute(action="delete", name="ghost")
        assert "not found" in result.lower()


class TestSkillToolGetReload:

    @pytest.mark.asyncio
    async def test_get_skill(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        await tool.execute(action="create", name="info", body="Info body.", description="Info skill")
        result = await tool.execute(action="get", name="info")
        assert "Info body." in result

    @pytest.mark.asyncio
    async def test_get_missing_name(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        result = await tool.execute(action="get", name="")
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_reload(self, tmp_path: Path) -> None:
        tool = _make_skill_tool(tmp_path)
        result = await tool.execute(action="reload")
        assert "reloaded" in result.lower()


# ---------------------------------------------------------------------------
# ConfigTool
# ---------------------------------------------------------------------------

class TestConfigTool:

    def _make(self, tmp_path: Path) -> ConfigTool:
        cfg = NiBotConfig()
        return ConfigTool(config=cfg, workspace=tmp_path)

    @pytest.mark.asyncio
    async def test_list_shows_keys(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(action="list")
        assert "agent.model" in result
        assert "agent.temperature" in result

    @pytest.mark.asyncio
    async def test_get_existing_key(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(action="get", key="agent.temperature")
        assert "temperature" in result

    @pytest.mark.asyncio
    async def test_get_unknown_key(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(action="get", key="agent.nonexistent")
        assert "Unknown key" in result

    @pytest.mark.asyncio
    async def test_set_allowed_key(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(action="set", key="agent.temperature", value="0.5")
        assert "Set" in result
        assert "0.5" in result

    @pytest.mark.asyncio
    async def test_set_disallowed_key_refused(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(action="set", key="agent.bootstrap_files", value="evil")
        assert "Refused" in result

    @pytest.mark.asyncio
    async def test_set_int_type_conversion(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        await tool.execute(action="set", key="agent.max_tokens", value="2048")
        result = await tool.execute(action="get", key="agent.max_tokens")
        assert "2048" in result

    @pytest.mark.asyncio
    async def test_set_list_type_conversion(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(action="set", key="agent.gateway_tools", value="web_search,exec")
        assert "Set" in result

    @pytest.mark.asyncio
    async def test_get_without_key_returns_error(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(action="get")
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# ScheduleTool
# ---------------------------------------------------------------------------

class TestScheduleTool:

    def _make(self, tmp_path: Path) -> ScheduleTool:
        cfg = NiBotConfig()
        bus = MagicMock(spec=MessageBus)
        scheduler = SchedulerManager(bus=bus, jobs=[])
        config_path = tmp_path / "config.json"
        return ScheduleTool(scheduler=scheduler, config=cfg, workspace=tmp_path, config_path=config_path)

    @pytest.mark.asyncio
    async def test_list_empty(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(action="list")
        assert "No scheduled" in result

    @pytest.mark.asyncio
    async def test_add_and_list(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(
            action="add", id="morning", cron="0 9 * * *", prompt="Good morning report",
        )
        assert "Added" in result
        result = await tool.execute(action="list")
        assert "morning" in result
        assert "0 9 * * *" in result

    @pytest.mark.asyncio
    async def test_add_missing_fields(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(action="add", id="", cron="", prompt="")
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_remove_existing(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        await tool.execute(action="add", id="temp", cron="* * * * *", prompt="test")
        result = await tool.execute(action="remove", id="temp")
        assert "Removed" in result

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        result = await tool.execute(action="remove", id="ghost")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_persist_writes_config(self, tmp_path: Path) -> None:
        tool = self._make(tmp_path)
        await tool.execute(action="add", id="persist_test", cron="0 8 * * *", prompt="wake up")
        config_file = tmp_path / "config.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert "schedules" in data
        assert any(s["id"] == "persist_test" for s in data["schedules"])
