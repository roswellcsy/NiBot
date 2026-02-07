"""Tests for v1.0.0: Platform features (marketplace, scaffold, web panel, deploy)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.marketplace import MarketplaceSkill, SkillMarketplace
from nibot.tools.scaffold_tool import ScaffoldTool, TEMPLATES


class TestScaffoldTool:

    @pytest.mark.asyncio
    async def test_list_templates(self) -> None:
        tool = ScaffoldTool(Path(tempfile.mkdtemp()))
        result = await tool.execute(action="list")
        assert "python-lib" in result
        assert "fastapi-app" in result
        assert "nibot-skill" in result

    @pytest.mark.asyncio
    async def test_create_python_lib(self) -> None:
        ws = Path(tempfile.mkdtemp())
        tool = ScaffoldTool(ws)
        result = await tool.execute(
            action="create",
            template="python-lib",
            project_name="mylib",
            project_description="A test library",
        )
        assert "mylib" in result
        assert (ws / "mylib" / "pyproject.toml").exists()
        assert (ws / "mylib" / "mylib" / "__init__.py").exists()
        assert (ws / "mylib" / "tests" / "test_main.py").exists()

        # Check variable substitution
        content = (ws / "mylib" / "pyproject.toml").read_text()
        assert "mylib" in content
        assert "A test library" in content

    @pytest.mark.asyncio
    async def test_create_fastapi_app(self) -> None:
        ws = Path(tempfile.mkdtemp())
        tool = ScaffoldTool(ws)
        result = await tool.execute(
            action="create",
            template="fastapi-app",
            project_name="myapi",
        )
        assert "myapi" in result
        assert (ws / "myapi" / "myapi" / "app.py").exists()

    @pytest.mark.asyncio
    async def test_create_nibot_skill(self) -> None:
        ws = Path(tempfile.mkdtemp())
        tool = ScaffoldTool(ws)
        result = await tool.execute(
            action="create",
            template="nibot-skill",
            project_name="my-skill",
            project_description="A custom skill",
        )
        assert "my-skill" in result
        assert (ws / "my-skill" / "SKILL.md").exists()

    @pytest.mark.asyncio
    async def test_create_unknown_template(self) -> None:
        tool = ScaffoldTool(Path(tempfile.mkdtemp()))
        result = await tool.execute(action="create", template="unknown", project_name="test")
        assert "unknown" in result.lower()

    @pytest.mark.asyncio
    async def test_create_no_name(self) -> None:
        tool = ScaffoldTool(Path(tempfile.mkdtemp()))
        result = await tool.execute(action="create", template="python-lib")
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_create_existing_dir(self) -> None:
        ws = Path(tempfile.mkdtemp())
        (ws / "existing").mkdir()
        tool = ScaffoldTool(ws)
        result = await tool.execute(action="create", template="python-lib", project_name="existing")
        assert "exists" in result.lower()


class TestSkillMarketplace:

    def test_marketplace_skill_dataclass(self) -> None:
        skill = MarketplaceSkill(name="test", description="desc", url="https://github.com/test/test")
        assert skill.name == "test"

    @pytest.mark.asyncio
    async def test_install_creates_files(self) -> None:
        """Test install with mocked HTTP."""
        skills_dir = Path(tempfile.mkdtemp())
        mp = SkillMarketplace(skills_dir=skills_dir)

        # Mock httpx.AsyncClient at the library level
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "---\nname: test-skill\n---\nTest body"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await mp.install("https://github.com/user/test-skill", "test-skill")
            assert "installed" in result.lower()
            assert (skills_dir / "test-skill" / "SKILL.md").exists()
            assert (skills_dir / "test-skill" / ".source").exists()

    @pytest.mark.asyncio
    async def test_install_invalid_url(self) -> None:
        mp = SkillMarketplace(skills_dir=Path(tempfile.mkdtemp()))
        result = await mp.install("not-a-github-url")
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_install_no_skills_dir(self) -> None:
        mp = SkillMarketplace()
        result = await mp.install("https://github.com/user/skill")
        assert "error" in result.lower()


class TestWebRoutes:

    @pytest.mark.asyncio
    async def test_health_route(self) -> None:
        from nibot.web.routes import handle_route

        mock_app = MagicMock()
        mock_app.agent._running = True
        mock_app.config.agent.model = "test-model"
        mock_app._channels = []
        mock_app.sessions._cache = {}
        mock_app.agent._tasks = set()

        result = await handle_route(mock_app, "GET", "/api/health", b"", Path("."))
        assert result["status"] == "ok"
        assert result["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_config_route(self) -> None:
        from nibot.web.routes import handle_route

        mock_app = MagicMock()
        mock_app.config.agent.model = "test-model"
        mock_app.config.agent.temperature = 0.7
        mock_app.config.agent.max_tokens = 4096
        mock_app.config.agent.max_iterations = 20
        mock_app.config.agents = {}

        result = await handle_route(mock_app, "GET", "/api/config", b"", Path("."))
        assert result["agent"]["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_404_route(self) -> None:
        from nibot.web.routes import handle_route
        mock_app = MagicMock()
        result = await handle_route(mock_app, "GET", "/nonexistent", b"", Path("."))
        assert "not found" in str(result.get("error", "")).lower()

    @pytest.mark.asyncio
    async def test_tasks_route(self) -> None:
        from nibot.web.routes import handle_route

        mock_app = MagicMock()
        mock_app.subagents.list_tasks.return_value = []

        result = await handle_route(mock_app, "GET", "/api/tasks", b"", Path("."))
        assert "tasks" in result


class TestDeployScript:

    def test_generate_config(self) -> None:
        from scripts.deploy_mac import generate_config
        config = generate_config(local_model="codellama:7b")
        assert "ollama" in config["providers"]
        assert "coder" in config["agents"]

    def test_generate_config_with_cloud(self) -> None:
        from scripts.deploy_mac import generate_config
        config = generate_config(
            cloud_provider="anthropic/claude-sonnet-4-5-20250929",
            cloud_api_key="sk-test",
        )
        assert "anthropic" in config["providers"]
        assert config["providers"]["anthropic"]["api_key"] == "sk-test"

    def test_generate_compose_override(self) -> None:
        from scripts.deploy_mac import generate_compose_override
        override = generate_compose_override()
        assert "services" in override
        assert "nibot" in override["services"]
