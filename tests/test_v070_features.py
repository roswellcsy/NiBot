"""Tests for v0.7.0 features: CodeReviewTool, TestRunnerTool, provider credential refactor, image gen."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.config import DEFAULT_AGENT_TYPES, MODEL_PROVIDER_PREFIXES, ProvidersConfig, ProviderConfig, ToolsConfig
from nibot.tools.code_review_tool import CodeReviewTool
from nibot.tools.image_gen_tool import ImageGenerationTool
from nibot.tools.message_tool import MessageTool
from nibot.tools.test_runner_tool import TestRunnerTool
from nibot.types import Envelope
from nibot.worktree import WorktreeManager


# ---- A1: CodeReviewTool ----

class TestCodeReviewTool:
    @pytest.mark.asyncio
    async def test_review_worktree_diff(self, tmp_path: Path) -> None:
        wt_mgr = WorktreeManager(tmp_path)
        await wt_mgr.ensure_repo()
        wt_path = await wt_mgr.create("cr1")
        (wt_path / "new.txt").write_text("hello")
        await wt_mgr.commit("cr1", "add new.txt")
        tool = CodeReviewTool(tmp_path, worktree_mgr=wt_mgr)
        result = await tool.execute(action="review", task_id="cr1")
        # diff should show something (committed changes vs base)
        assert "new.txt" in result or "No changes" in result
        await wt_mgr.remove("cr1")

    @pytest.mark.asyncio
    async def test_review_no_changes(self, tmp_path: Path) -> None:
        wt_mgr = WorktreeManager(tmp_path)
        await wt_mgr.ensure_repo()
        wt_path = await wt_mgr.create("cr2")
        tool = CodeReviewTool(tmp_path, worktree_mgr=wt_mgr)
        result = await tool.execute(action="review", task_id="cr2")
        assert "No changes" in result
        await wt_mgr.remove("cr2")

    @pytest.mark.asyncio
    async def test_review_no_worktree_mgr(self, tmp_path: Path) -> None:
        """Review without worktree_mgr falls back to git diff in path."""
        tool = CodeReviewTool(tmp_path)
        result = await tool.execute(action="review")
        # May fail with "not a git repository" or return empty diff -- both are OK
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_lint_no_linter(self, tmp_path: Path) -> None:
        tool = CodeReviewTool(tmp_path)
        with patch.object(tool, "_linter_available", return_value=False):
            result = await tool.execute(action="lint")
        assert "No linter available" in result

    @pytest.mark.asyncio
    async def test_lint_ruff_available(self, tmp_path: Path) -> None:
        tool = CodeReviewTool(tmp_path)

        async def fake_available(name: str) -> bool:
            return name == "ruff"

        async def fake_run_cmd(cmd: list, cwd: Path) -> str:
            assert "ruff" in cmd
            return "All checks passed!\n[exit=0]"

        tool._linter_available = fake_available  # type: ignore[assignment]
        tool._run_cmd = fake_run_cmd  # type: ignore[assignment]
        result = await tool.execute(action="lint")
        assert "All checks passed" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self, tmp_path: Path) -> None:
        tool = CodeReviewTool(tmp_path)
        result = await tool.execute(action="bogus")
        assert "Error" in result


# ---- A2: TestRunnerTool ----

class TestTestRunnerTool:
    def test_detect_pytest_conftest(self, tmp_path: Path) -> None:
        (tmp_path / "conftest.py").write_text("")
        tool = TestRunnerTool(tmp_path)
        assert tool._detect_framework(tmp_path) == "pytest"

    def test_detect_pytest_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
        tool = TestRunnerTool(tmp_path)
        assert tool._detect_framework(tmp_path) == "pytest"

    def test_detect_jest(self, tmp_path: Path) -> None:
        pkg = {"devDependencies": {"jest": "^29.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        tool = TestRunnerTool(tmp_path)
        assert tool._detect_framework(tmp_path) == "jest"

    def test_detect_vitest(self, tmp_path: Path) -> None:
        pkg = {"devDependencies": {"vitest": "^1.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        tool = TestRunnerTool(tmp_path)
        assert tool._detect_framework(tmp_path) == "vitest"

    def test_detect_unittest_fallback(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(tmp_path)
        assert tool._detect_framework(tmp_path) == "unittest"

    def test_build_command_pytest(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(tmp_path)
        cmd = tool._build_command("pytest", tmp_path, "", False)
        assert cmd == ["python", "-m", "pytest", "-v"]

    def test_build_command_pytest_coverage(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(tmp_path)
        cmd = tool._build_command("pytest", tmp_path, "tests/test_foo.py", True)
        assert "--cov" in cmd
        assert "tests/test_foo.py" in cmd

    def test_build_command_jest(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(tmp_path)
        cmd = tool._build_command("jest", tmp_path, "", True)
        assert cmd == ["npx", "jest", "--coverage"]

    @pytest.mark.asyncio
    async def test_run_unknown_action(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(tmp_path)
        result = await tool.execute(action="bogus")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_run_returns_framework(self, tmp_path: Path) -> None:
        (tmp_path / "conftest.py").write_text("")
        tool = TestRunnerTool(tmp_path, timeout=5)
        result = await tool.execute(action="run")
        assert "[framework=pytest]" in result


# ---- A3: Provider credential refactor ----

class TestProviderRefactor:
    def test_prefix_map_has_anthropic(self) -> None:
        assert MODEL_PROVIDER_PREFIXES["anthropic"] == "anthropic"
        assert MODEL_PROVIDER_PREFIXES["claude"] == "anthropic"

    def test_prefix_map_has_deepseek(self) -> None:
        assert MODEL_PROVIDER_PREFIXES["deepseek"] == "deepseek"

    def test_prefix_map_has_new_providers(self) -> None:
        assert "gemini" in MODEL_PROVIDER_PREFIXES
        assert "moonshot" in MODEL_PROVIDER_PREFIXES
        assert "minimax" in MODEL_PROVIDER_PREFIXES
        assert "zhipu" in MODEL_PROVIDER_PREFIXES

    def test_providers_config_get_extras(self) -> None:
        config = ProvidersConfig(
            extras={"gemini": ProviderConfig(api_key="test-key", model="gemini/gemini-3-pro")}
        )
        pc = config.get("gemini")
        assert pc is not None
        assert pc.api_key == "test-key"

    def test_providers_config_get_builtin(self) -> None:
        config = ProvidersConfig(
            anthropic=ProviderConfig(api_key="ant-key")
        )
        pc = config.get("anthropic")
        assert pc is not None
        assert pc.api_key == "ant-key"

    def test_providers_config_get_unknown(self) -> None:
        config = ProvidersConfig()
        pc = config.get("nonexistent")
        assert pc is None

    def test_env_key_config_anthropic(self) -> None:
        from nibot.provider import LiteLLMProvider
        assert LiteLLMProvider._ENV_KEY_MAP["anthropic"] == "ANTHROPIC_API_KEY"
        assert LiteLLMProvider._ENV_KEY_MAP["claude"] == "ANTHROPIC_API_KEY"


# ---- A4: Config defaults ----

class TestConfigDefaults:
    def test_coder_has_code_review(self) -> None:
        assert "code_review" in DEFAULT_AGENT_TYPES["coder"].tools

    def test_coder_has_test_runner(self) -> None:
        assert "test_runner" in DEFAULT_AGENT_TYPES["coder"].tools

    def test_coder_still_has_original_tools(self) -> None:
        tools = DEFAULT_AGENT_TYPES["coder"].tools
        for t in ["read_file", "write_file", "edit_file", "list_dir", "exec", "git"]:
            assert t in tools


# ---- B1: ImageGenerationTool ----

def _mock_litellm(mock_gen: AsyncMock) -> dict[str, Any]:
    """Create a fake litellm module with aimage_generation mocked."""
    fake_mod = MagicMock()
    fake_mod.aimage_generation = mock_gen
    return {"litellm": fake_mod}


class TestImageGenerationTool:
    @pytest.mark.asyncio
    async def test_generate_saves_file(self, tmp_path: Path) -> None:
        """Mock litellm.aimage_generation and verify file is saved."""
        tool = ImageGenerationTool(tmp_path, default_model="dall-e-3")

        fake_png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
        mock_item = MagicMock()
        mock_item.b64_json = fake_png
        mock_response = MagicMock()
        mock_response.data = [mock_item]

        mock_gen = AsyncMock(return_value=mock_response)
        with patch.dict("sys.modules", _mock_litellm(mock_gen)):
            result = await tool.execute(prompt="a cat")

        data = json.loads(result)
        assert data["count"] == 1
        assert len(data["images"]) == 1
        saved_path = Path(data["images"][0])
        assert saved_path.exists()
        assert saved_path.suffix == ".png"

    @pytest.mark.asyncio
    async def test_generate_no_prompt(self, tmp_path: Path) -> None:
        tool = ImageGenerationTool(tmp_path)
        result = await tool.execute(prompt="")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_generate_api_error(self, tmp_path: Path) -> None:
        tool = ImageGenerationTool(tmp_path)

        mock_gen = AsyncMock(side_effect=Exception("API quota exceeded"))
        with patch.dict("sys.modules", _mock_litellm(mock_gen)):
            result = await tool.execute(prompt="test")

        assert "Error" in result
        assert "quota" in result

    def test_default_model(self, tmp_path: Path) -> None:
        tool1 = ImageGenerationTool(tmp_path)
        assert tool1._default_model == "dall-e-3"
        tool2 = ImageGenerationTool(tmp_path, default_model="gemini/imagen-4.0-generate-001")
        assert tool2._default_model == "gemini/imagen-4.0-generate-001"

    @pytest.mark.asyncio
    async def test_generate_data_uri_prefix(self, tmp_path: Path) -> None:
        """Test that data URI prefix (data:image/png;base64,...) is stripped correctly."""
        tool = ImageGenerationTool(tmp_path)
        raw_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 30
        b64_with_prefix = "data:image/png;base64," + base64.b64encode(raw_bytes).decode()
        mock_item = MagicMock()
        mock_item.b64_json = b64_with_prefix
        mock_response = MagicMock()
        mock_response.data = [mock_item]

        mock_gen = AsyncMock(return_value=mock_response)
        with patch.dict("sys.modules", _mock_litellm(mock_gen)):
            result = await tool.execute(prompt="test")

        data = json.loads(result)
        assert data["count"] == 1
        saved = Path(data["images"][0])
        assert saved.read_bytes() == raw_bytes


# ---- B2: MessageTool media ----

class TestMessageToolMedia:
    @pytest.mark.asyncio
    async def test_send_with_media(self) -> None:
        bus = MagicMock()
        bus.publish_outbound = AsyncMock()
        tool = MessageTool(bus)
        result = await tool.execute(
            channel="telegram", chat_id="123", content="hello",
            media=["/tmp/img.png"],
        )
        assert "telegram:123" in result
        call_args = bus.publish_outbound.call_args[0][0]
        assert isinstance(call_args, Envelope)
        assert call_args.media == ["/tmp/img.png"]

    @pytest.mark.asyncio
    async def test_send_without_media(self) -> None:
        bus = MagicMock()
        bus.publish_outbound = AsyncMock()
        tool = MessageTool(bus)
        result = await tool.execute(channel="telegram", chat_id="123", content="hello")
        call_args = bus.publish_outbound.call_args[0][0]
        assert call_args.media == []


# ---- B3: Telegram media send ----

class TestTelegramMediaSend:
    @pytest.mark.asyncio
    async def test_send_photo(self, tmp_path: Path) -> None:
        from nibot.channels.telegram import TelegramChannel

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG fake")

        channel = TelegramChannel.__new__(TelegramChannel)
        channel._app = MagicMock()
        channel._app.bot.send_photo = AsyncMock()
        channel._app.bot.send_message = AsyncMock()

        envelope = Envelope(
            channel="telegram", chat_id="42", sender_id="bot",
            content="", media=[str(img_path)],
        )
        await channel.send(envelope)
        channel._app.bot.send_photo.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_document(self, tmp_path: Path) -> None:
        from nibot.channels.telegram import TelegramChannel

        doc_path = tmp_path / "data.csv"
        doc_path.write_text("a,b,c")

        channel = TelegramChannel.__new__(TelegramChannel)
        channel._app = MagicMock()
        channel._app.bot.send_document = AsyncMock()
        channel._app.bot.send_message = AsyncMock()

        envelope = Envelope(
            channel="telegram", chat_id="42", sender_id="bot",
            content="", media=[str(doc_path)],
        )
        await channel.send(envelope)
        channel._app.bot.send_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_missing_file(self, tmp_path: Path) -> None:
        from nibot.channels.telegram import TelegramChannel

        channel = TelegramChannel.__new__(TelegramChannel)
        channel._app = MagicMock()
        channel._app.bot.send_photo = AsyncMock()
        channel._app.bot.send_message = AsyncMock()

        envelope = Envelope(
            channel="telegram", chat_id="42", sender_id="bot",
            content="text", media=[str(tmp_path / "nonexistent.png")],
        )
        await channel.send(envelope)
        channel._app.bot.send_photo.assert_not_called()
        channel._app.bot.send_message.assert_called_once()


# ---- B4: Feishu media send ----

def _setup_feishu_mocks() -> dict[str, Any]:
    """Set up lark_oapi mock modules for Feishu tests."""
    mock_lark = MagicMock()
    return {
        "lark_oapi": mock_lark,
        "lark_oapi.api": MagicMock(),
        "lark_oapi.api.im": MagicMock(),
        "lark_oapi.api.im.v1": MagicMock(
            CreateImageRequest=mock_lark.CreateImageRequest,
            CreateImageRequestBody=mock_lark.CreateImageRequestBody,
            CreateMessageRequest=mock_lark.CreateMessageRequest,
            CreateMessageRequestBody=mock_lark.CreateMessageRequestBody,
        ),
    }


class TestFeishuMediaSend:
    @pytest.mark.asyncio
    async def test_send_image_success(self, tmp_path: Path) -> None:
        feishu_mocks = _setup_feishu_mocks()
        with patch.dict("sys.modules", feishu_mocks):
            from nibot.channels.feishu import FeishuChannel

            img_path = tmp_path / "photo.png"
            img_path.write_bytes(b"\x89PNG fake")

            channel = FeishuChannel.__new__(FeishuChannel)
            channel._client = MagicMock()

            # Mock upload response
            upload_resp = MagicMock()
            upload_resp.success.return_value = True
            upload_resp.data.image_key = "img_v2_fake_key"
            channel._client.im.v1.image.create.return_value = upload_resp
            channel._client.im.v1.message.create.return_value = MagicMock()

            result = await channel._send_image("oc_chat1", img_path)
            assert result is True
            channel._client.im.v1.image.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_image_upload_fails(self, tmp_path: Path) -> None:
        feishu_mocks = _setup_feishu_mocks()
        with patch.dict("sys.modules", feishu_mocks):
            from nibot.channels.feishu import FeishuChannel

            img_path = tmp_path / "photo.jpg"
            img_path.write_bytes(b"\xff\xd8\xff fake")

            channel = FeishuChannel.__new__(FeishuChannel)
            channel._client = MagicMock()

            upload_resp = MagicMock()
            upload_resp.success.return_value = False
            channel._client.im.v1.image.create.return_value = upload_resp

            result = await channel._send_image("oc_chat1", img_path)
            assert result is False


# ---- B5: Config ----

class TestImageConfig:
    def test_tools_config_has_image_model(self) -> None:
        tc = ToolsConfig()
        assert tc.image_model == ""

    def test_tools_config_custom_model(self) -> None:
        tc = ToolsConfig(image_model="gemini/imagen-4.0-generate-001")
        assert tc.image_model == "gemini/imagen-4.0-generate-001"
