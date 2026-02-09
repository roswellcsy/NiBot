"""v1.2 Multi-model routing: reviewer/tester agent types and OpenAI provider mapping."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nibot.bus import MessageBus
from nibot.config import (
    DEFAULT_AGENT_TYPES,
    MODEL_PROVIDER_PREFIXES,
    AgentTypeConfig,
)
from nibot.provider import LiteLLMProvider
from nibot.types import LLMResponse


# ---- Helpers ----

class FakeProvider:
    """Provider that records model kwarg from chat() calls."""

    def __init__(self, content: str = "ok"):
        self.content = content
        self.call_count = 0
        self.last_model: str = ""
        self.last_kwargs: dict = {}

    async def chat(self, messages=None, tools=None, **kwargs) -> LLMResponse:
        self.call_count += 1
        self.last_model = kwargs.get("model", "")
        self.last_kwargs = kwargs
        return LLMResponse(content=self.content, finish_reason="stop", usage={})


# ---- DEFAULT_AGENT_TYPES tests ----

class TestNewAgentTypes:
    """Verify reviewer and tester agent types exist with correct defaults."""

    def test_reviewer_type_exists(self) -> None:
        assert "reviewer" in DEFAULT_AGENT_TYPES

    def test_tester_type_exists(self) -> None:
        assert "tester" in DEFAULT_AGENT_TYPES

    def test_reviewer_tools(self) -> None:
        cfg = DEFAULT_AGENT_TYPES["reviewer"]
        assert "file_read" in cfg.tools
        assert "code_review" in cfg.tools
        assert "git" in cfg.tools
        # reviewer should NOT have write tools
        assert "write_file" not in cfg.tools
        assert "exec" not in cfg.tools

    def test_tester_tools(self) -> None:
        cfg = DEFAULT_AGENT_TYPES["tester"]
        assert "file_read" in cfg.tools
        assert "write_file" in cfg.tools
        assert "exec" in cfg.tools
        assert "test_runner" in cfg.tools

    def test_tester_uses_worktree(self) -> None:
        cfg = DEFAULT_AGENT_TYPES["tester"]
        assert cfg.workspace_mode == "worktree"

    def test_reviewer_has_system_prompt(self) -> None:
        cfg = DEFAULT_AGENT_TYPES["reviewer"]
        assert cfg.system_prompt
        assert "code reviewer" in cfg.system_prompt.lower()

    def test_tester_has_system_prompt(self) -> None:
        cfg = DEFAULT_AGENT_TYPES["tester"]
        assert cfg.system_prompt
        assert "testing" in cfg.system_prompt.lower()

    def test_all_six_types_present(self) -> None:
        expected = {"coder", "researcher", "system", "reviewer", "tester", "evolution"}
        assert set(DEFAULT_AGENT_TYPES.keys()) == expected


# ---- MODEL_PROVIDER_PREFIXES tests ----

class TestProviderPrefixes:
    """Verify OpenAI model prefix mapping."""

    def test_openai_prefix(self) -> None:
        assert MODEL_PROVIDER_PREFIXES["openai"] == "openai"

    def test_gpt_prefix(self) -> None:
        assert MODEL_PROVIDER_PREFIXES["gpt"] == "openai"

    def test_o1_prefix(self) -> None:
        assert MODEL_PROVIDER_PREFIXES["o1"] == "openai"

    def test_o3_prefix(self) -> None:
        assert MODEL_PROVIDER_PREFIXES["o3"] == "openai"

    def test_o4_prefix(self) -> None:
        assert MODEL_PROVIDER_PREFIXES["o4"] == "openai"

    def test_anthropic_still_works(self) -> None:
        assert MODEL_PROVIDER_PREFIXES["anthropic"] == "anthropic"
        assert MODEL_PROVIDER_PREFIXES["claude"] == "anthropic"

    def test_deepseek_still_works(self) -> None:
        assert MODEL_PROVIDER_PREFIXES["deepseek"] == "deepseek"


# ---- ENV_KEY_MAP tests ----

class TestEnvKeyMap:
    """Verify LiteLLMProvider._ENV_KEY_MAP has OpenAI entries."""

    def test_openai_env_key(self) -> None:
        assert "openai" in LiteLLMProvider._ENV_KEY_MAP
        assert LiteLLMProvider._ENV_KEY_MAP["openai"] == "OPENAI_API_KEY"

    def test_gpt_env_key(self) -> None:
        assert "gpt" in LiteLLMProvider._ENV_KEY_MAP
        assert LiteLLMProvider._ENV_KEY_MAP["gpt"] == "OPENAI_API_KEY"


# ---- Model override routing tests ----

class TestModelOverrideRouting:
    """SubagentManager passes model override from AgentTypeConfig to provider.chat()."""

    @pytest.mark.asyncio
    async def test_model_override_passed_to_provider(self) -> None:
        from nibot.subagent import SubagentManager

        bus = MessageBus()
        provider = FakeProvider("done")
        registry = MagicMock()
        registry.get_definitions.return_value = []
        registry.has.return_value = False

        mgr = SubagentManager(provider, registry, bus)

        agent_config = AgentTypeConfig(
            tools=["exec"],
            model="openai/gpt-5.3-codex",
        )
        await mgr.spawn(
            task="write tests",
            label="codex-test",
            origin_channel="test",
            origin_chat_id="1",
            agent_type="tester",
            agent_config=agent_config,
            max_iterations=1,
        )
        await asyncio.sleep(0.3)

        assert provider.call_count >= 1
        assert provider.last_model == "openai/gpt-5.3-codex"

    @pytest.mark.asyncio
    async def test_no_model_override_uses_provider_default(self) -> None:
        from nibot.subagent import SubagentManager

        bus = MessageBus()
        provider = FakeProvider("done")
        registry = MagicMock()
        registry.get_definitions.return_value = []
        registry.has.return_value = False

        mgr = SubagentManager(provider, registry, bus)

        agent_config = AgentTypeConfig(tools=["exec"])
        await mgr.spawn(
            task="review code",
            label="review-test",
            origin_channel="test",
            origin_chat_id="1",
            agent_type="reviewer",
            agent_config=agent_config,
            max_iterations=1,
        )
        await asyncio.sleep(0.3)

        assert provider.call_count >= 1
        # Empty model override means provider uses its own default
        assert provider.last_model == ""

    @pytest.mark.asyncio
    async def test_named_provider_with_model_override(self) -> None:
        from nibot.subagent import SubagentManager

        bus = MessageBus()
        default_provider = FakeProvider("default")
        named_provider = FakeProvider("codex-response")
        registry = MagicMock()
        registry.get_definitions.return_value = []
        registry.has.return_value = False

        mock_pool = MagicMock()
        mock_pool.get = MagicMock(return_value=named_provider)

        mgr = SubagentManager(default_provider, registry, bus, provider_pool=mock_pool)

        agent_config = AgentTypeConfig(
            tools=["exec"],
            model="openai/gpt-5.3-codex",
            provider="openai",
        )
        await mgr.spawn(
            task="fix bug",
            label="coder-test",
            origin_channel="test",
            origin_chat_id="1",
            agent_type="coder",
            agent_config=agent_config,
            max_iterations=1,
        )
        await asyncio.sleep(0.3)

        mock_pool.get.assert_called_with("openai")
        assert named_provider.call_count >= 1
        assert named_provider.last_model == "openai/gpt-5.3-codex"
        assert default_provider.call_count == 0


# ---- AgentTypeConfig user override tests ----

class TestAgentTypeConfigOverride:
    """User config.agents overrides DEFAULT_AGENT_TYPES."""

    def test_model_field_defaults_empty(self) -> None:
        cfg = AgentTypeConfig(tools=["exec"])
        assert cfg.model == ""

    def test_model_field_accepts_value(self) -> None:
        cfg = AgentTypeConfig(tools=["exec"], model="openai/gpt-5.3-codex")
        assert cfg.model == "openai/gpt-5.3-codex"

    def test_provider_field_defaults_empty(self) -> None:
        cfg = AgentTypeConfig(tools=["exec"])
        assert cfg.provider == ""

    def test_fallback_chain_defaults_empty(self) -> None:
        cfg = AgentTypeConfig(tools=["exec"])
        assert cfg.fallback_chain == []

    def test_fallback_chain_accepts_list(self) -> None:
        cfg = AgentTypeConfig(tools=["exec"], fallback_chain=["anthropic", "openai"])
        assert cfg.fallback_chain == ["anthropic", "openai"]
