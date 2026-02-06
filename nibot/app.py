"""Composition root -- wire everything together."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from nibot.agent import AgentLoop
from nibot.bus import MessageBus
from nibot.channel import BaseChannel
from nibot.config import NiBotConfig, load_config
from nibot.context import ContextBuilder
from nibot.log import logger
from nibot.memory import MemoryStore
from nibot.provider import LLMProvider, LiteLLMProvider
from nibot.registry import Tool, ToolRegistry
from nibot.session import SessionManager
from nibot.skills import SkillsLoader
from nibot.subagent import SubagentManager


class NiBot:
    """Main application. Create, configure, run."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config = load_config(config_path)
        self.bus = MessageBus()
        self.registry = ToolRegistry()

        workspace = Path(self.config.agent.workspace).expanduser()
        workspace.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace

        self.sessions = SessionManager(workspace / "sessions")
        self.memory = MemoryStore(workspace / "memory")
        self.skills = SkillsLoader([
            workspace / "skills",
            Path(__file__).parent / "skills",
        ])
        self.provider: LLMProvider = self._create_provider()
        self.context_builder = ContextBuilder(
            config=self.config,
            memory=self.memory,
            skills=self.skills,
            workspace=workspace,
        )
        self.agent = AgentLoop(
            bus=self.bus,
            provider=self.provider,
            registry=self.registry,
            sessions=self.sessions,
            context_builder=self.context_builder,
            config=self.config,
        )
        self.subagents = SubagentManager(self.provider, self.registry, self.bus)
        self._channels: list[BaseChannel] = []

    def add_channel(self, channel: BaseChannel) -> "NiBot":
        self._channels.append(channel)
        self.bus.subscribe_outbound(channel.name, channel.send)
        return self

    def add_tool(self, tool: Tool) -> "NiBot":
        self.registry.register(tool)
        return self

    async def run(self) -> None:
        self._register_builtin_tools()
        self.skills.load_all()
        logger.info(f"NiBot starting -- model={self.config.agent.model}, "
                     f"workspace={self.workspace}, channels={len(self._channels)}")
        tasks = [
            asyncio.create_task(self.agent.run()),
            asyncio.create_task(self.bus.dispatch_outbound()),
        ]
        for ch in self._channels:
            tasks.append(asyncio.create_task(ch.start()))
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.agent.stop()
            self.bus.stop()
            for ch in self._channels:
                await ch.stop()

    def _create_provider(self) -> LLMProvider:
        model = self.config.agent.model
        api_key, api_base = self._resolve_provider_credentials(model)
        return LiteLLMProvider(
            model=model,
            api_key=api_key,
            api_base=api_base,
            max_tokens=self.config.agent.max_tokens,
            temperature=self.config.agent.temperature,
        )

    def _register_builtin_tools(self) -> None:
        from nibot.tools.file_tools import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
        from nibot.tools.exec_tool import ExecTool
        from nibot.tools.message_tool import MessageTool
        from nibot.tools.spawn_tool import SpawnTool
        from nibot.tools.web_tools import WebFetchTool, WebSearchTool

        ws = self.workspace
        for tool in [
            ReadFileTool(ws),
            WriteFileTool(ws),
            EditFileTool(ws),
            ListDirTool(ws),
            ExecTool(ws, timeout=self.config.tools.exec_timeout),
            WebSearchTool(api_key=self.config.tools.web_search_api_key),
            WebFetchTool(),
            MessageTool(self._bus_ref()),
            SpawnTool(self.subagents),
        ]:
            if not self.registry.has(tool.name):
                self.registry.register(tool)

    def _bus_ref(self) -> MessageBus:
        return self.bus

    def _resolve_provider_credentials(self, model: str) -> tuple[str, str]:
        providers = self.config.providers
        if "anthropic" in model or "claude" in model:
            return providers.anthropic.api_key, providers.anthropic.api_base
        if "deepseek" in model:
            return providers.deepseek.api_key, providers.deepseek.api_base
        if "openrouter" in model:
            return providers.openrouter.api_key, providers.openrouter.api_base
        return providers.openai.api_key, providers.openai.api_base
