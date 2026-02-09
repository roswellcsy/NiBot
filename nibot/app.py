"""Composition root -- wire everything together."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any

from nibot.agent import AgentLoop
from nibot.bus import MessageBus
from nibot.channel import BaseChannel
from nibot.config import DEFAULT_AGENT_TYPES, NiBotConfig, _default_config_path, default_evolution_schedule, load_config, validate_startup
from nibot.context import ContextBuilder
from nibot.event_log import EventLog
from nibot.evolution_log import EvolutionLog
from nibot.evolution_trigger import EvolutionTrigger
from nibot.log import logger
from nibot.memory import MemoryStore
from nibot.provider import LLMProvider, LiteLLMProvider
from nibot.provider_pool import ProviderPool
from nibot.registry import Tool, ToolRegistry
from nibot.scheduler import SchedulerManager
from nibot.session import SessionManager
from nibot.skills import SkillsLoader
from nibot.subagent import SubagentManager
from nibot.worktree import WorktreeManager


class NiBot:
    """Main application. Create, configure, run."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config = load_config(config_path)

        # Configure logging early -- before any logger.info() calls
        from nibot.log import configure as _configure_log
        lc = self.config.log
        _configure_log(
            level=lc.level, fmt=lc.format, json_format=lc.json_format,
            file=lc.file, rotation=lc.rotation, retention=lc.retention,
        )

        validate_startup(self.config)

        self._config_path = Path(config_path).expanduser() if config_path else _default_config_path()
        self.bus = MessageBus(maxsize=self.config.agent.bus_queue_maxsize)

        workspace = Path(self.config.agent.workspace).expanduser()
        workspace.mkdir(parents=True, exist_ok=True)
        import os
        if not os.access(workspace, os.W_OK):
            raise RuntimeError(f"Workspace directory not writable: {workspace}")
        self.workspace = workspace

        # Structured event log (created early -- injected into registry, pool, agent)
        el_cfg = self.config.event_log
        el_path = Path(el_cfg.file).expanduser() if el_cfg.file else workspace / "events.jsonl"
        self.event_log = EventLog(el_path, enabled=el_cfg.enabled)

        self.registry = ToolRegistry(event_log=self.event_log)
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
        self.evolution_log = EvolutionLog(workspace)
        # Build quota configs from provider settings
        quota_configs: dict[str, Any] = {}
        for pname in ("anthropic", "openai", "openrouter", "deepseek"):
            pc = self.config.providers.get(pname)
            if pc:
                quota_configs[pname] = pc.quota
        for pname, pc in self.config.providers.extras.items():
            quota_configs[pname] = pc.quota
        self.provider_pool = ProviderPool(self.config.providers, self.provider, quota_configs=quota_configs, event_log=self.event_log)
        self.worktree_mgr = WorktreeManager(workspace)
        self.evo_trigger = EvolutionTrigger(
            bus=self.bus,
            sessions=self.sessions,
            enabled=self.config.agent.auto_evolution,
        )

        # Rate limiter
        from nibot.rate_limiter import SlidingWindowRateLimiter, RateLimitConfig as _RLC
        rl_cfg = self.config.rate_limit
        self._rate_limiter = SlidingWindowRateLimiter(
            _RLC(per_user_rpm=rl_cfg.per_user_rpm, per_channel_rpm=rl_cfg.per_channel_rpm, enabled=rl_cfg.enabled)
        )

        self.agent = AgentLoop(
            bus=self.bus,
            provider=self.provider,
            registry=self.registry,
            sessions=self.sessions,
            context_builder=self.context_builder,
            config=self.config,
            evo_trigger=self.evo_trigger,
            rate_limiter=self._rate_limiter,
            provider_pool=self.provider_pool,
            event_log=self.event_log,
        )
        self.subagents = SubagentManager(
            self.provider, self.registry, self.bus,
            provider_pool=self.provider_pool,
            worktree_mgr=self.worktree_mgr,
            workspace=workspace,
            sessions=self.sessions,
            skills=self.skills,
            evolution_log=self.evolution_log,
        )
        # Skill marketplace
        from nibot.marketplace import SkillMarketplace
        mp_cfg = self.config.marketplace
        self.marketplace: SkillMarketplace | None = None
        if mp_cfg.enabled:
            self.marketplace = SkillMarketplace(
                github_token=mp_cfg.github_token,
                skills_dir=workspace / "skills",
            )

        # Pipeline engine
        if not self.config.agents:
            self.config.agents = dict(DEFAULT_AGENT_TYPES)

        from nibot.tools.pipeline_tool import PipelineEngine
        self.pipeline_engine = PipelineEngine(self.subagents, self.config.agents)

        self.scheduler = SchedulerManager(self.bus, self.config.schedules)
        if self.config.agent.auto_evolution:
            sched = default_evolution_schedule()
            sched.enabled = True  # auto_evolution=True means schedule should be active
            if not any(j.id == sched.id for j in self.config.schedules):
                self.scheduler.add(sched)
        self._channels: list[BaseChannel] = []
        self._register_builtin_tools()

    def add_channel(self, channel: BaseChannel) -> "NiBot":
        self._channels.append(channel)
        self.bus.subscribe_outbound(channel.name, channel.send)
        return self

    def add_tool(self, tool: Tool) -> "NiBot":
        self.registry.register(tool)
        return self

    async def run(self) -> None:
        self.skills.load_all()

        # Start health server (no-op if not enabled in config)
        from nibot.health import start_health_server
        self._health_server = await start_health_server(self)

        # Start MCP bridges (connect to external MCP servers, register adapted tools)
        self._mcp_bridges: list[Any] = []
        await self._start_mcp_bridges()

        # Start webhook server (WeCom + API channels)
        self._webhook_server = None
        await self._start_webhook_server()

        # Start web management panel
        self._web_panel = None
        await self._start_web_panel()

        # Start vault channel (file system watcher)
        await self._start_vault_channel()

        logger.info(f"NiBot starting -- model={self.config.agent.model}, "
                     f"workspace={self.workspace}, channels={len(self._channels)}")

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def _signal_handler() -> None:
            if not shutdown_event.is_set():
                logger.info("Shutdown signal received, stopping gracefully...")
                shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except (NotImplementedError, OSError):
                # Windows ProactorEventLoop doesn't support add_signal_handler.
                # Graceful shutdown still works: Ctrl+C -> KeyboardInterrupt ->
                # CancelledError -> finally: _shutdown(). SIGTERM is unsupported
                # on Windows (Python limitation).
                pass

        # Start channels first (these return after initialization, not long-running)
        for ch in self._channels:
            await ch.start()

        tasks = [
            asyncio.create_task(self.agent.run()),
            asyncio.create_task(self.bus.dispatch_outbound()),
            asyncio.create_task(self.scheduler.run()),
        ]

        try:
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            done, _ = await asyncio.wait(
                [*tasks, shutdown_task], return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown(tasks)

    async def _shutdown(self, tasks: list[asyncio.Task[Any]]) -> None:
        """Graceful shutdown: stop components, wait for in-flight work, cancel rest."""
        logger.info("Shutting down components...")

        # 0. Close servers
        if getattr(self, "_health_server", None):
            self._health_server.close()
            await self._health_server.wait_closed()
        if getattr(self, "_webhook_server", None):
            await self._webhook_server.stop()
        if getattr(self, "_web_panel", None):
            await self._web_panel.stop()
        for bridge in getattr(self, "_mcp_bridges", []):
            try:
                await bridge.disconnect()
            except Exception as e:
                logger.warning(f"MCP bridge disconnect error: {e}")

        # 1. Signal all components to stop accepting new work
        self.agent.stop()
        self.bus.stop()
        self.scheduler.stop()
        for ch in self._channels:
            try:
                await ch.stop()
            except Exception as e:
                logger.warning(f"Channel stop error: {e}")

        # 2. Wait for in-flight agent tasks (with timeout)
        pending = list(self.agent._tasks) + list(self.agent._bg_tasks)
        if pending:
            logger.info(f"Waiting for {len(pending)} in-flight agent tasks...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for agent tasks, cancelling...")

        # 3. Wait for subagent background tasks
        if self.subagents._tasks:
            logger.info(f"Waiting for {len(self.subagents._tasks)} subagent tasks...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*list(self.subagents._tasks.values()), return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for subagent tasks, cancelling...")

        # 4. Cancel all remaining top-level tasks
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("NiBot shutdown complete.")

    def _create_provider(self) -> LLMProvider:
        model = self.config.agent.model
        api_key, api_base = self._resolve_provider_credentials(model)
        return LiteLLMProvider(
            model=model,
            api_key=api_key,
            api_base=api_base,
            max_tokens=self.config.agent.max_tokens,
            temperature=self.config.agent.temperature,
            max_retries=self.config.agent.llm_max_retries,
            retry_base_delay=self.config.agent.llm_retry_base_delay,
        )

    def _register_builtin_tools(self) -> None:
        from nibot.tools.admin_tools import ConfigTool, ScheduleTool, SkillTool
        from nibot.tools.analyze_tool import AnalyzeTool
        from nibot.tools.code_review_tool import CodeReviewTool
        from nibot.tools.exec_tool import ExecTool
        from nibot.tools.file_tools import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
        from nibot.tools.git_tool import GitTool
        from nibot.tools.image_gen_tool import ImageGenerationTool
        from nibot.tools.message_tool import MessageTool
        from nibot.tools.pipeline_tool import PipelineTool
        from nibot.tools.scaffold_tool import ScaffoldTool
        from nibot.tools.skill_runner import SkillRunnerTool
        from nibot.tools.spawn_tool import DelegateTool
        from nibot.tools.test_runner_tool import TestRunnerTool
        from nibot.tools.web_tools import WebFetchTool, WebSearchTool

        ws = self.workspace
        restrict = self.config.tools.restrict_to_workspace
        for tool in [
            ReadFileTool(ws, restrict=restrict),
            WriteFileTool(ws, restrict=restrict),
            EditFileTool(ws, restrict=restrict),
            ListDirTool(ws, restrict=restrict),
            ExecTool(ws, timeout=self.config.tools.exec_timeout,
                    sandbox_enabled=self.config.tools.sandbox_enabled,
                    sandbox_memory_mb=self.config.tools.sandbox_memory_mb),
            WebSearchTool(
                api_key=self.config.tools.web_search_api_key,
                anthropic_api_key=self.config.providers.anthropic.api_key,
            ),
            WebFetchTool(),
            MessageTool(self.bus),
            GitTool(self.worktree_mgr),
            AnalyzeTool(self.sessions, skills=self.skills, evolution_log=self.evolution_log),
            CodeReviewTool(ws, worktree_mgr=self.worktree_mgr),
            TestRunnerTool(ws, timeout=self.config.tools.exec_timeout * 2),
            ImageGenerationTool(ws, default_model=self.config.tools.image_model),
            DelegateTool(self.subagents, self.config.agents),
            ConfigTool(self.config, ws),
            ScheduleTool(self.scheduler, self.config, ws, config_path=self._config_path),
            SkillTool(self.skills, marketplace=self.marketplace),
            SkillRunnerTool(self.skills, ws, timeout=self.config.tools.exec_timeout,
                           sandbox_enabled=self.config.tools.sandbox_enabled),
            PipelineTool(self.pipeline_engine),
            ScaffoldTool(ws),
        ]:
            if not self.registry.has(tool.name):
                self.registry.register(tool)

    async def _start_mcp_bridges(self) -> None:
        """Connect to configured MCP servers and register their tools."""
        from nibot.tools.mcp_bridge import MCPBridgeTool
        for name, srv_cfg in self.config.tools.mcp_servers.items():
            if not srv_cfg.command:
                continue
            try:
                bridge = MCPBridgeTool(name, srv_cfg.command, srv_cfg.args, srv_cfg.env)
                adapters = await bridge.connect_and_discover()
                for adapter in adapters:
                    if not self.registry.has(adapter.name):
                        self.registry.register(adapter)
                self._mcp_bridges.append(bridge)
            except Exception as e:
                logger.warning(f"MCP bridge '{name}' failed to start: {e}")

    async def _start_webhook_server(self) -> None:
        """Start unified webhook/API HTTP server if WeCom or API channel is enabled."""
        wecom_cfg = self.config.channels.wecom
        api_cfg = self.config.channels.api
        wh_cfg = self.config.webhook

        if not (wh_cfg.enabled or wecom_cfg.enabled or api_cfg.enabled):
            return

        from nibot.webhook_server import WebhookServer

        wecom_ch = None
        if wecom_cfg.enabled:
            from nibot.channels.wecom import WeComChannel
            wecom_ch = WeComChannel(wecom_cfg, self.bus)
            self.add_channel(wecom_ch)

        api_ch = None
        if api_cfg.enabled:
            from nibot.channels.api import APIChannel
            api_ch = APIChannel(api_cfg, self.bus)
            self.add_channel(api_ch)

        self._webhook_server = WebhookServer(
            host=wh_cfg.host, port=wh_cfg.port,
            wecom_channel=wecom_ch, api_channel=api_ch,
        )
        await self._webhook_server.start()

    async def _start_web_panel(self) -> None:
        """Start web management panel if enabled."""
        wp_cfg = self.config.web_panel
        if not wp_cfg.enabled:
            return

        # Per-stream queues for web chat SSE
        self._web_streams: dict[str, asyncio.Queue[Any]] = {}

        async def _web_outbound(envelope: "Envelope") -> None:
            meta = envelope.metadata or {}
            stream_id = meta.get("stream_id", "")
            if not stream_id:
                return
            queue = self._web_streams.get(stream_id)
            if not queue:
                return
            # Progress events (thinking / tool_start / tool_done)
            progress = meta.get("progress")
            if progress:
                await queue.put({
                    "type": "progress",
                    "event": progress,
                    "tool_name": meta.get("tool_name", ""),
                    "iteration": meta.get("iteration", 0),
                    "max_iterations": meta.get("max_iterations", 0),
                    "elapsed": meta.get("elapsed", 0),
                })
                return
            if meta.get("streaming"):
                await queue.put({"type": "chunk", "content": envelope.content})
                if meta.get("stream_done"):
                    # Only close SSE when no tool_calls follow
                    if not meta.get("has_tool_calls"):
                        await queue.put(None)
            else:
                await queue.put({"type": "done", "content": envelope.content})
                await queue.put(None)

        self.bus.subscribe_outbound("web", _web_outbound)

        from nibot.web.server import WebPanel
        self._web_panel = WebPanel(
            app=self, host=wp_cfg.host, port=wp_cfg.port, auth_token=wp_cfg.auth_token,
            rate_limit_rpm=wp_cfg.rate_limit_rpm, cors_origin=wp_cfg.cors_origin,
        )
        await self._web_panel.start()

    async def _start_vault_channel(self) -> None:
        """Start vault file-watcher channel if enabled."""
        vault_cfg = self.config.channels.vault
        if not vault_cfg.enabled or not vault_cfg.watch_dir:
            return
        from nibot.channels.vault import VaultChannel
        ch = VaultChannel(vault_cfg, self.bus, workspace=self.workspace)
        self.add_channel(ch)

    def _resolve_provider_credentials(self, model: str) -> tuple[str, str]:
        from nibot.config import MODEL_PROVIDER_PREFIXES

        model_lower = model.lower()
        for prefix, provider_name in MODEL_PROVIDER_PREFIXES.items():
            if prefix in model_lower:
                pc = self.config.providers.get(provider_name)
                if pc and pc.api_key:
                    return pc.api_key, pc.api_base
        return self.config.providers.openai.api_key, self.config.providers.openai.api_base
