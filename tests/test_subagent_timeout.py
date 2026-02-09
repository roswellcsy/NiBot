"""Subagent wall-clock timeout tests (Phase 8 v1.4)."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nibot.bus import MessageBus
from nibot.config import AgentTypeConfig
from nibot.provider import LLMProvider, LLMResponse
from nibot.registry import ToolRegistry
from nibot.subagent import SubagentManager


class _HangingProvider(LLMProvider):
    """Provider that blocks forever, simulating a stuck LLM call."""
    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None, **kw: Any) -> LLMResponse:
        await asyncio.sleep(3600)  # 1 hour -- will be cancelled by timeout
        return LLMResponse(content="should never reach here")


class _FastProvider(LLMProvider):
    """Provider that returns immediately."""
    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None, **kw: Any) -> LLMResponse:
        return LLMResponse(content="fast result")


@pytest.mark.asyncio
async def test_subagent_timeout_kills_stuck_task():
    """Stuck provider triggers wall-clock timeout; task_info.status='error'."""
    bus = MessageBus()
    mgr = SubagentManager(
        provider=_HangingProvider(), registry=ToolRegistry(), bus=bus,
    )

    captured: list = []

    async def cap(env):
        captured.append(env)

    bus.subscribe_outbound("test", cap)
    dt = asyncio.create_task(bus.dispatch_outbound())

    config = AgentTypeConfig(timeout_seconds=1)  # 1-second timeout
    task_id = await mgr.spawn(
        task="hang forever", label="timeout-test",
        origin_channel="test", origin_chat_id="c1",
        agent_type="test", agent_config=config,
    )

    # Wait enough for timeout to fire (1s timeout + buffer)
    await asyncio.sleep(2.5)
    bus.stop()
    dt.cancel()
    try:
        await dt
    except asyncio.CancelledError:
        pass

    info = mgr.get_task_info(task_id)
    assert info is not None
    assert info.status == "error"
    assert "timed out" in info.result.lower()


@pytest.mark.asyncio
async def test_subagent_normal_completes_before_timeout():
    """Fast provider completes well within timeout; status='completed'."""
    bus = MessageBus()
    mgr = SubagentManager(
        provider=_FastProvider(), registry=ToolRegistry(), bus=bus,
    )

    captured: list = []

    async def cap(env):
        captured.append(env)

    bus.subscribe_outbound("test", cap)
    dt = asyncio.create_task(bus.dispatch_outbound())

    config = AgentTypeConfig(timeout_seconds=30)
    task_id = await mgr.spawn(
        task="quick task", label="fast-test",
        origin_channel="test", origin_chat_id="c1",
        agent_type="test", agent_config=config,
    )

    await asyncio.sleep(1.0)
    bus.stop()
    dt.cancel()
    try:
        await dt
    except asyncio.CancelledError:
        pass

    info = mgr.get_task_info(task_id)
    assert info is not None
    assert info.status == "completed"
    assert "fast result" in info.result
