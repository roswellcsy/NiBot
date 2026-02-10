"""Auto-compact tests (Phase 3)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nibot.compact import compact_messages
from nibot.config import NiBotConfig
from nibot.context import ContextBuilder
from nibot.memory import MemoryStore
from nibot.provider import LLMProvider, LLMResponse
from nibot.session import Session, SessionManager
from nibot.skills import SkillsLoader
from nibot.types import Envelope


class _SummaryProvider(LLMProvider):
    """Returns a fixed summary."""
    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None, **kw: Any) -> LLMResponse:
        return LLMResponse(content="Summary: user discussed X and Y.")


class _FailProvider(LLMProvider):
    """Always raises."""
    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None, **kw: Any) -> LLMResponse:
        raise RuntimeError("provider down")


@pytest.mark.asyncio
async def test_compact_messages_returns_summary():
    """compact_messages() calls provider and returns summary string."""
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = await compact_messages(msgs, _SummaryProvider())
    assert "Summary" in result
    assert "X and Y" in result


@pytest.mark.asyncio
async def test_compact_messages_empty_input():
    """Empty messages produce empty summary."""
    result = await compact_messages([], _SummaryProvider())
    assert result == ""


@pytest.mark.asyncio
async def test_compact_messages_provider_failure():
    """Provider failure returns empty string, no crash."""
    msgs = [{"role": "user", "content": "test"}]
    result = await compact_messages(msgs, _FailProvider())
    assert result == ""


def test_build_injects_compacted_summary(tmp_path: Path):
    """When session has compacted_summary, it appears in built messages."""
    cfg = NiBotConfig()
    mem = MemoryStore(tmp_path / "mem")
    skills = SkillsLoader([])
    builder = ContextBuilder(config=cfg, memory=mem, skills=skills, workspace=tmp_path)

    session = Session(key="test")
    session.compacted_summary = "Earlier: discussed deployment plan."
    session.add_message("user", "old msg")

    envelope = Envelope(channel="test", chat_id="c1", sender_id="u1", content="new msg")
    messages = builder.build(session=session, current=envelope)

    # Find the summary injection
    summaries = [m for m in messages if "Earlier conversation summary" in (m.get("content") or "")]
    assert len(summaries) == 1
    assert "deployment plan" in summaries[0]["content"]


def test_build_no_summary_when_empty(tmp_path: Path):
    """No summary injection when compacted_summary is empty."""
    cfg = NiBotConfig()
    mem = MemoryStore(tmp_path / "mem")
    skills = SkillsLoader([])
    builder = ContextBuilder(config=cfg, memory=mem, skills=skills, workspace=tmp_path)

    session = Session(key="test")
    envelope = Envelope(channel="test", chat_id="c1", sender_id="u1", content="hi")
    messages = builder.build(session=session, current=envelope)

    summaries = [m for m in messages if "Earlier conversation summary" in (m.get("content") or "")]
    assert len(summaries) == 0


def test_session_compacted_summary_persistence(tmp_path: Path):
    """compacted_summary survives save/load cycle."""
    mgr = SessionManager(tmp_path / "sessions")
    session = mgr.get_or_create("test-persist")
    session.add_message("user", "hello")
    session.compacted_summary = "Summary of prior conversation."
    mgr.save(session)

    # Force reload from disk
    mgr._cache.clear()
    loaded = mgr.get_or_create("test-persist")
    assert loaded.compacted_summary == "Summary of prior conversation."


@pytest.mark.asyncio
async def test_compact_dedup_no_double_schedule(tmp_path: Path):
    """Same session called twice in build() should only schedule one compact task."""
    cfg = NiBotConfig()
    # Set tiny context window to force message dropping
    cfg.agent.context_window = 500
    cfg.agent.context_reserve = 100
    mem = MemoryStore(tmp_path / "mem")
    skills = SkillsLoader([])
    provider = _SummaryProvider()
    sessions_mgr = SessionManager(tmp_path / "sessions")
    builder = ContextBuilder(
        config=cfg, memory=mem, skills=skills, workspace=tmp_path,
        provider=provider, sessions=sessions_mgr,
    )

    session = Session(key="dedup-test")
    # Add enough messages to exceed context window and trigger dropping
    for i in range(30):
        session.add_message("user", f"message {i} " * 20)
        session.add_message("assistant", f"reply {i} " * 20)

    envelope = Envelope(channel="test", chat_id="c1", sender_id="u1", content="new msg")

    # First build triggers compact
    builder.build(session=session, current=envelope)
    assert "dedup-test" in builder._compacting_sessions

    # Second build should NOT create another compact task
    initial_task_count = len(builder._compact_tasks)
    builder.build(session=session, current=envelope)
    assert len(builder._compact_tasks) == initial_task_count  # no new task
