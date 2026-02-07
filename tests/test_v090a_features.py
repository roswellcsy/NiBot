"""Tests for v0.9.0a features: stability hardening."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.session import Session, SessionManager


# ---- helpers ----

def _make_session(sm: SessionManager, key: str, content: str = "hello") -> Session:
    """Create and save a minimal session."""
    s = Session(key=key)
    s.add_message("user", content)
    sm.save(s)
    return s


# ---- A1: Bounded Session Cache ----

class TestBoundedSessionCache:
    def test_cache_eviction_lru(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path, max_cache_size=3)
        _make_session(sm, "s1")
        _make_session(sm, "s2")
        _make_session(sm, "s3")
        assert len(sm._cache) == 3

        # Adding s4 should evict s1 (oldest)
        _make_session(sm, "s4")
        assert len(sm._cache) == 3
        assert "s1" not in sm._cache
        assert "s4" in sm._cache

    def test_cache_eviction_preserves_locks(self, tmp_path: Path) -> None:
        """Locks must NOT be evicted with cache entries -- they may be held by coroutines."""
        sm = SessionManager(tmp_path, max_cache_size=2)
        lock = sm.lock_for("s1")
        _make_session(sm, "s1")
        _make_session(sm, "s2")

        # s3 evicts s1 from cache, but lock must survive
        _make_session(sm, "s3")
        assert "s1" not in sm._cache
        assert "s1" in sm._locks
        assert sm.lock_for("s1") is lock  # same lock object returned

    def test_cache_move_to_end_on_access(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path, max_cache_size=3)
        _make_session(sm, "s1")
        _make_session(sm, "s2")
        _make_session(sm, "s3")

        # Access s1 to make it "recent"
        sm.get_or_create("s1")

        # Now s2 should be evicted (oldest unused)
        _make_session(sm, "s4")
        assert "s1" in sm._cache
        assert "s2" not in sm._cache

    def test_iter_recent_from_disk_no_cache_pollution(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path, max_cache_size=2)
        _make_session(sm, "s1")
        _make_session(sm, "s2")
        # Clear cache to simulate cold start
        sm._cache.clear()

        sessions = sm.iter_recent_from_disk(limit=10)
        assert len(sessions) == 2
        # iter should NOT add to cache
        assert len(sm._cache) == 0

    def test_iter_recent_from_disk_returns_cached(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path, max_cache_size=10)
        s1 = _make_session(sm, "s1")
        # s1 is in cache
        sessions = sm.iter_recent_from_disk(limit=10)
        assert len(sessions) == 1
        assert sessions[0] is s1  # same object from cache

    def test_max_cache_size_default(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path)
        assert sm._max_cache_size == 200

    def test_load_all_respects_cache_limit(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path, max_cache_size=3)
        for i in range(5):
            _make_session(sm, f"s{i}")
        sm._cache.clear()

        sm._load_all()
        assert len(sm._cache) <= 3


# ---- A2: Bounded TaskInfo ----

class TestBoundedTaskInfo:
    def test_prune_removes_old_completed(self) -> None:
        from nibot.subagent import SubagentManager, TaskInfo

        sm = SubagentManager(MagicMock(), MagicMock(), MagicMock())
        sm._max_task_history = 5
        now = datetime.now()

        # Add 10 completed tasks, 6 older than 1 hour
        for i in range(10):
            info = TaskInfo(
                task_id=f"t{i}", agent_type="test", label=f"task {i}",
                status="completed",
                created_at=now - timedelta(hours=2, minutes=i),
                finished_at=now - timedelta(hours=1, minutes=30 - i),
            )
            sm._task_info[f"t{i}"] = info

        sm._prune_task_info()
        assert len(sm._task_info) <= 5

    def test_prune_keeps_running(self) -> None:
        from nibot.subagent import SubagentManager, TaskInfo

        sm = SubagentManager(MagicMock(), MagicMock(), MagicMock())
        sm._max_task_history = 2
        now = datetime.now()

        # Add 3 running tasks + 2 completed
        for i in range(3):
            sm._task_info[f"running{i}"] = TaskInfo(
                task_id=f"running{i}", agent_type="test", label="r",
                status="running", created_at=now - timedelta(hours=3),
            )
        for i in range(2):
            sm._task_info[f"done{i}"] = TaskInfo(
                task_id=f"done{i}", agent_type="test", label="d",
                status="completed",
                created_at=now - timedelta(hours=2),
                finished_at=now - timedelta(hours=1, minutes=30),
            )

        sm._prune_task_info()
        # Running tasks should NOT be pruned
        for i in range(3):
            assert f"running{i}" in sm._task_info

    def test_prune_no_op_under_limit(self) -> None:
        from nibot.subagent import SubagentManager, TaskInfo

        sm = SubagentManager(MagicMock(), MagicMock(), MagicMock())
        sm._max_task_history = 200
        sm._task_info["t1"] = TaskInfo(
            task_id="t1", agent_type="test", label="task",
            status="completed", finished_at=datetime.now(),
        )
        sm._prune_task_info()
        assert "t1" in sm._task_info


# ---- A3: Fire-and-Forget Exception Logging ----

class TestFireAndForgetSafety:
    def test_log_task_exception_logs_error(self) -> None:
        from nibot.agent import _log_task_exception

        # Create a task that raised an exception
        loop = asyncio.new_event_loop()
        try:
            async def _fail():
                raise ValueError("test error")

            task = loop.create_task(_fail())
            loop.run_until_complete(asyncio.sleep(0.01))
        finally:
            loop.close()

        # The callback should log the exception without raising
        with patch("nibot.agent.logger") as mock_logger:
            _log_task_exception(task)
            mock_logger.error.assert_called_once()
            assert "test error" in mock_logger.error.call_args[0][0]

    def test_log_task_exception_ignores_cancelled(self) -> None:
        from nibot.agent import _log_task_exception

        task = MagicMock()
        task.cancelled.return_value = True
        # Should not call exception() if cancelled
        _log_task_exception(task)
        task.exception.assert_not_called()

    def test_subagent_task_done_updates_status(self) -> None:
        from nibot.subagent import SubagentManager, TaskInfo

        sm = SubagentManager(MagicMock(), MagicMock(), MagicMock())
        sm._task_info["t1"] = TaskInfo(
            task_id="t1", agent_type="test", label="task", status="running",
        )
        sm._tasks["t1"] = MagicMock()

        # Simulate a task that crashed
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = RuntimeError("boom")

        sm._task_done(mock_task, "t1")

        assert sm._task_info["t1"].status == "error"
        assert "boom" in sm._task_info["t1"].result
        assert "t1" not in sm._tasks


# ---- A4: Graceful Shutdown ----

class TestGracefulShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_calls_stop_on_all(self, tmp_path: Path) -> None:
        """Verify _shutdown stops all components."""
        from nibot.app import NiBot

        with patch("nibot.app.load_config") as mock_config, \
             patch("nibot.app.NiBot._create_provider") as mock_prov, \
             patch("nibot.app.NiBot._register_builtin_tools"):
            cfg = MagicMock()
            cfg.agent.bus_queue_maxsize = 0
            cfg.agent.workspace = str(tmp_path)
            cfg.agent.model = "test"
            cfg.agent.auto_evolution = False
            cfg.agent.bootstrap_files = []
            cfg.agent.context_window = 128000
            cfg.agent.context_reserve = 4096
            cfg.agents = {}
            cfg.schedules = []
            cfg.providers = MagicMock()
            cfg.tools = MagicMock()
            cfg.tools.restrict_to_workspace = True
            mock_config.return_value = cfg
            mock_prov.return_value = MagicMock()

            app = NiBot.__new__(NiBot)
            app.config = cfg
            app.bus = MagicMock()
            app.agent = MagicMock()
            app.agent._tasks = set()
            app.scheduler = MagicMock()
            app.subagents = MagicMock()
            app.subagents._tasks = {}
            app._channels = []

            # Mock channel
            ch = MagicMock()
            ch.stop = AsyncMock()
            app._channels = [ch]

            await app._shutdown([])

            app.agent.stop.assert_called_once()
            app.bus.stop.assert_called_once()
            app.scheduler.stop.assert_called_once()
            ch.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_remaining_tasks(self) -> None:
        """Verify lingering tasks get cancelled."""
        from nibot.app import NiBot

        app = NiBot.__new__(NiBot)
        app.agent = MagicMock()
        app.agent._tasks = set()
        app.bus = MagicMock()
        app.scheduler = MagicMock()
        app.subagents = MagicMock()
        app.subagents._tasks = {}
        app._channels = []

        # Create a real asyncio task that we can cancel
        async def _hang():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_hang())
        await app._shutdown([task])
        assert task.cancelled()


# ---- A5: Cleanup verification ----

class TestToolCtxCleanup:
    @pytest.mark.asyncio
    async def test_write_thought_no_tool_ctx(self, tmp_path: Path) -> None:
        """_WriteThoughtTool.execute should work without _tool_ctx handling."""
        from nibot.subagent import _WriteThoughtTool

        tool = _WriteThoughtTool(tmp_path)
        result = await tool.execute(filename="test", content="hello world")
        assert "written" in result.lower()
        assert (tmp_path / "thoughts" / "test.md").read_text(encoding="utf-8") == "hello world"

    @pytest.mark.asyncio
    async def test_analyze_tool_no_tool_ctx(self, tmp_path: Path) -> None:
        """AnalyzeTool.execute should work without _tool_ctx handling."""
        from nibot.tools.analyze_tool import AnalyzeTool

        sm = SessionManager(tmp_path / "sessions")
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="summary")
        assert "no sessions" in result.lower() or "sessions" in result.lower()
