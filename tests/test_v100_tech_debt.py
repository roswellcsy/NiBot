"""Tests for v1.0.0 tech debt fixes: GitTool isolation + worktree diff."""
from __future__ import annotations
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from nibot.tools.git_tool import GitTool
from nibot.worktree import WorktreeManager


class TestGitToolIsolation:
    """GitTool with allowed_task_id blocks cross-task operations."""

    @pytest.mark.asyncio
    async def test_restricted_tool_blocks_other_task(self) -> None:
        mock_wt = AsyncMock(spec=WorktreeManager)
        tool = GitTool(mock_wt, allowed_task_id="abc123")
        result = await tool.execute(action="status", task_id="other_task")
        assert "access denied" in result.lower() or "denied" in result.lower()

    @pytest.mark.asyncio
    async def test_restricted_tool_allows_own_task(self) -> None:
        mock_wt = AsyncMock(spec=WorktreeManager)
        mock_wt.status = AsyncMock(return_value="clean")
        tool = GitTool(mock_wt, allowed_task_id="abc123")
        result = await tool.execute(action="status", task_id="abc123")
        assert "denied" not in result.lower()

    @pytest.mark.asyncio
    async def test_restricted_tool_auto_fills_task_id(self) -> None:
        mock_wt = AsyncMock(spec=WorktreeManager)
        mock_wt.status = AsyncMock(return_value="clean")
        tool = GitTool(mock_wt, allowed_task_id="abc123")
        result = await tool.execute(action="status")
        mock_wt.status.assert_called_once_with("abc123")

    @pytest.mark.asyncio
    async def test_unrestricted_tool_allows_any_task(self) -> None:
        mock_wt = AsyncMock(spec=WorktreeManager)
        mock_wt.status = AsyncMock(return_value="clean")
        tool = GitTool(mock_wt)  # no allowed_task_id
        result = await tool.execute(action="status", task_id="anything")
        assert "denied" not in result.lower()

    def test_isolated_registry_includes_git_tool(self) -> None:
        """SubagentManager._create_isolated_registry should create GitTool with task isolation."""
        from nibot.subagent import SubagentManager
        from nibot.registry import ToolRegistry
        from nibot.bus import MessageBus
        from nibot.config import AgentTypeConfig

        class FakeProvider:
            async def chat(self, **kw): pass

        bus = MessageBus()
        wt_mgr = MagicMock(spec=WorktreeManager)
        mgr = SubagentManager(FakeProvider(), ToolRegistry(), bus, worktree_mgr=wt_mgr)

        config = AgentTypeConfig(tools=["git", "file_read", "exec"])
        fake_path = Path(tempfile.mkdtemp())
        reg = mgr._create_isolated_registry(fake_path, config)

        assert reg.has("git"), "Isolated registry should include git tool"
        git_tool = reg._tools["git"]
        assert hasattr(git_tool, "_allowed_task_id")


class TestWorktreeDiffNoSideEffects:
    """diff() must not modify the git index."""

    @pytest.mark.asyncio
    async def test_diff_does_not_call_add_or_reset(self) -> None:
        """diff() should use 'git diff HEAD' and 'git ls-files', not 'git add -A'."""
        import inspect
        source = inspect.getsource(WorktreeManager.diff)
        assert "add" not in source or "ls-files" in source, (
            "diff() should not use 'git add'; use 'git diff HEAD' + 'git ls-files' instead"
        )
        # The key assertion: no "add", "-A" pattern in the method
        assert '"-A"' not in source, "diff() must not stage files with -A"

    @pytest.mark.asyncio
    async def test_diff_returns_tracked_and_untracked(self) -> None:
        """diff() should combine tracked diff and untracked file list."""
        wt = WorktreeManager(Path(tempfile.mkdtemp()))

        calls = []
        async def mock_git(*args, cwd=None):
            calls.append(args)
            if args[0] == "diff":
                return (0, " file.py | 3 +++\n 1 file changed\n", "")
            if args[0] == "ls-files":
                return (0, "new_file.py\n", "")
            return (0, "", "")

        wt._git = mock_git
        result = await wt.diff("test_task")

        assert "file.py" in result
        assert "new_file.py" in result or "untracked" in result.lower()
        # Verify no "add" command was called
        for call_args in calls:
            assert call_args[0] != "add", "diff() should not call git add"
