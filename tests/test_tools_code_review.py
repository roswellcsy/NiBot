"""Code review and test runner tool tests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.tools.code_review_tool import CodeReviewTool
from nibot.tools.test_runner_tool import TestRunnerTool


# ---------------------------------------------------------------------------
# CodeReviewTool
# ---------------------------------------------------------------------------

class TestCodeReviewTool:

    @pytest.mark.asyncio
    async def test_unknown_action(self, tmp_path: Path) -> None:
        tool = CodeReviewTool(workspace=tmp_path)
        result = await tool.execute(action="invalid")
        assert "unknown action" in result.lower()

    @pytest.mark.asyncio
    async def test_review_runs_git_diff(self, tmp_path: Path) -> None:
        tool = CodeReviewTool(workspace=tmp_path)
        with patch.object(tool, "_run_cmd", new_callable=AsyncMock, return_value="diff output\n[exit=0]"):
            result = await tool.execute(action="review")
        assert "diff output" in result

    @pytest.mark.asyncio
    async def test_review_with_worktree(self, tmp_path: Path) -> None:
        mock_wt = MagicMock()
        mock_wt.diff = AsyncMock(return_value="worktree diff")
        tool = CodeReviewTool(workspace=tmp_path, worktree_mgr=mock_wt)
        result = await tool.execute(action="review", task_id="task-1")
        assert "worktree diff" in result

    @pytest.mark.asyncio
    async def test_review_worktree_empty_diff(self, tmp_path: Path) -> None:
        mock_wt = MagicMock()
        mock_wt.diff = AsyncMock(return_value="  \n")
        tool = CodeReviewTool(workspace=tmp_path, worktree_mgr=mock_wt)
        result = await tool.execute(action="review", task_id="task-1")
        assert "No changes" in result

    @pytest.mark.asyncio
    async def test_lint_no_linter(self, tmp_path: Path) -> None:
        tool = CodeReviewTool(workspace=tmp_path)
        with patch.object(tool, "_linter_available", new_callable=AsyncMock, return_value=False):
            result = await tool.execute(action="lint")
        assert "No linter available" in result

    @pytest.mark.asyncio
    async def test_lint_uses_first_available(self, tmp_path: Path) -> None:
        tool = CodeReviewTool(workspace=tmp_path)

        async def avail(name: str) -> bool:
            return name == "ruff"

        with (
            patch.object(tool, "_linter_available", side_effect=avail),
            patch.object(tool, "_run_cmd", new_callable=AsyncMock, return_value="All clean\n[exit=0]"),
        ):
            result = await tool.execute(action="lint")
        assert "All clean" in result

    def test_truncate_long_output(self) -> None:
        big = "x" * 60000
        result = CodeReviewTool._truncate(big)
        assert len(result) <= 50020  # 50000 + "... (truncated)"
        assert "truncated" in result

    def test_truncate_short_output(self) -> None:
        small = "short"
        assert CodeReviewTool._truncate(small) == small


# ---------------------------------------------------------------------------
# TestRunnerTool
# ---------------------------------------------------------------------------

class TestTestRunnerTool:

    def test_detect_pytest_conftest(self, tmp_path: Path) -> None:
        (tmp_path / "conftest.py").touch()
        tool = TestRunnerTool(workspace=tmp_path)
        assert tool._detect_framework(tmp_path) == "pytest"

    def test_detect_pytest_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
        tool = TestRunnerTool(workspace=tmp_path)
        assert tool._detect_framework(tmp_path) == "pytest"

    def test_detect_jest(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            json.dumps({"devDependencies": {"jest": "^29"}}), encoding="utf-8"
        )
        tool = TestRunnerTool(workspace=tmp_path)
        assert tool._detect_framework(tmp_path) == "jest"

    def test_detect_vitest_over_jest(self, tmp_path: Path) -> None:
        """vitest takes priority over jest."""
        (tmp_path / "package.json").write_text(
            json.dumps({"devDependencies": {"jest": "^29", "vitest": "^1"}}), encoding="utf-8"
        )
        tool = TestRunnerTool(workspace=tmp_path)
        assert tool._detect_framework(tmp_path) == "vitest"

    def test_detect_unittest_fallback(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(workspace=tmp_path)
        assert tool._detect_framework(tmp_path) == "unittest"

    def test_build_command_pytest(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(workspace=tmp_path)
        cmd = tool._build_command("pytest", tmp_path, "", False)
        assert cmd == ["python", "-m", "pytest", "-v"]

    def test_build_command_pytest_coverage(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(workspace=tmp_path)
        cmd = tool._build_command("pytest", tmp_path, "", True)
        assert "--cov" in cmd

    def test_build_command_pytest_pattern(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(workspace=tmp_path)
        cmd = tool._build_command("pytest", tmp_path, "tests/test_foo.py", False)
        assert "tests/test_foo.py" in cmd

    def test_build_command_jest(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(workspace=tmp_path)
        cmd = tool._build_command("jest", tmp_path, "", False)
        assert cmd == ["npx", "jest"]

    def test_build_command_unittest_coverage(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(workspace=tmp_path)
        cmd = tool._build_command("unittest", tmp_path, "", True)
        assert "coverage" in cmd

    @pytest.mark.asyncio
    async def test_unknown_action(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(workspace=tmp_path)
        result = await tool.execute(action="invalid")
        assert "unknown action" in result.lower()

    @pytest.mark.asyncio
    async def test_run_timeout(self, tmp_path: Path) -> None:
        tool = TestRunnerTool(workspace=tmp_path, timeout=1)
        with patch.object(tool, "_run", new_callable=AsyncMock, return_value="Tests timed out after 1s."):
            result = await tool.execute(action="run", timeout=1)
        assert "timed out" in result.lower()
