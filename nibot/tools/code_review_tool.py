"""Code review tool -- diff viewing and linter execution."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from nibot.registry import Tool

_MAX_OUTPUT = 50_000


class CodeReviewTool(Tool):
    """Review code changes (diff) or run linters on a directory."""

    def __init__(self, workspace: Path, worktree_mgr: Any = None) -> None:
        self._workspace = workspace
        self._worktree_mgr = worktree_mgr

    @property
    def name(self) -> str:
        return "code_review"

    @property
    def description(self) -> str:
        return "Review code changes (git diff) or run static analysis (ruff/pylint/flake8)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["review", "lint"],
                    "description": "review = show git diff; lint = run static analysis",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to review/lint (default: workspace)",
                },
                "task_id": {
                    "type": "string",
                    "description": "Worktree task ID for diff (review action only)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if action == "review":
            return await self._review(kwargs)
        if action == "lint":
            return await self._lint(kwargs)
        return f"Error: unknown action '{action}'. Use 'review' or 'lint'."

    async def _review(self, kwargs: dict[str, Any]) -> str:
        task_id = kwargs.get("task_id", "")
        if task_id and self._worktree_mgr:
            diff = await self._worktree_mgr.diff(task_id)
            if not diff.strip():
                return "No changes in worktree."
            return self._truncate(diff)
        target = Path(kwargs.get("path", str(self._workspace)))
        return await self._run_cmd(["git", "diff", "HEAD"], cwd=target)

    async def _lint(self, kwargs: dict[str, Any]) -> str:
        target = Path(kwargs.get("path", str(self._workspace)))
        for linter in ["ruff", "pylint", "flake8"]:
            if await self._linter_available(linter):
                if linter == "ruff":
                    cmd = ["ruff", "check", str(target)]
                elif linter == "pylint":
                    cmd = ["pylint", str(target), "--output-format=text"]
                else:
                    cmd = ["flake8", str(target)]
                return await self._run_cmd(cmd, cwd=target)
        return "No linter available. Install ruff, pylint, or flake8."

    async def _linter_available(self, name: str) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                name, "--version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5)
            return proc.returncode == 0
        except (FileNotFoundError, asyncio.TimeoutError, OSError):
            return False

    async def _run_cmd(self, cmd: list[str], cwd: Path) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            return "Command timed out after 60s."
        except Exception as e:
            return f"Error: {e}"
        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        parts.append(f"[exit={proc.returncode}]")
        return self._truncate("\n".join(parts))

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) > _MAX_OUTPUT:
            return text[:_MAX_OUTPUT] + "\n... (truncated)"
        return text
