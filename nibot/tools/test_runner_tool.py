"""Test runner tool -- auto-detect framework, run tests, collect results."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from nibot.registry import Tool

_MAX_OUTPUT = 50_000


class TestRunnerTool(Tool):  # noqa: N801 -- not a test class
    __test__ = False  # Tell pytest to skip collection
    """Run tests and collect results. Auto-detects pytest/jest/unittest."""

    def __init__(self, workspace: Path, timeout: int = 120) -> None:
        self._workspace = workspace
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "test_runner"

    @property
    def description(self) -> str:
        return "Run tests in a project directory. Auto-detects pytest, jest, or unittest."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run", "coverage"],
                    "description": "run = execute tests; coverage = run with coverage report",
                },
                "path": {
                    "type": "string",
                    "description": "Project directory (default: workspace)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Test file or pattern (e.g., 'tests/test_foo.py')",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if action not in ("run", "coverage"):
            return f"Error: unknown action '{action}'. Use 'run' or 'coverage'."
        target = Path(kwargs.get("path", str(self._workspace)))
        pattern = kwargs.get("pattern", "")
        timeout = kwargs.get("timeout", self._timeout)
        coverage = action == "coverage"
        framework = self._detect_framework(target)
        cmd = self._build_command(framework, target, pattern, coverage)
        return await self._run(cmd, cwd=target, timeout=timeout, framework=framework)

    def _detect_framework(self, path: Path) -> str:
        if (path / "conftest.py").exists():
            return "pytest"
        pyproject = path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                if "[tool.pytest" in content:
                    return "pytest"
            except OSError:
                pass
        pkg_json = path / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text(encoding="utf-8"))
                deps = {**data.get("devDependencies", {}), **data.get("dependencies", {})}
                if "vitest" in deps:
                    return "vitest"
                if "jest" in deps:
                    return "jest"
            except (json.JSONDecodeError, OSError):
                pass
        return "unittest"

    def _build_command(
        self, framework: str, path: Path, pattern: str, coverage: bool,
    ) -> list[str]:
        if framework == "pytest":
            cmd = ["python", "-m", "pytest", "-v"]
            if coverage:
                cmd.append("--cov")
            if pattern:
                cmd.append(pattern)
            return cmd
        if framework in ("jest", "vitest"):
            cmd = ["npx", framework]
            if coverage:
                cmd.append("--coverage")
            if pattern:
                cmd.append(pattern)
            return cmd
        # unittest fallback
        cmd = ["python", "-m", "unittest", "discover"]
        if pattern:
            cmd.extend(["-p", pattern])
        if coverage:
            return ["python", "-m", "coverage", "run", "-m", "unittest", "discover"]
        return cmd

    async def _run(
        self, cmd: list[str], cwd: Path, timeout: int, framework: str,
    ) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            return f"Tests timed out after {timeout}s."
        except FileNotFoundError:
            return f"Error: '{cmd[0]}' not found. Is {framework} installed?"
        except Exception as e:
            return f"Error running tests: {e}"
        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        parts = [f"[framework={framework}]"]
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        parts.append(f"[exit={proc.returncode}]")
        result = "\n".join(parts)
        if len(result) > _MAX_OUTPUT:
            result = result[:_MAX_OUTPUT] + "\n... (truncated)"
        return result
