"""Shell execution tool with safety guardrails."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from nibot.registry import Tool

DANGEROUS_PATTERNS = [
    r"\brm\s+(-\w*r\w*f|--force|-rf|-fr)\b",
    r"\bformat\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r":\(\)\{.*\}",  # fork bomb
    r"\bchmod\s+-R\s+777\b",
    r"\b>\s*/dev/sd",
]


class ExecTool(Tool):
    def __init__(self, workspace: Path, timeout: int = 60) -> None:
        self._workspace = workspace
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command in the workspace directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds"},
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs["command"]
        timeout = kwargs.get("timeout", self._timeout)

        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return f"Blocked: command matches dangerous pattern ({pattern})"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._workspace),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Exec error: {e}"

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        parts.append(f"[exit={proc.returncode}]")
        result = "\n".join(parts)
        if len(result) > 50000:
            result = result[:50000] + "\n... (truncated)"
        return result
