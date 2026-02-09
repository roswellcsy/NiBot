"""Shell execution tool with safety guardrails and optional sandboxing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nibot.registry import Tool
from nibot.sandbox import SandboxConfig, sandboxed_exec

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
    def __init__(self, workspace: Path, timeout: int = 60,
                 sandbox_enabled: bool = True, sandbox_memory_mb: int = 512) -> None:
        self._workspace = workspace
        self._timeout = timeout
        self._sandbox_enabled = sandbox_enabled
        self._sandbox_memory_mb = sandbox_memory_mb

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

        cfg = SandboxConfig(
            timeout=timeout,
            memory_mb=self._sandbox_memory_mb,
            enabled=self._sandbox_enabled,
        )
        out, err, rc = await sandboxed_exec(command, self._workspace, cfg)

        if rc == -1 and ("Timed out" in err or "Exec error" in err):
            return err

        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        parts.append(f"[exit={rc}]")
        return "\n".join(parts)
