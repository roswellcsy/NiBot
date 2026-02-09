"""Sandboxed subprocess execution with resource limits and environment sanitization."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SandboxConfig:
    """Resource limits for sandboxed execution."""

    timeout: int = 60
    memory_mb: int = 512
    max_output: int = 50000
    allowed_env: list[str] = field(
        default_factory=lambda: ["PATH", "HOME", "LANG", "TERM", "USER", "SHELL",
                                 "PYTHONPATH", "PYTHONIOENCODING"]
    )
    enabled: bool = True


def _sanitize_env(allowed: list[str]) -> dict[str, str]:
    """Build a minimal environment from allowed variable names."""
    env = {}
    for key in allowed:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    # Always ensure UTF-8 encoding
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _build_limited_command(command: str, config: SandboxConfig) -> str:
    """Prepend ulimit resource limits on Unix platforms."""
    if sys.platform == "win32" or not config.enabled:
        return command
    # ulimit -v: virtual memory in KB; -f: max file size in 512-byte blocks
    limits = (
        f"ulimit -v {config.memory_mb * 1024} 2>/dev/null; "
        f"ulimit -f {config.memory_mb * 2} 2>/dev/null; "
    )
    return limits + command


async def sandboxed_exec(
    command: str,
    cwd: Path,
    config: SandboxConfig | None = None,
    stdin_data: bytes = b"",
) -> tuple[str, str, int]:
    """Execute a shell command with resource limits and environment sanitization.

    Returns (stdout, stderr, returncode).
    """
    cfg = config or SandboxConfig()
    env = _sanitize_env(cfg.allowed_env) if cfg.enabled else None
    wrapped = _build_limited_command(command, cfg)

    try:
        proc = await asyncio.create_subprocess_shell(
            wrapped,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if stdin_data else None,
            cwd=str(cwd),
            env=env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(stdin_data if stdin_data else None),
            timeout=cfg.timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        return "", f"Timed out after {cfg.timeout}s", -1
    except Exception as e:
        return "", f"Exec error: {e}", -1

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

    if len(stdout) > cfg.max_output:
        stdout = stdout[:cfg.max_output] + "\n... (truncated)"
    if len(stderr) > 10000:
        stderr = stderr[:10000] + "\n... (truncated)"

    return stdout, stderr, proc.returncode


async def sandboxed_exec_py(
    script_path: Path,
    cwd: Path,
    config: SandboxConfig | None = None,
    stdin_data: bytes = b"",
) -> tuple[str, str, int]:
    """Execute a Python script in a sandboxed subprocess.

    Returns (stdout, stderr, returncode).
    """
    cfg = config or SandboxConfig()
    env = _sanitize_env(cfg.allowed_env) if cfg.enabled else None

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(stdin_data),
            timeout=cfg.timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        return "", f"Timed out after {cfg.timeout}s", -1
    except Exception as e:
        return "", f"Exec error: {e}", -1

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

    if len(stdout) > cfg.max_output:
        stdout = stdout[:cfg.max_output] + "\n... (truncated)"
    if len(stderr) > 10000:
        stderr = stderr[:10000] + "\n... (truncated)"

    return stdout, stderr, proc.returncode
