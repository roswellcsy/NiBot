"""Exec tool tests -- dangerous pattern blocking, sandbox, timeout."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nibot.sandbox import SandboxConfig, _build_limited_command, _sanitize_env, sandboxed_exec
from nibot.tools.exec_tool import DANGEROUS_PATTERNS, ExecTool


# ---------------------------------------------------------------------------
# DANGEROUS_PATTERNS blocking
# ---------------------------------------------------------------------------

class TestDangerousPatterns:
    """Each pattern in DANGEROUS_PATTERNS must block the corresponding command."""

    @pytest.mark.asyncio
    async def test_blocks_rm_rf(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path)
        result = await tool.execute(command="rm -rf /")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_blocks_rm_fr(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path)
        result = await tool.execute(command="rm -fr /tmp/important")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_blocks_rm_force_recursive(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path)
        result = await tool.execute(command="rm --force -r /data")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_blocks_dd(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path)
        result = await tool.execute(command="dd if=/dev/zero of=/dev/sda")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_blocks_shutdown(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path)
        result = await tool.execute(command="shutdown -h now")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_blocks_reboot(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path)
        result = await tool.execute(command="reboot")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_blocks_mkfs(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path)
        result = await tool.execute(command="mkfs.ext4 /dev/sda1")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_blocks_fork_bomb(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path)
        result = await tool.execute(command=":(){ :|:& };:")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_blocks_chmod_777(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path)
        result = await tool.execute(command="chmod -R 777 /")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_blocks_write_to_dev_word_boundary(self, tmp_path: Path) -> None:
        r"""Pattern \b>\s*/dev/sd requires word char before >."""
        tool = ExecTool(workspace=tmp_path)
        # The pattern uses \b which requires a word-char boundary before >
        result = await tool.execute(command="cat>/dev/sda")
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_allows_safe_ls(self, tmp_path: Path) -> None:
        """Legitimate commands should NOT be blocked."""
        tool = ExecTool(workspace=tmp_path, sandbox_enabled=False)
        result = await tool.execute(command="echo hello")
        assert "Blocked" not in result

    @pytest.mark.asyncio
    async def test_allows_rm_without_rf(self, tmp_path: Path) -> None:
        """rm without -rf should not be blocked."""
        tool = ExecTool(workspace=tmp_path, sandbox_enabled=False)
        # Just check it's not pattern-blocked (may fail for other reasons)
        result = await tool.execute(command="rm nonexistent_file_xyz 2>/dev/null; echo done")
        assert "Blocked" not in result

    @pytest.mark.asyncio
    async def test_blocks_format(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path)
        result = await tool.execute(command="format C:")
        assert "Blocked" in result


# ---------------------------------------------------------------------------
# Sandbox functions
# ---------------------------------------------------------------------------

class TestSandboxConfig:

    def test_default_config(self) -> None:
        cfg = SandboxConfig()
        assert cfg.timeout == 60
        assert cfg.memory_mb == 512
        assert cfg.max_output == 50000
        assert cfg.enabled is True

    def test_sanitize_env_filters(self) -> None:
        env = _sanitize_env(["PATH", "HOME", "NONEXISTENT_VAR_XYZ"])
        assert "PATH" in env or "HOME" in env  # at least one should exist
        assert "NONEXISTENT_VAR_XYZ" not in env
        assert env.get("PYTHONIOENCODING") == "utf-8"

    def test_build_limited_command_unix(self) -> None:
        cfg = SandboxConfig(memory_mb=256)
        if sys.platform != "win32":
            result = _build_limited_command("echo hi", cfg)
            assert "ulimit" in result
            assert "echo hi" in result
        else:
            result = _build_limited_command("echo hi", cfg)
            assert result == "echo hi"  # no ulimit on Windows

    def test_build_limited_command_disabled(self) -> None:
        cfg = SandboxConfig(enabled=False)
        result = _build_limited_command("echo hi", cfg)
        assert result == "echo hi"


# ---------------------------------------------------------------------------
# sandboxed_exec
# ---------------------------------------------------------------------------

class TestSandboxedExec:

    @pytest.mark.asyncio
    async def test_simple_echo(self, tmp_path: Path) -> None:
        cfg = SandboxConfig(enabled=False)
        out, err, rc = await sandboxed_exec("echo hello", tmp_path, cfg)
        assert "hello" in out
        assert rc == 0

    @pytest.mark.asyncio
    async def test_nonzero_exit(self, tmp_path: Path) -> None:
        cfg = SandboxConfig(enabled=False)
        out, err, rc = await sandboxed_exec("exit 42", tmp_path, cfg)
        assert rc == 42

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path: Path) -> None:
        cfg = SandboxConfig(timeout=1, enabled=False)
        cmd = "sleep 30" if sys.platform != "win32" else "ping -n 30 127.0.0.1"
        out, err, rc = await sandboxed_exec(cmd, tmp_path, cfg)
        assert "Timed out" in err
        assert rc == -1

    @pytest.mark.asyncio
    async def test_stdout_truncation(self, tmp_path: Path) -> None:
        cfg = SandboxConfig(max_output=100, enabled=False)
        cmd = f'{sys.executable} -c "print(\'x\' * 500)"'
        out, err, rc = await sandboxed_exec(cmd, tmp_path, cfg)
        assert "truncated" in out

    @pytest.mark.asyncio
    async def test_stderr_capture(self, tmp_path: Path) -> None:
        cfg = SandboxConfig(enabled=False)
        cmd = f'{sys.executable} -c "import sys; sys.stderr.write(\'oops\')"'
        out, err, rc = await sandboxed_exec(cmd, tmp_path, cfg)
        assert "oops" in err


# ---------------------------------------------------------------------------
# ExecTool integration
# ---------------------------------------------------------------------------

class TestExecToolIntegration:

    @pytest.mark.asyncio
    async def test_exec_echo(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path, sandbox_enabled=False)
        result = await tool.execute(command="echo integration_test")
        assert "integration_test" in result
        assert "[exit=0]" in result

    @pytest.mark.asyncio
    async def test_exec_nonzero_exit(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path, sandbox_enabled=False)
        result = await tool.execute(command="exit 1")
        assert "[exit=1]" in result

    @pytest.mark.asyncio
    async def test_exec_timeout_from_kwargs(self, tmp_path: Path) -> None:
        tool = ExecTool(workspace=tmp_path, sandbox_enabled=False)
        cmd = "sleep 30" if sys.platform != "win32" else "ping -n 30 127.0.0.1"
        result = await tool.execute(command=cmd, timeout=1)
        assert "Timed out" in result
