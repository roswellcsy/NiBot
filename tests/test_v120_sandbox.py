"""v1.2 Sandbox: resource limits, env sanitization, timeout handling."""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from nibot.sandbox import SandboxConfig, _sanitize_env, _build_limited_command, sandboxed_exec, sandboxed_exec_py


# ---- SandboxConfig ----

class TestSandboxConfig:

    def test_defaults(self) -> None:
        cfg = SandboxConfig()
        assert cfg.timeout == 60
        assert cfg.memory_mb == 512
        assert cfg.max_output == 50000
        assert cfg.enabled is True
        assert "PATH" in cfg.allowed_env

    def test_custom_values(self) -> None:
        cfg = SandboxConfig(timeout=10, memory_mb=256, enabled=False)
        assert cfg.timeout == 10
        assert cfg.memory_mb == 256
        assert cfg.enabled is False


# ---- Environment sanitization ----

class TestSanitizeEnv:

    def test_only_allowed_vars_pass_through(self) -> None:
        env = _sanitize_env(["PATH", "HOME"])
        # PATH should always exist
        assert "PATH" in env
        # Random vars should not leak
        assert "ANTHROPIC_API_KEY" not in env
        assert "SECRET_TOKEN" not in env

    def test_utf8_encoding_always_set(self) -> None:
        env = _sanitize_env(["PATH"])
        assert env.get("PYTHONIOENCODING") == "utf-8"

    def test_missing_var_not_in_result(self) -> None:
        env = _sanitize_env(["NONEXISTENT_VAR_12345"])
        assert "NONEXISTENT_VAR_12345" not in env


# ---- Command wrapping ----

class TestBuildLimitedCommand:

    def test_disabled_sandbox_passes_through(self) -> None:
        cfg = SandboxConfig(enabled=False)
        assert _build_limited_command("echo hi", cfg) == "echo hi"

    @pytest.mark.skipif(sys.platform == "win32", reason="ulimit only on Unix")
    def test_unix_adds_ulimit(self) -> None:
        cfg = SandboxConfig(enabled=True, memory_mb=256)
        result = _build_limited_command("echo hi", cfg)
        assert "ulimit" in result
        assert "echo hi" in result
        assert str(256 * 1024) in result

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows path")
    def test_windows_passes_through(self) -> None:
        cfg = SandboxConfig(enabled=True)
        assert _build_limited_command("echo hi", cfg) == "echo hi"


# ---- sandboxed_exec ----

class TestSandboxedExec:

    @pytest.mark.asyncio
    async def test_simple_command(self, tmp_path: Path) -> None:
        out, err, rc = await sandboxed_exec("echo hello", tmp_path)
        assert "hello" in out
        assert rc == 0

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path: Path) -> None:
        cfg = SandboxConfig(timeout=1)
        if sys.platform == "win32":
            cmd = "ping -n 30 127.0.0.1"
        else:
            cmd = "sleep 30"
        out, err, rc = await sandboxed_exec(cmd, tmp_path, cfg)
        assert "Timed out" in err
        assert rc == -1

    @pytest.mark.asyncio
    async def test_nonzero_exit(self, tmp_path: Path) -> None:
        if sys.platform == "win32":
            cmd = "cmd /c exit 42"
        else:
            cmd = "exit 42"
        out, err, rc = await sandboxed_exec(cmd, tmp_path)
        assert rc == 42

    @pytest.mark.asyncio
    async def test_output_truncation(self, tmp_path: Path) -> None:
        cfg = SandboxConfig(max_output=100)
        if sys.platform == "win32":
            cmd = 'python -c "print(\'x\' * 500)"'
        else:
            cmd = "python3 -c \"print('x' * 500)\""
        out, err, rc = await sandboxed_exec(cmd, tmp_path, cfg)
        assert len(out) <= 120  # 100 + truncation message
        assert "truncated" in out

    @pytest.mark.asyncio
    async def test_sandbox_disabled(self, tmp_path: Path) -> None:
        cfg = SandboxConfig(enabled=False)
        out, err, rc = await sandboxed_exec("echo works", tmp_path, cfg)
        assert "works" in out
        assert rc == 0

    @pytest.mark.asyncio
    async def test_env_sanitization_hides_secrets(self, tmp_path: Path) -> None:
        """When sandbox enabled, secret env vars should not leak."""
        os.environ["_NIBOT_TEST_SECRET"] = "leaked"
        try:
            cfg = SandboxConfig(enabled=True)
            if sys.platform == "win32":
                cmd = 'python -c "import os; print(os.environ.get(\'_NIBOT_TEST_SECRET\', \'clean\'))"'
            else:
                cmd = "python3 -c \"import os; print(os.environ.get('_NIBOT_TEST_SECRET', 'clean'))\""
            out, err, rc = await sandboxed_exec(cmd, tmp_path, cfg)
            assert "clean" in out
            assert "leaked" not in out
        finally:
            del os.environ["_NIBOT_TEST_SECRET"]


# ---- sandboxed_exec_py ----

class TestSandboxedExecPy:

    @pytest.mark.asyncio
    async def test_run_python_script(self, tmp_path: Path) -> None:
        script = tmp_path / "test.py"
        script.write_text("import json, sys\nprint(json.dumps({'ok': True}))", encoding="utf-8")
        out, err, rc = await sandboxed_exec_py(script, tmp_path)
        assert rc == 0
        assert '"ok": true' in out

    @pytest.mark.asyncio
    async def test_stdin_passed(self, tmp_path: Path) -> None:
        script = tmp_path / "echo.py"
        script.write_text(
            "import json, sys\ndata = json.loads(sys.stdin.read())\nprint(data['msg'])",
            encoding="utf-8",
        )
        out, err, rc = await sandboxed_exec_py(
            script, tmp_path, stdin_data=b'{"msg": "hello"}'
        )
        assert rc == 0
        assert "hello" in out

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self, tmp_path: Path) -> None:
        script = tmp_path / "slow.py"
        script.write_text("import time\ntime.sleep(30)", encoding="utf-8")
        cfg = SandboxConfig(timeout=1)
        out, err, rc = await sandboxed_exec_py(script, tmp_path, cfg)
        assert "Timed out" in err
        assert rc == -1


# ---- ExecTool with sandbox ----

class TestExecToolSandboxIntegration:

    @pytest.mark.asyncio
    async def test_exec_tool_with_sandbox(self, tmp_path: Path) -> None:
        from nibot.tools.exec_tool import ExecTool

        tool = ExecTool(tmp_path, timeout=10, sandbox_enabled=True)
        result = await tool.execute(command="echo sandbox_test")
        assert "sandbox_test" in result

    @pytest.mark.asyncio
    async def test_exec_tool_without_sandbox(self, tmp_path: Path) -> None:
        from nibot.tools.exec_tool import ExecTool

        tool = ExecTool(tmp_path, timeout=10, sandbox_enabled=False)
        result = await tool.execute(command="echo no_sandbox")
        assert "no_sandbox" in result

    @pytest.mark.asyncio
    async def test_dangerous_pattern_still_blocked(self, tmp_path: Path) -> None:
        from nibot.tools.exec_tool import ExecTool

        tool = ExecTool(tmp_path, timeout=10, sandbox_enabled=True)
        result = await tool.execute(command="rm -rf /")
        assert "Blocked" in result
