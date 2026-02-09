"""v1.2 Executable skills: SkillSpec.executable, SkillRunnerTool, creation with run.py."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from nibot.skills import SkillsLoader
from nibot.types import SkillSpec


# ---- Helpers ----

def _make_skill_dir(tmp_path: Path, name: str, body: str = "test body",
                    executable: bool = False, run_py_content: str = "") -> Path:
    """Create a skill directory with SKILL.md and optional run.py."""
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: test skill\nversion: 1\n---\n\n{body}",
        encoding="utf-8",
    )
    if executable or run_py_content:
        run_py = skill_dir / "run.py"
        content = run_py_content or textwrap.dedent("""\
            import json, sys
            args = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
            print(json.dumps({"echo": args, "status": "ok"}))
        """)
        run_py.write_text(content, encoding="utf-8")
    return skill_dir


# ---- SkillSpec.executable field ----

class TestSkillSpecExecutable:

    def test_default_not_executable(self) -> None:
        spec = SkillSpec(name="test", description="", body="", path="")
        assert spec.executable is False

    def test_explicit_executable(self) -> None:
        spec = SkillSpec(name="test", description="", body="", path="", executable=True)
        assert spec.executable is True


# ---- SkillsLoader detects run.py ----

class TestSkillsLoaderExecutableDetection:

    def test_skill_without_run_py_not_executable(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "plain_skill", executable=False)
        loader = SkillsLoader([tmp_path])
        loader.load_all()
        spec = loader.get("plain_skill")
        assert spec is not None
        assert spec.executable is False

    def test_skill_with_run_py_is_executable(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "exec_skill", executable=True)
        loader = SkillsLoader([tmp_path])
        loader.load_all()
        spec = loader.get("exec_skill")
        assert spec is not None
        assert spec.executable is True

    def test_reload_picks_up_new_run_py(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "evolving_skill", executable=False)
        loader = SkillsLoader([tmp_path])
        loader.load_all()
        assert loader.get("evolving_skill").executable is False

        # Add run.py after initial load
        (skill_dir / "run.py").write_text("print('hello')", encoding="utf-8")
        loader.reload()
        assert loader.get("evolving_skill").executable is True


# ---- SkillRunnerTool ----

class TestSkillRunnerTool:

    @pytest.mark.asyncio
    async def test_run_executable_skill(self, tmp_path: Path) -> None:
        from nibot.tools.skill_runner import SkillRunnerTool

        _make_skill_dir(tmp_path, "echo_skill", executable=True)
        loader = SkillsLoader([tmp_path])
        loader.load_all()

        tool = SkillRunnerTool(loader, tmp_path, timeout=10)
        result = await tool.execute(skill_name="echo_skill", args={"msg": "hi"})
        data = json.loads(result)
        assert data["status"] == "ok"
        assert data["echo"]["msg"] == "hi"

    @pytest.mark.asyncio
    async def test_run_nonexistent_skill(self, tmp_path: Path) -> None:
        from nibot.tools.skill_runner import SkillRunnerTool

        loader = SkillsLoader([tmp_path])
        loader.load_all()

        tool = SkillRunnerTool(loader, tmp_path, timeout=10)
        result = await tool.execute(skill_name="ghost")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_run_non_executable_skill(self, tmp_path: Path) -> None:
        from nibot.tools.skill_runner import SkillRunnerTool

        _make_skill_dir(tmp_path, "text_only", executable=False)
        loader = SkillsLoader([tmp_path])
        loader.load_all()

        tool = SkillRunnerTool(loader, tmp_path, timeout=10)
        result = await tool.execute(skill_name="text_only")
        assert "not executable" in result

    @pytest.mark.asyncio
    async def test_run_skill_timeout(self, tmp_path: Path) -> None:
        from nibot.tools.skill_runner import SkillRunnerTool

        _make_skill_dir(tmp_path, "slow_skill", run_py_content=textwrap.dedent("""\
            import time, sys, json
            args = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
            time.sleep(30)
            print("done")
        """))
        loader = SkillsLoader([tmp_path])
        loader.load_all()

        tool = SkillRunnerTool(loader, tmp_path, timeout=1)
        result = await tool.execute(skill_name="slow_skill")
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_run_skill_nonzero_exit(self, tmp_path: Path) -> None:
        from nibot.tools.skill_runner import SkillRunnerTool

        _make_skill_dir(tmp_path, "fail_skill", run_py_content=textwrap.dedent("""\
            import sys
            print("crash", file=sys.stderr)
            sys.exit(1)
        """))
        loader = SkillsLoader([tmp_path])
        loader.load_all()

        tool = SkillRunnerTool(loader, tmp_path, timeout=10)
        result = await tool.execute(skill_name="fail_skill")
        assert "failed" in result
        assert "exit=1" in result

    @pytest.mark.asyncio
    async def test_empty_skill_name(self, tmp_path: Path) -> None:
        from nibot.tools.skill_runner import SkillRunnerTool

        loader = SkillsLoader([tmp_path])
        tool = SkillRunnerTool(loader, tmp_path, timeout=10)
        result = await tool.execute(skill_name="")
        assert "required" in result


# ---- SkillTool._create_skill with executable flag ----

class TestSkillToolCreateExecutable:

    def test_create_executable_skill_generates_run_py(self, tmp_path: Path) -> None:
        from nibot.tools.admin_tools import SkillTool

        loader = SkillsLoader([tmp_path])
        tool = SkillTool(loader)
        result = tool._create_skill({
            "name": "my_exec_skill",
            "description": "A test executable skill",
            "body": "## Instructions\nDo something.",
            "executable": True,
        })
        assert "executable" in result
        assert (tmp_path / "my_exec_skill" / "run.py").exists()
        assert (tmp_path / "my_exec_skill" / "SKILL.md").exists()

        # Verify run.py is valid Python
        run_py = (tmp_path / "my_exec_skill" / "run.py").read_text()
        assert "def main" in run_py
        assert "json" in run_py

    def test_create_non_executable_skill_no_run_py(self, tmp_path: Path) -> None:
        from nibot.tools.admin_tools import SkillTool

        loader = SkillsLoader([tmp_path])
        tool = SkillTool(loader)
        tool._create_skill({
            "name": "plain_skill",
            "description": "A plain skill",
            "body": "Just instructions.",
        })
        assert not (tmp_path / "plain_skill" / "run.py").exists()
        assert (tmp_path / "plain_skill" / "SKILL.md").exists()

    def test_create_executable_does_not_overwrite_existing_run_py(self, tmp_path: Path) -> None:
        from nibot.tools.admin_tools import SkillTool

        # Pre-create skill with custom run.py
        skill_dir = tmp_path / "custom_skill"
        skill_dir.mkdir()
        (skill_dir / "run.py").write_text("# custom", encoding="utf-8")

        loader = SkillsLoader([tmp_path])
        tool = SkillTool(loader)
        tool._create_skill({
            "name": "custom_skill",
            "description": "Test",
            "body": "Test body",
            "executable": True,
        })
        # Should keep the original run.py
        assert (skill_dir / "run.py").read_text() == "# custom"
