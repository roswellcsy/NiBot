"""Execute skill run.py scripts in a sandboxed subprocess."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nibot.registry import Tool
from nibot.sandbox import SandboxConfig, sandboxed_exec_py
from nibot.skills import SkillsLoader


class SkillRunnerTool(Tool):
    """Run an executable skill's run.py via subprocess with JSON stdin/stdout protocol."""

    def __init__(self, skills: SkillsLoader, workspace: Path, timeout: int = 60,
                 sandbox_enabled: bool = True) -> None:
        self._skills = skills
        self._workspace = workspace
        self._timeout = timeout
        self._sandbox_enabled = sandbox_enabled

    @property
    def name(self) -> str:
        return "run_skill"

    @property
    def description(self) -> str:
        return (
            "Execute a registered executable skill by name. "
            "Pass arguments as JSON; receives stdout as result."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the executable skill to run",
                },
                "args": {
                    "type": "object",
                    "description": "Arguments to pass to the skill (JSON object)",
                },
            },
            "required": ["skill_name"],
        }

    async def execute(self, **kwargs: Any) -> str:
        skill_name = kwargs.get("skill_name", "")
        args = kwargs.get("args", {})

        if not skill_name:
            return "Error: 'skill_name' is required."

        spec = self._skills.get(skill_name)
        if not spec:
            return f"Error: skill '{skill_name}' not found."
        if not spec.executable:
            return f"Error: skill '{skill_name}' is not executable (no run.py)."

        run_py = Path(spec.path).parent / "run.py"
        if not run_py.exists():
            return f"Error: run.py not found at {run_py}"

        input_data = json.dumps(args, ensure_ascii=False).encode("utf-8")

        cfg = SandboxConfig(
            timeout=self._timeout,
            enabled=self._sandbox_enabled,
        )
        out, err, rc = await sandboxed_exec_py(run_py, self._workspace, cfg, input_data)

        if rc == -1 and ("Timed out" in err or "Exec error" in err):
            self._skills.record_usage(skill_name, success=False)
            return f"Skill '{skill_name}' {err.lower()}"

        if rc != 0:
            self._skills.record_usage(skill_name, success=False)
            msg = f"Skill '{skill_name}' failed (exit={rc})"
            if err:
                msg += f"\n[stderr]\n{err}"
            if out:
                msg += f"\n[stdout]\n{out}"
            return msg

        self._skills.record_usage(skill_name, success=True)
        return out if out else "(no output)"
