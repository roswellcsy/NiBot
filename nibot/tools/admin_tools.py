"""Admin tools -- config, schedule, and skill management via conversation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from nibot.config import NiBotConfig, ScheduledJob
from nibot.registry import Tool
from nibot.scheduler import SchedulerManager
from nibot.skills import SkillsLoader

# Fields that ConfigTool is allowed to modify (security boundary).
_CONFIG_SAFE_FIELDS = {
    "agent.model", "agent.temperature", "agent.max_tokens",
    "agent.max_iterations", "agent.gateway_tools",
}


class ConfigTool(Tool):
    """Read or modify NiBot configuration through conversation."""

    def __init__(self, config: NiBotConfig, workspace: Path) -> None:
        self._config = config
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "config"

    @property
    def description(self) -> str:
        return "Read or modify NiBot configuration (model, temperature, gateway_tools, etc.)"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "set", "list"],
                           "description": "get: read a key, set: change a key, list: show all"},
                "key": {"type": "string", "description": "Dot-notation key (e.g. agent.model)"},
                "value": {"type": "string", "description": "New value (for set)"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        kwargs.pop("_tool_ctx", None)
        action = kwargs["action"]
        if action == "list":
            return self._list()
        key = kwargs.get("key", "")
        if not key:
            return "Error: 'key' is required for get/set."
        if action == "get":
            return self._get(key)
        if action == "set":
            value = kwargs.get("value", "")
            return self._set(key, value)
        return f"Unknown action: {action}"

    def _list(self) -> str:
        lines = [
            f"agent.model = {self._config.agent.model}",
            f"agent.temperature = {self._config.agent.temperature}",
            f"agent.max_tokens = {self._config.agent.max_tokens}",
            f"agent.max_iterations = {self._config.agent.max_iterations}",
            f"agent.gateway_tools = {self._config.agent.gateway_tools}",
            f"agents = {list(self._config.agents.keys())}",
        ]
        return "\n".join(lines)

    def _get(self, key: str) -> str:
        parts = key.split(".")
        obj: Any = self._config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return f"Unknown key: {key}"
        return f"{key} = {obj}"

    def _set(self, key: str, value: str) -> str:
        if key not in _CONFIG_SAFE_FIELDS:
            return f"Refused: '{key}' is not in safe-modify list. Allowed: {', '.join(sorted(_CONFIG_SAFE_FIELDS))}"
        parts = key.split(".")
        obj: Any = self._config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        field = parts[-1]
        current = getattr(obj, field)
        if isinstance(current, list):
            parsed = json.loads(value) if value.startswith("[") else [v.strip() for v in value.split(",")]
            setattr(obj, field, parsed)
        elif isinstance(current, int):
            setattr(obj, field, int(value))
        elif isinstance(current, float):
            setattr(obj, field, float(value))
        else:
            setattr(obj, field, value)
        return f"Set {key} = {getattr(obj, field)}"


class ScheduleTool(Tool):
    """Manage cron-scheduled tasks through conversation."""

    def __init__(self, scheduler: SchedulerManager, config: NiBotConfig, workspace: Path,
                 config_path: Path | None = None) -> None:
        self._scheduler = scheduler
        self._config = config
        self._workspace = workspace
        self._config_path = config_path or (workspace.parent / "config.json")

    @property
    def name(self) -> str:
        return "schedule"

    @property
    def description(self) -> str:
        return "Manage scheduled tasks. Add, remove, or list cron jobs."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "remove", "list"]},
                "id": {"type": "string", "description": "Job ID (for add/remove)"},
                "cron": {"type": "string", "description": "Cron expression, e.g. '0 9 * * *'"},
                "prompt": {"type": "string", "description": "Message to trigger when job fires"},
                "channel": {"type": "string", "description": "Target channel (e.g. telegram)"},
                "chat_id": {"type": "string", "description": "Target chat ID"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        kwargs.pop("_tool_ctx", None)
        action = kwargs["action"]
        if action == "list":
            jobs = self._scheduler.list_jobs()
            if not jobs:
                return "No scheduled jobs."
            lines = [f"- {j.id}: '{j.cron}' -> {j.prompt[:50]}... (channel={j.channel}, enabled={j.enabled})"
                     for j in jobs]
            return "\n".join(lines)
        if action == "add":
            job_id = kwargs.get("id", "")
            cron = kwargs.get("cron", "")
            prompt = kwargs.get("prompt", "")
            if not all([job_id, cron, prompt]):
                return "Error: 'id', 'cron', and 'prompt' are required for add."
            job = ScheduledJob(
                id=job_id, cron=cron, prompt=prompt,
                channel=kwargs.get("channel", "scheduler"),
                chat_id=kwargs.get("chat_id", ""),
            )
            self._scheduler.add(job)
            self._persist()
            return f"Added scheduled job '{job_id}': {cron}"
        if action == "remove":
            job_id = kwargs.get("id", "")
            if not job_id:
                return "Error: 'id' is required for remove."
            if self._scheduler.remove(job_id):
                self._persist()
                return f"Removed job '{job_id}'."
            return f"Job '{job_id}' not found."
        return f"Unknown action: {action}"

    def _persist(self) -> None:
        self._config.schedules = [
            ScheduledJob(**j.model_dump()) for j in self._scheduler.list_jobs()
        ]
        # Write to disk so changes survive restart
        config_file = self._config_path
        try:
            existing: dict = {}
            if config_file.exists():
                existing = json.loads(config_file.read_text(encoding="utf-8"))
            existing["schedules"] = [j.model_dump() for j in self._config.schedules]
            config_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass  # best-effort; runtime state is still authoritative


class SkillTool(Tool):
    """List, reload, or inspect skills."""

    def __init__(self, skills: SkillsLoader, marketplace: Any | None = None) -> None:
        self._skills = skills
        self._marketplace = marketplace

    @property
    def name(self) -> str:
        return "skill"

    @property
    def description(self) -> str:
        return "List, reload, inspect, or create skills."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "reload", "get", "create",
                                                       "disable", "enable", "delete", "update",
                                                       "search_marketplace", "install"]},
                "name": {"type": "string", "description": "Skill name (for get/create/disable/enable/delete/update)"},
                "description": {"type": "string", "description": "Skill description (for create/update)"},
                "body": {"type": "string", "description": "Skill body in markdown (for create/update)"},
                "url": {"type": "string", "description": "GitHub repo URL (for install)"},
                "query": {"type": "string", "description": "Search query (for search_marketplace)"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        kwargs.pop("_tool_ctx", None)
        action = kwargs["action"]
        if action == "list":
            all_skills = self._skills.get_all()
            if not all_skills:
                return "No skills loaded."
            lines = [f"- {s.name}: {s.description} (always={s.always})" for s in all_skills]
            return "\n".join(lines)
        if action == "reload":
            self._skills.reload()
            count = len(self._skills.get_all())
            return f"Skills reloaded. {count} skill(s) available."
        if action == "get":
            name = kwargs.get("name", "")
            if not name:
                return "Error: 'name' is required for get."
            spec = self._skills.get(name)
            if not spec:
                return f"Skill '{name}' not found."
            return f"# {spec.name}\n{spec.description}\n\nPath: {spec.path}\nAlways: {spec.always}\n\n{spec.body}"
        if action == "create":
            return self._create_skill(kwargs)
        if action == "disable":
            return self._disable_skill(kwargs.get("name", ""))
        if action == "enable":
            return self._enable_skill(kwargs.get("name", ""))
        if action == "delete":
            return self._delete_skill(kwargs.get("name", ""))
        if action == "update":
            return self._update_skill(kwargs)
        if action == "search_marketplace":
            return await self._search_marketplace(kwargs.get("query", ""))
        if action == "install":
            return await self._install_skill(kwargs.get("url", ""), kwargs.get("name", ""))
        return f"Unknown action: {action}"

    def _create_skill(self, kwargs: dict[str, Any]) -> str:
        from datetime import datetime as _dt

        name = kwargs.get("name", "")
        desc = kwargs.get("description", "")
        body = kwargs.get("body", "")
        if not name or not body:
            return "Error: 'name' and 'body' are required for create."
        if not self._skills.skills_dirs:
            return "Error: no skills directory configured."
        skill_dir = self._skills.skills_dirs[0] / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        created_at = _dt.now().isoformat()
        created_by = kwargs.get("created_by", "evolution")
        content = (
            f"---\nname: {name}\ndescription: {desc}\n"
            f"created_at: {created_at}\ncreated_by: {created_by}\nversion: 1\n"
            f"---\n\n{body}"
        )
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        self._skills.reload()
        return f"Skill '{name}' created and loaded. (created_at={created_at})"

    def _find_skill_dir(self, name: str) -> Path | None:
        for d in self._skills.skills_dirs:
            candidate = d / name
            if candidate.is_dir():
                return candidate
        return None

    def _disable_skill(self, name: str) -> str:
        if not name:
            return "Error: 'name' is required for disable."
        skill_dir = self._find_skill_dir(name)
        if not skill_dir:
            return f"Skill directory '{name}' not found."
        md = skill_dir / "SKILL.md"
        if not md.exists():
            return f"Skill '{name}' is already disabled or missing SKILL.md."
        md.rename(skill_dir / "SKILL.md.disabled")
        self._skills.reload()
        return f"Skill '{name}' disabled."

    def _enable_skill(self, name: str) -> str:
        if not name:
            return "Error: 'name' is required for enable."
        skill_dir = self._find_skill_dir(name)
        if not skill_dir:
            return f"Skill directory '{name}' not found."
        disabled = skill_dir / "SKILL.md.disabled"
        if not disabled.exists():
            return f"Skill '{name}' is not disabled (no SKILL.md.disabled found)."
        disabled.rename(skill_dir / "SKILL.md")
        self._skills.reload()
        return f"Skill '{name}' enabled."

    def _delete_skill(self, name: str) -> str:
        if not name:
            return "Error: 'name' is required for delete."
        skill_dir = self._find_skill_dir(name)
        if not skill_dir:
            return f"Skill directory '{name}' not found."
        shutil.rmtree(skill_dir)
        self._skills.reload()
        return f"Skill '{name}' permanently deleted."

    def _update_skill(self, kwargs: dict[str, Any]) -> str:
        from datetime import datetime as _dt

        name = kwargs.get("name", "")
        if not name:
            return "Error: 'name' is required for update."
        spec = self._skills.get(name)
        if not spec:
            return f"Skill '{name}' not found. Use create instead."
        new_desc = kwargs.get("description", "") or spec.description
        new_body = kwargs.get("body", "") or spec.body
        new_version = spec.version + 1
        content = (
            f"---\nname: {name}\ndescription: {new_desc}\n"
            f"created_at: {spec.created_at}\ncreated_by: {spec.created_by}\n"
            f"version: {new_version}\n---\n\n{new_body}"
        )
        skill_dir = self._find_skill_dir(name)
        if not skill_dir:
            return f"Skill directory '{name}' not found."
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        self._skills.reload()
        return f"Skill '{name}' updated to v{new_version}."

    async def _search_marketplace(self, query: str) -> str:
        if not self._marketplace:
            return "Error: marketplace not configured."
        results = await self._marketplace.search(query)
        if not results:
            return "No skills found."
        lines = [f"  {s.name} ({s.author}) - {s.description[:80]} [{s.stars} stars]" for s in results]
        return "Marketplace skills:\n" + "\n".join(lines)

    async def _install_skill(self, url: str, name: str) -> str:
        if not self._marketplace:
            return "Error: marketplace not configured."
        if not url:
            return "Error: 'url' is required for install."
        result = await self._marketplace.install(url, name)
        if not result.startswith("Error"):
            self._skills.reload()
        return result
