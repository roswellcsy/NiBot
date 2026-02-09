"""Progressive skill loading -- always-skills inline, others as XML summary."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from nibot.log import logger
from nibot.types import SkillSpec


class SkillsLoader:
    """Load SKILL.md files from multiple directories. Workspace skills take priority."""

    def __init__(self, skills_dirs: list[Path]) -> None:
        self.skills_dirs = skills_dirs
        self._specs: dict[str, SkillSpec] = {}

    def load_all(self) -> None:
        seen: set[str] = set()
        for d in self.skills_dirs:
            if not d.exists():
                continue
            for skill_dir in sorted(d.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_file = skill_dir / "SKILL.md"
                if not skill_file.exists():
                    continue
                if skill_dir.name in seen:
                    continue
                spec = self._parse(skill_file)
                if spec and self._check_requirements(spec):
                    self._load_stats(spec)
                    self._specs[spec.name] = spec
                    seen.add(skill_dir.name)
                    logger.debug(f"Loaded skill: {spec.name} (always={spec.always})")

    def get_always_skills(self) -> list[SkillSpec]:
        return [s for s in self._specs.values() if s.always]

    def get_all(self) -> list[SkillSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> SkillSpec | None:
        return self._specs.get(name)

    def reload(self) -> None:
        """Clear and re-load all skills. Safe: no await between clear/load (atomic in asyncio)."""
        self._specs.clear()
        self.load_all()

    def build_summary(self) -> str:
        """XML summary of non-always skills for system prompt."""
        non_always = [s for s in self._specs.values() if not s.always]
        if not non_always:
            return ""
        lines = ["<skills>"]
        for s in non_always:
            lines.append(f'  <skill name="{s.name}">')
            lines.append(f"    <description>{s.description}</description>")
            lines.append(f"    <location>{s.path}</location>")
            lines.append("  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    def _parse(self, path: Path) -> SkillSpec | None:
        try:
            import frontmatter
        except ImportError:
            logger.warning("python-frontmatter not installed, skipping skills")
            return None
        try:
            post = frontmatter.load(str(path))
        except Exception as e:
            logger.warning(f"Failed to parse {path}: {e}")
            return None
        meta = post.metadata or {}
        nanobot_meta: dict = {}
        meta_str = meta.get("metadata", "")
        if meta_str and isinstance(meta_str, str):
            import json

            try:
                nanobot_meta = json.loads(meta_str).get("nanobot", {})
            except (ValueError, AttributeError):
                pass
        requires = nanobot_meta.get("requires", {})
        has_run_py = (path.parent / "run.py").exists()
        return SkillSpec(
            name=meta.get("name", path.parent.name),
            description=meta.get("description", ""),
            body=post.content,
            path=str(path),
            always=nanobot_meta.get("always", False),
            requires_bins=requires.get("bins", []),
            requires_env=requires.get("env", []),
            created_at=self._to_str(meta.get("created_at", "")),
            created_by=str(meta.get("created_by", "")),
            version=int(meta.get("version", 1)),
            executable=has_run_py,
        )

    @staticmethod
    def _to_str(val: object) -> str:
        if isinstance(val, str):
            return val
        if hasattr(val, "isoformat"):
            return val.isoformat()  # type: ignore[union-attr]
        return str(val) if val else ""

    def record_usage(self, name: str, success: bool) -> None:
        """Record a skill execution. Persists stats to disk."""
        spec = self._specs.get(name)
        if not spec:
            return
        spec.usage_count += 1
        if success:
            spec.success_count += 1
        spec.last_used = datetime.now().isoformat()
        self._save_stats(spec)

    def _load_stats(self, spec: SkillSpec) -> None:
        """Load usage stats from stats.json sidecar file."""
        stats_path = Path(spec.path).parent / "stats.json"
        if not stats_path.exists():
            return
        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
            spec.usage_count = data.get("usage_count", 0)
            spec.success_count = data.get("success_count", 0)
            spec.last_used = data.get("last_used", "")
        except (json.JSONDecodeError, OSError):
            pass

    def _save_stats(self, spec: SkillSpec) -> None:
        """Persist usage stats to stats.json next to SKILL.md."""
        stats_path = Path(spec.path).parent / "stats.json"
        try:
            stats_path.write_text(json.dumps({
                "usage_count": spec.usage_count,
                "success_count": spec.success_count,
                "last_used": spec.last_used,
            }, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Failed to save skill stats for {spec.name}: {e}")

    def _check_requirements(self, spec: SkillSpec) -> bool:
        for b in spec.requires_bins:
            if not shutil.which(b):
                logger.debug(f"Skill {spec.name}: missing binary '{b}'")
                return False
        for env in spec.requires_env:
            if not os.environ.get(env):
                logger.debug(f"Skill {spec.name}: missing env var '{env}'")
                return False
        return True
