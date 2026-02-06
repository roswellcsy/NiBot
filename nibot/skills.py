"""Progressive skill loading -- always-skills inline, others as XML summary."""

from __future__ import annotations

import os
import shutil
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
                    self._specs[spec.name] = spec
                    seen.add(skill_dir.name)
                    logger.debug(f"Loaded skill: {spec.name} (always={spec.always})")

    def get_always_skills(self) -> list[SkillSpec]:
        return [s for s in self._specs.values() if s.always]

    def get_all(self) -> list[SkillSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> SkillSpec | None:
        return self._specs.get(name)

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
        return SkillSpec(
            name=meta.get("name", path.parent.name),
            description=meta.get("description", ""),
            body=post.content,
            path=str(path),
            always=nanobot_meta.get("always", False),
            requires_bins=requires.get("bins", []),
            requires_env=requires.get("env", []),
        )

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
