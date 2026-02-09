"""Tests for skill usage tracking (evolution feedback loop)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nibot.skills import SkillsLoader
from nibot.types import SkillSpec


def _create_skill(skills_dir: Path, name: str) -> None:
    """Create a minimal skill directory with SKILL.md."""
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\nBody of {name}",
        encoding="utf-8",
    )


class TestSkillUsageTracking:
    def test_record_usage_increments_counts(self, tmp_path):
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "greet")

        loader = SkillsLoader([skills_dir])
        loader.load_all()

        loader.record_usage("greet", success=True)
        loader.record_usage("greet", success=True)
        loader.record_usage("greet", success=False)

        spec = loader.get("greet")
        assert spec is not None
        assert spec.usage_count == 3
        assert spec.success_count == 2
        assert spec.last_used != ""

    def test_record_usage_unknown_skill_is_noop(self, tmp_path):
        loader = SkillsLoader([tmp_path])
        loader.load_all()
        loader.record_usage("nonexistent", success=True)  # no error

    def test_stats_persist_to_disk(self, tmp_path):
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "persist_test")

        loader = SkillsLoader([skills_dir])
        loader.load_all()

        loader.record_usage("persist_test", success=True)
        loader.record_usage("persist_test", success=False)

        stats_path = skills_dir / "persist_test" / "stats.json"
        assert stats_path.exists()

        data = json.loads(stats_path.read_text(encoding="utf-8"))
        assert data["usage_count"] == 2
        assert data["success_count"] == 1
        assert data["last_used"] != ""

    def test_stats_loaded_on_reload(self, tmp_path):
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "reload_test")

        loader = SkillsLoader([skills_dir])
        loader.load_all()

        loader.record_usage("reload_test", success=True)
        loader.record_usage("reload_test", success=True)
        loader.record_usage("reload_test", success=True)

        # Reload from disk
        loader.reload()

        spec = loader.get("reload_test")
        assert spec is not None
        assert spec.usage_count == 3
        assert spec.success_count == 3

    def test_success_rate_calculation(self, tmp_path):
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "rate_test")

        loader = SkillsLoader([skills_dir])
        loader.load_all()

        for _ in range(7):
            loader.record_usage("rate_test", success=True)
        for _ in range(3):
            loader.record_usage("rate_test", success=False)

        spec = loader.get("rate_test")
        assert spec is not None
        assert spec.usage_count == 10
        assert spec.success_count == 7
        rate = spec.success_count / spec.usage_count
        assert abs(rate - 0.7) < 0.001
