"""Tests for v0.8.0b features: evolution log, skill management, evolution trigger, context injection."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nibot.evolution_log import EvolutionDecision, EvolutionLog
from nibot.metrics import SessionMetrics, should_trigger_evolution
from nibot.session import Session, SessionManager
from nibot.skills import SkillsLoader
from nibot.tools.admin_tools import SkillTool
from nibot.tools.analyze_tool import AnalyzeTool
from nibot.types import SkillSpec


# ---- helpers ----

def _msg(role: str, content: str = "", ts: str = "", **extra: Any) -> dict[str, Any]:
    m: dict[str, Any] = {"role": role, "content": content}
    if ts:
        m["timestamp"] = ts
    m.update(extra)
    return m


def _make_skill_dir(tmp_path: Path, name: str, body: str = "Test body",
                    desc: str = "A test skill", version: int = 1) -> Path:
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True)
    content = (
        f"---\nname: {name}\ndescription: {desc}\n"
        f"created_at: 2025-01-15T10:00:00\ncreated_by: evolution\nversion: {version}\n"
        f"---\n\n{body}"
    )
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


# ---- B4: EvolutionLog ----

class TestEvolutionLog:
    def test_append_and_read(self, tmp_path: Path) -> None:
        log = EvolutionLog(tmp_path)
        d = EvolutionDecision(
            trigger="cron", action="create_skill", skill_name="test",
            reasoning="High error rate", outcome="success",
        )
        log.append(d)

        recent = log.read_recent(10)
        assert len(recent) == 1
        assert recent[0].trigger == "cron"
        assert recent[0].action == "create_skill"
        assert recent[0].skill_name == "test"
        assert recent[0].timestamp  # auto-filled

    def test_read_recent_order(self, tmp_path: Path) -> None:
        log = EvolutionLog(tmp_path)
        for i in range(5):
            log.append(EvolutionDecision(
                trigger="cron", action="skip", skill_name=f"s{i}",
                reasoning=f"reason {i}", outcome="skipped",
            ))

        recent = log.read_recent(3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].skill_name == "s4"
        assert recent[2].skill_name == "s2"

    def test_read_empty(self, tmp_path: Path) -> None:
        log = EvolutionLog(tmp_path)
        assert log.read_recent() == []

    def test_summary(self, tmp_path: Path) -> None:
        log = EvolutionLog(tmp_path)
        log.append(EvolutionDecision(
            trigger="error_rate", action="create_skill", skill_name="fix_errors",
            reasoning="error_rate=0.45", outcome="success",
        ))
        s = log.summary(5)
        assert "error_rate" in s
        assert "fix_errors" in s
        assert "success" in s

    def test_summary_empty(self, tmp_path: Path) -> None:
        log = EvolutionLog(tmp_path)
        s = log.summary()
        assert "no evolution decisions" in s


# ---- B3-part1: should_trigger_evolution ----

class TestShouldTriggerEvolution:
    def test_insufficient_data(self) -> None:
        sessions = [SessionMetrics(tool_calls=1, tool_errors=1) for _ in range(3)]
        triggered, reason = should_trigger_evolution(sessions, min_sessions=5)
        assert not triggered
        assert "insufficient" in reason

    def test_high_error_rate_triggers(self) -> None:
        sessions = [SessionMetrics(tool_calls=10, tool_errors=5) for _ in range(5)]
        triggered, reason = should_trigger_evolution(sessions, error_rate_threshold=0.3)
        assert triggered
        assert "error_rate=" in reason

    def test_normal_metrics_no_trigger(self) -> None:
        sessions = [SessionMetrics(tool_calls=10, tool_errors=0) for _ in range(5)]
        triggered, reason = should_trigger_evolution(sessions, error_rate_threshold=0.3)
        assert not triggered
        assert "normal range" in reason

    def test_exact_threshold(self) -> None:
        sessions = [SessionMetrics(tool_calls=10, tool_errors=3) for _ in range(5)]
        triggered, _ = should_trigger_evolution(sessions, error_rate_threshold=0.3)
        assert triggered  # 0.3 >= 0.3


# ---- B2: SkillTool disable/enable/delete/update ----

class TestSkillToolManagement:
    @pytest.mark.asyncio
    async def test_disable_skill(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "test_skill")
        loader = SkillsLoader([tmp_path / "skills"])
        loader.load_all()
        tool = SkillTool(loader)

        result = await tool.execute(action="disable", name="test_skill")
        assert "disabled" in result
        assert (tmp_path / "skills" / "test_skill" / "SKILL.md.disabled").exists()
        assert not (tmp_path / "skills" / "test_skill" / "SKILL.md").exists()
        # Should not be visible after reload
        assert loader.get("test_skill") is None

    @pytest.mark.asyncio
    async def test_enable_skill(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "test_skill")
        # Manually disable it
        (skill_dir / "SKILL.md").rename(skill_dir / "SKILL.md.disabled")
        loader = SkillsLoader([tmp_path / "skills"])
        loader.load_all()
        assert loader.get("test_skill") is None

        tool = SkillTool(loader)
        result = await tool.execute(action="enable", name="test_skill")
        assert "enabled" in result
        assert (skill_dir / "SKILL.md").exists()
        assert loader.get("test_skill") is not None

    @pytest.mark.asyncio
    async def test_delete_skill(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "test_skill")
        loader = SkillsLoader([tmp_path / "skills"])
        loader.load_all()
        tool = SkillTool(loader)

        result = await tool.execute(action="delete", name="test_skill")
        assert "deleted" in result
        assert not (tmp_path / "skills" / "test_skill").exists()
        assert loader.get("test_skill") is None

    @pytest.mark.asyncio
    async def test_update_skill(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "test_skill", version=1, body="Old body")
        loader = SkillsLoader([tmp_path / "skills"])
        loader.load_all()
        tool = SkillTool(loader)

        result = await tool.execute(action="update", name="test_skill", body="New body")
        assert "v2" in result
        spec = loader.get("test_skill")
        assert spec is not None
        assert spec.version == 2
        assert "New body" in spec.body

    @pytest.mark.asyncio
    async def test_disable_missing_skill(self, tmp_path: Path) -> None:
        loader = SkillsLoader([tmp_path / "skills"])
        tool = SkillTool(loader)
        result = await tool.execute(action="disable", name="nonexistent")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_update_preserves_metadata(self, tmp_path: Path) -> None:
        _make_skill_dir(tmp_path, "test_skill", version=3, desc="Original desc")
        loader = SkillsLoader([tmp_path / "skills"])
        loader.load_all()
        tool = SkillTool(loader)

        result = await tool.execute(action="update", name="test_skill", body="Updated body")
        assert "v4" in result
        spec = loader.get("test_skill")
        assert spec is not None
        assert spec.version == 4
        assert spec.created_at == "2025-01-15T10:00:00"
        assert spec.created_by == "evolution"

    @pytest.mark.asyncio
    async def test_disable_already_disabled(self, tmp_path: Path) -> None:
        skill_dir = _make_skill_dir(tmp_path, "test_skill")
        (skill_dir / "SKILL.md").rename(skill_dir / "SKILL.md.disabled")
        loader = SkillsLoader([tmp_path / "skills"])
        loader.load_all()
        tool = SkillTool(loader)
        result = await tool.execute(action="disable", name="test_skill")
        assert "already disabled" in result


# ---- B4-part2: AnalyzeTool log_decision + decision_history ----

class TestAnalyzeDecisionLog:
    @pytest.mark.asyncio
    async def test_log_decision(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        log = EvolutionLog(tmp_path / "workspace")
        tool = AnalyzeTool(sm, evolution_log=log)

        result = await tool.execute(
            action="log_decision",
            trigger="cron",
            decision_action="create_skill",
            skill_name="new_skill",
            reasoning="High error rate in exec calls",
            outcome="success",
        )
        assert "logged" in result.lower()

        decisions = log.read_recent(5)
        assert len(decisions) == 1
        assert decisions[0].action == "create_skill"

    @pytest.mark.asyncio
    async def test_decision_history(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        log = EvolutionLog(tmp_path / "workspace")
        log.append(EvolutionDecision(
            trigger="manual", action="skip", reasoning="metrics healthy", outcome="skipped",
        ))
        tool = AnalyzeTool(sm, evolution_log=log)

        result = await tool.execute(action="decision_history", limit=10)
        assert "skip" in result
        assert "healthy" in result

    @pytest.mark.asyncio
    async def test_log_decision_no_log(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="log_decision")
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_decision_history_no_log(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="decision_history")
        assert "not available" in result


# ---- B3-part2: EvolutionTrigger ----

class TestEvolutionTrigger:
    @pytest.mark.asyncio
    async def test_disabled_returns_false(self, tmp_path: Path) -> None:
        from nibot.evolution_trigger import EvolutionTrigger

        bus = MagicMock()
        sm = SessionManager(tmp_path / "sessions")
        trigger = EvolutionTrigger(bus, sm, enabled=False)
        assert await trigger.check() is False

    @pytest.mark.asyncio
    async def test_triggers_on_high_error_rate(self, tmp_path: Path) -> None:
        from nibot.evolution_trigger import EvolutionTrigger

        bus = MagicMock()
        bus.publish_inbound = AsyncMock()
        sm = SessionManager(tmp_path / "sessions")
        now = datetime.now()

        # Create sessions with high error rate (save to disk for iter_recent_from_disk)
        for i in range(6):
            s = Session(
                key=f"err:{i}",
                messages=[
                    _msg("tool", "Error: something broke", name="exec"),
                    _msg("tool", "ok", name="exec"),
                ],
                updated_at=now - timedelta(minutes=i),
            )
            sm.save(s)

        trigger = EvolutionTrigger(
            bus, sm, enabled=True, error_rate_threshold=0.3, min_sessions=5,
        )
        result = await trigger.check()
        assert result is True
        bus.publish_inbound.assert_called_once()

    @pytest.mark.asyncio
    async def test_cooldown_prevents_repeat(self, tmp_path: Path) -> None:
        from nibot.evolution_trigger import EvolutionTrigger

        bus = MagicMock()
        bus.publish_inbound = AsyncMock()
        sm = SessionManager(tmp_path / "sessions")
        now = datetime.now()

        for i in range(6):
            s = Session(
                key=f"err:{i}",
                messages=[
                    _msg("tool", "Error: broke", name="exec"),
                    _msg("tool", "ok", name="exec"),
                ],
                updated_at=now - timedelta(minutes=i),
            )
            sm.save(s)

        trigger = EvolutionTrigger(
            bus, sm, enabled=True, cooldown_seconds=3600, min_sessions=5,
        )
        # First check should trigger
        assert await trigger.check() is True
        # Second check should be blocked by cooldown
        assert await trigger.check() is False


# ---- B1: build_evolution_context ----

class TestBuildEvolutionContext:
    def test_basic_context(self, tmp_path: Path) -> None:
        from nibot.subagent import build_evolution_context

        sm = SessionManager(tmp_path / "sessions")
        now = datetime.now()
        s = Session(
            key="test:1",
            messages=[
                _msg("user", "hello"),
                _msg("tool", "ok", name="exec"),
                _msg("assistant", "done"),
            ],
            updated_at=now,
        )
        sm._cache["test:1"] = s

        skills = SkillsLoader([tmp_path / "skills"])
        result = build_evolution_context(sm, skills)
        assert "Aggregate Metrics" in result
        assert "Skill Inventory" in result
        assert "(no skills)" in result

    def test_context_with_skills(self, tmp_path: Path) -> None:
        from nibot.subagent import build_evolution_context

        sm = SessionManager(tmp_path / "sessions")
        _make_skill_dir(tmp_path, "my_skill", desc="A useful skill")
        skills = SkillsLoader([tmp_path / "skills"])
        skills.load_all()

        result = build_evolution_context(sm, skills)
        assert "my_skill" in result
        assert "A useful skill" in result

    def test_context_with_evolution_log(self, tmp_path: Path) -> None:
        from nibot.subagent import build_evolution_context

        sm = SessionManager(tmp_path / "sessions")
        skills = SkillsLoader([tmp_path / "skills"])
        log = EvolutionLog(tmp_path / "evo")
        log.append(EvolutionDecision(
            trigger="cron", action="skip", reasoning="all good", outcome="skipped",
        ))

        result = build_evolution_context(sm, skills, evolution_log=log)
        assert "Evolution Decisions" in result
        assert "skip" in result
