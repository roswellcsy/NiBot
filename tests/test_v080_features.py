"""Tests for v0.8.0a features: metrics computation, evolution tracking, analyze enhancements."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nibot.metrics import (
    AggregateMetrics,
    SessionMetrics,
    aggregate_metrics,
    compute_session_metrics,
)
from nibot.session import Session, SessionManager
from nibot.skills import SkillsLoader
from nibot.tools.analyze_tool import AnalyzeTool
from nibot.types import SkillSpec


# ---- helpers ----

def _msg(role: str, content: str = "", ts: str = "", **extra: Any) -> dict[str, Any]:
    m: dict[str, Any] = {"role": role, "content": content}
    if ts:
        m["timestamp"] = ts
    m.update(extra)
    return m


# ---- A1: compute_session_metrics ----

class TestComputeSessionMetrics:
    def test_empty_messages(self) -> None:
        m = compute_session_metrics([])
        assert m.message_count == 0
        assert m.tool_calls == 0
        assert m.error_rate == 0.0

    def test_basic_conversation(self) -> None:
        msgs = [
            _msg("user", "Hello"),
            _msg("assistant", "Hi there!"),
        ]
        m = compute_session_metrics(msgs)
        assert m.message_count == 2
        assert m.user_messages == 1
        assert m.assistant_messages == 1
        assert m.conversation_turns == 1
        assert m.tool_calls == 0

    def test_tool_calls_and_errors(self) -> None:
        msgs = [
            _msg("user", "Do something"),
            _msg("tool", "result ok", name="file_read"),
            _msg("tool", "Error: file not found", name="write_file"),
            _msg("tool", "another result", name="file_read"),
            _msg("assistant", "Done"),
        ]
        m = compute_session_metrics(msgs)
        assert m.tool_calls == 3
        assert m.tool_errors == 1
        assert abs(m.error_rate - 1 / 3) < 0.01
        assert len(m.error_messages) == 1
        assert "file not found" in m.error_messages[0]

    def test_unique_tools(self) -> None:
        msgs = [
            _msg("tool", "ok", name="exec"),
            _msg("tool", "ok", name="file_read"),
            _msg("tool", "ok", name="exec"),
        ]
        m = compute_session_metrics(msgs)
        assert m.unique_tools == ["exec", "file_read"]
        assert m.tool_diversity == 2

    def test_timestamps_and_duration(self) -> None:
        t0 = "2025-01-15T10:00:00"
        t1 = "2025-01-15T10:05:00"
        msgs = [
            _msg("user", "start", ts=t0),
            _msg("assistant", "done", ts=t1),
        ]
        m = compute_session_metrics(msgs)
        assert m.first_message_at == t0
        assert m.last_message_at == t1
        assert m.duration_seconds == 300.0

    def test_avg_response_length(self) -> None:
        msgs = [
            _msg("assistant", "short"),       # 5 chars
            _msg("assistant", "a bit longer"),  # 12 chars
        ]
        m = compute_session_metrics(msgs)
        assert m.avg_response_length == (5 + 12) / 2

    def test_no_timestamps(self) -> None:
        msgs = [_msg("user", "hi"), _msg("assistant", "hey")]
        m = compute_session_metrics(msgs)
        assert m.duration_seconds == 0.0
        assert m.first_message_at == ""


# ---- aggregate_metrics ----

class TestAggregateMetrics:
    def test_empty_list(self) -> None:
        agg = aggregate_metrics([])
        assert agg.session_count == 0
        assert agg.total_messages == 0

    def test_multiple_sessions(self) -> None:
        s1 = SessionMetrics(
            message_count=10, tool_calls=5, tool_errors=1,
            unique_tools=["exec", "file_read"],
            conversation_turns=3, duration_seconds=60.0,
        )
        s2 = SessionMetrics(
            message_count=8, tool_calls=3, tool_errors=0,
            unique_tools=["exec"],
            conversation_turns=2, duration_seconds=30.0,
        )
        agg = aggregate_metrics([s1, s2])
        assert agg.session_count == 2
        assert agg.total_messages == 18
        assert agg.total_tool_calls == 8
        assert agg.total_tool_errors == 1
        assert abs(agg.overall_error_rate - 1 / 8) < 0.01
        assert abs(agg.avg_turns_per_session - 2.5) < 0.01
        assert abs(agg.avg_duration_seconds - 45.0) < 0.01

    def test_tool_usage_ranking(self) -> None:
        s1 = SessionMetrics(unique_tools=["exec", "file_read"])
        s2 = SessionMetrics(unique_tools=["exec"])
        s3 = SessionMetrics(unique_tools=["exec", "write_file"])
        agg = aggregate_metrics([s1, s2, s3])
        keys = list(agg.tool_usage.keys())
        # After to_dict() sorts by frequency
        d = agg.to_dict()
        tool_keys = list(d["tool_usage"].keys())
        assert tool_keys[0] == "exec"  # appears in 3 sessions

    def test_top_errors_dedup(self) -> None:
        s1 = SessionMetrics(error_messages=["Error: not found blah blah"])
        s2 = SessionMetrics(error_messages=["Error: not found blah blah"])  # same prefix
        s3 = SessionMetrics(error_messages=["Error: permission denied xyz"])
        agg = aggregate_metrics([s1, s2, s3])
        assert len(agg.top_errors) == 2  # deduplicated by 80-char prefix


# ---- A2: SkillSpec tracking ----

class TestSkillSpecTracking:
    def test_default_values(self) -> None:
        spec = SkillSpec(name="test", description="", body="", path="")
        assert spec.created_at == ""
        assert spec.created_by == ""
        assert spec.version == 1

    def test_parse_skill_with_tracking(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "test_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test_skill\ndescription: A test\n"
            "created_at: 2025-01-15T10:00:00\ncreated_by: evolution\nversion: 2\n"
            "---\n\nBody content here.",
            encoding="utf-8",
        )
        loader = SkillsLoader([tmp_path / "skills"])
        loader.load_all()
        spec = loader.get("test_skill")
        assert spec is not None
        assert spec.created_at == "2025-01-15T10:00:00"
        assert spec.created_by == "evolution"
        assert spec.version == 2

    def test_parse_skill_without_tracking(self, tmp_path: Path) -> None:
        """Old SKILL.md without tracking fields should use defaults."""
        skill_dir = tmp_path / "skills" / "old_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: old_skill\ndescription: Legacy\n---\n\nOld body.",
            encoding="utf-8",
        )
        loader = SkillsLoader([tmp_path / "skills"])
        loader.load_all()
        spec = loader.get("old_skill")
        assert spec is not None
        assert spec.created_at == ""
        assert spec.created_by == ""
        assert spec.version == 1


# ---- A4: AnalyzeTool enhancements ----

def _make_session(key: str, messages: list[dict], updated_at: datetime) -> Session:
    s = Session(key=key, messages=messages, updated_at=updated_at)
    return s


class TestAnalyzeMetrics:
    @pytest.mark.asyncio
    async def test_metrics_action(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        now = datetime.now()
        s = Session(key="test:1", messages=[
            _msg("user", "hello", ts=now.isoformat()),
            _msg("tool", "ok", name="exec", timestamp=now.isoformat()),
            _msg("assistant", "done", ts=(now + timedelta(seconds=30)).isoformat()),
        ], updated_at=now)
        sm.save(s)

        tool = AnalyzeTool(sm)
        result = await tool.execute(action="metrics", limit=10)
        data = json.loads(result)
        assert data["session_count"] == 1
        assert data["total_messages"] == 3
        assert data["total_tool_calls"] == 1

    @pytest.mark.asyncio
    async def test_metrics_empty_sessions(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="metrics", limit=10)
        data = json.loads(result)
        assert data["session_count"] == 0
        assert data["total_messages"] == 0

    @pytest.mark.asyncio
    async def test_skill_impact_before_after(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        skill_created = datetime(2025, 1, 15, 12, 0, 0)

        # Before sessions
        s_before = Session(
            key="before:1",
            messages=[
                _msg("tool", "Error: fail", name="exec"),
                _msg("tool", "ok", name="exec"),
            ],
            updated_at=skill_created - timedelta(hours=1),
        )
        # After sessions
        s_after = Session(
            key="after:1",
            messages=[
                _msg("tool", "ok", name="exec"),
                _msg("tool", "ok", name="exec"),
            ],
            updated_at=skill_created + timedelta(hours=1),
        )
        sm.save(s_before)
        sm.save(s_after)

        # Mock skills loader
        mock_skills = MagicMock()
        mock_skills.get.return_value = SkillSpec(
            name="error_fix", description="", body="", path="",
            created_at=skill_created.isoformat(),
        )

        tool = AnalyzeTool(sm, skills=mock_skills)
        result = await tool.execute(action="skill_impact", skill_name="error_fix", limit=20)
        data = json.loads(result)
        assert data["skill"] == "error_fix"
        assert data["before"]["sessions"] == 1
        assert data["before"]["error_rate"] == 0.5
        assert data["after"]["sessions"] == 1
        assert data["after"]["error_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_skill_impact_missing_skill(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        mock_skills = MagicMock()
        mock_skills.get.return_value = None

        tool = AnalyzeTool(sm, skills=mock_skills)
        result = await tool.execute(action="skill_impact", skill_name="nonexistent")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_skill_impact_no_created_at(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        mock_skills = MagicMock()
        mock_skills.get.return_value = SkillSpec(
            name="old", description="", body="", path="",
        )

        tool = AnalyzeTool(sm, skills=mock_skills)
        result = await tool.execute(action="skill_impact", skill_name="old")
        assert "no created_at" in result


# ---- A5: SkillTool create tracking ----

class TestSkillCreateTracking:
    @pytest.mark.asyncio
    async def test_create_includes_timestamp(self, tmp_path: Path) -> None:
        from nibot.tools.admin_tools import SkillTool

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        loader = SkillsLoader([skills_dir])
        tool = SkillTool(loader)

        result = await tool.execute(
            action="create", name="test_tracked", body="Test body content",
            description="A tracked skill",
        )
        assert "created_at=" in result

        skill_file = skills_dir / "test_tracked" / "SKILL.md"
        content = skill_file.read_text(encoding="utf-8")
        assert "created_at:" in content
        assert "created_by: evolution" in content
        assert "version: 1" in content
