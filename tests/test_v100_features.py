"""Tests for v0.10.0 features: session management enhancements."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nibot.metrics import (
    SessionMetrics,
    UsageBucket,
    aggregate_metrics,
    compute_session_metrics,
    compute_usage_stats,
)
from nibot.session import (
    Session,
    SessionManager,
    SearchHit,
    format_session_export,
    search_sessions,
)
from nibot.tools.analyze_tool import AnalyzeTool


# ---- helpers ----

def _make_session(key: str, messages: list[dict], days_ago: int = 0) -> Session:
    """Create a Session with controlled timestamps."""
    now = datetime.now()
    created = now - timedelta(days=days_ago + 1)
    updated = now - timedelta(days=days_ago)
    return Session(key=key, messages=messages, created_at=created, updated_at=updated)


def _save_session(sm: SessionManager, session: Session) -> None:
    """Save a session via the SessionManager."""
    sm.save(session)


def _basic_messages(n: int = 3) -> list[dict]:
    """Create a simple conversation."""
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"Question {i}", "timestamp": datetime.now().isoformat()})
        msgs.append({"role": "assistant", "content": f"Answer {i}", "timestamp": datetime.now().isoformat()})
    return msgs


# ---- C7: Usage Statistics ----

class TestComputeUsageStats:
    def test_empty_input(self) -> None:
        result = compute_usage_stats([], granularity="day")
        assert result == []

    def test_day_granularity(self) -> None:
        d1 = datetime(2026, 1, 15)
        d2 = datetime(2026, 1, 15, 14, 0)
        d3 = datetime(2026, 1, 16)
        sm1 = SessionMetrics(message_count=10, tool_calls=2)
        sm2 = SessionMetrics(message_count=5, tool_calls=1)
        sm3 = SessionMetrics(message_count=8, tool_calls=3)
        result = compute_usage_stats([(d1, sm1), (d2, sm2), (d3, sm3)], granularity="day")
        assert len(result) == 2
        assert result[0].period == "2026-01-15"
        assert result[0].session_count == 2
        assert result[0].total_messages == 15
        assert result[1].period == "2026-01-16"
        assert result[1].session_count == 1

    def test_week_granularity(self) -> None:
        d1 = datetime(2026, 1, 5)  # Week 2
        d2 = datetime(2026, 1, 12)  # Week 3
        sm = SessionMetrics(message_count=5)
        result = compute_usage_stats([(d1, sm), (d2, sm)], granularity="week")
        assert len(result) == 2
        assert "-W" in result[0].period

    def test_month_granularity(self) -> None:
        d1 = datetime(2026, 1, 15)
        d2 = datetime(2026, 2, 10)
        sm = SessionMetrics(message_count=5)
        result = compute_usage_stats([(d1, sm), (d2, sm)], granularity="month")
        assert len(result) == 2
        assert result[0].period == "2026-01"
        assert result[1].period == "2026-02"

    def test_bucket_to_dict(self) -> None:
        b = UsageBucket(period="2026-01", session_count=3, error_rate=0.12345)
        d = b.to_dict()
        assert d["period"] == "2026-01"
        assert d["error_rate"] == 0.123

    def test_same_day_aggregation(self) -> None:
        d = datetime(2026, 1, 15)
        sm1 = SessionMetrics(message_count=10, tool_calls=5, tool_errors=1)
        sm2 = SessionMetrics(message_count=20, tool_calls=10, tool_errors=3)
        result = compute_usage_stats([(d, sm1), (d, sm2)], granularity="day")
        assert len(result) == 1
        assert result[0].total_tool_calls == 15
        assert result[0].total_tool_errors == 4


class TestIterAllFromDisk:
    def test_returns_all_sessions(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        for i in range(5):
            _save_session(sm, _make_session(f"s{i}", _basic_messages(1)))
        # Clear cache to force disk read
        sm._cache.clear()
        result = sm.iter_all_from_disk()
        assert len(result) == 5

    def test_does_not_pollute_cache(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions", max_cache_size=2)
        for i in range(5):
            _save_session(sm, _make_session(f"s{i}", _basic_messages(1)))
        sm._cache.clear()
        result = sm.iter_all_from_disk()
        assert len(result) == 5
        assert len(sm._cache) == 0  # cache not polluted

    def test_returns_cached_when_available(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        session = _make_session("cached", _basic_messages(1))
        _save_session(sm, session)
        # Session is in cache
        result = sm.iter_all_from_disk()
        assert len(result) == 1
        assert result[0] is sm._cache["cached"]


class TestUsageStatsViaAnalyzeTool:
    @pytest.mark.asyncio
    async def test_usage_stats_action(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        for i in range(3):
            _save_session(sm, _make_session(f"s{i}", _basic_messages(2)))
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="usage_stats", granularity="day")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1

    @pytest.mark.asyncio
    async def test_usage_stats_empty(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="usage_stats")
        assert "No sessions" in result


# ---- C4: Session Export ----

class TestSessionExport:
    def test_markdown_basic(self) -> None:
        session = _make_session("test:1", [
            {"role": "user", "content": "Hello", "timestamp": "2026-01-01T10:00:00"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2026-01-01T10:00:01"},
        ])
        result = format_session_export(session, fmt="markdown")
        assert "# Session: test:1" in result
        assert "**user**" in result
        assert "Hello" in result
        assert "Hi there!" in result

    def test_json_roundtrip(self) -> None:
        session = _make_session("test:2", _basic_messages(2))
        result = format_session_export(session, fmt="json")
        parsed = json.loads(result)
        assert parsed["key"] == "test:2"
        assert parsed["message_count"] == 4
        assert len(parsed["messages"]) == 4

    def test_html_basic(self) -> None:
        session = _make_session("test:3", [
            {"role": "user", "content": "Hello <script>alert(1)</script>"},
        ])
        result = format_session_export(session, fmt="html")
        assert "<!DOCTYPE html>" in result
        assert "&lt;script&gt;" in result  # XSS escaped
        assert "<script>" not in result

    def test_empty_session(self) -> None:
        session = _make_session("empty", [])
        result = format_session_export(session, fmt="markdown")
        assert "# Session: empty" in result
        assert "Messages: 0" in result

    def test_with_tool_calls(self) -> None:
        session = _make_session("tools", [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'}}
            ]},
            {"role": "tool", "name": "read_file", "content": "file contents here"},
        ])
        result = format_session_export(session, fmt="markdown")
        assert "`read_file`" in result
        assert "Tool result (read_file)" in result

    def test_default_format_is_markdown(self) -> None:
        session = _make_session("default", [{"role": "user", "content": "hi"}])
        result = format_session_export(session)
        assert "# Session:" in result


class TestExportViaAnalyzeTool:
    @pytest.mark.asyncio
    async def test_export_action(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("exp:1", _basic_messages(2)))
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="export", session_key="exp:1", format="json")
        parsed = json.loads(result)
        assert parsed["key"] == "exp:1"

    @pytest.mark.asyncio
    async def test_export_not_found(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="export", session_key="nonexistent")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_export_no_key(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="export")
        assert "required" in result


# ---- C5: Cross-Session Search ----

class TestSearchSessions:
    def test_basic_match(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("s1", [
            {"role": "user", "content": "How do I deploy to kubernetes?"},
        ]))
        _save_session(sm, _make_session("s2", [
            {"role": "user", "content": "What is docker?"},
        ]))
        hits = search_sessions(sm.sessions_dir, "kubernetes")
        assert len(hits) == 1
        assert hits[0].session_key == "s1"
        assert "kubernetes" in hits[0].content_preview.lower()

    def test_case_insensitive(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("s1", [
            {"role": "user", "content": "ERROR: something failed"},
        ]))
        hits = search_sessions(sm.sessions_dir, "error")
        assert len(hits) == 1

    def test_no_results(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("s1", _basic_messages(1)))
        hits = search_sessions(sm.sessions_dir, "xyznonexistent")
        assert hits == []

    def test_empty_query(self, tmp_path: Path) -> None:
        hits = search_sessions(tmp_path, "")
        assert hits == []

    def test_max_results_limit(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        # Create 5 sessions each with "findme" keyword
        for i in range(5):
            _save_session(sm, _make_session(f"s{i}", [
                {"role": "user", "content": f"findme message {i}"},
            ]))
        hits = search_sessions(sm.sessions_dir, "findme", max_results=3)
        assert len(hits) == 3

    def test_corrupt_file_skipped(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        # Valid session
        sm = SessionManager(sessions_dir)
        _save_session(sm, _make_session("good", [
            {"role": "user", "content": "searchable content"},
        ]))
        # Corrupt file
        (sessions_dir / "bad.jsonl").write_text("not json at all{{{", encoding="utf-8")
        hits = search_sessions(sessions_dir, "searchable")
        assert len(hits) == 1
        assert hits[0].session_key == "good"

    def test_multiple_hits_across_sessions(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("s1", [
            {"role": "user", "content": "deploy to production"},
        ]))
        _save_session(sm, _make_session("s2", [
            {"role": "assistant", "content": "to deploy, run this command"},
        ]))
        hits = search_sessions(sm.sessions_dir, "deploy")
        assert len(hits) == 2


class TestSearchViaAnalyzeTool:
    @pytest.mark.asyncio
    async def test_search_action(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("s1", [
            {"role": "user", "content": "how to configure nginx"},
        ]))
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="search", query="nginx")
        assert "nginx" in result
        assert "1 hits" in result

    @pytest.mark.asyncio
    async def test_search_no_query(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="search")
        assert "required" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("s1", _basic_messages(1)))
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="search", query="xyznonexistent")
        assert "No results" in result


# ---- C6: Session Archive ----

class TestSessionArchive:
    def test_archive_single_session(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("arc:1", _basic_messages(1)))
        assert sm.archive("arc:1") is True
        archive_dir = tmp_path / "sessions" / "archive"
        assert archive_dir.exists()
        assert len(list(archive_dir.glob("*.jsonl"))) == 1
        # Original gone
        assert not sm._path_for("arc:1").exists()

    def test_archive_removes_from_cache(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("arc:2", _basic_messages(1)))
        assert "arc:2" in sm._cache
        sm.archive("arc:2")
        assert "arc:2" not in sm._cache

    def test_archive_nonexistent(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        assert sm.archive("nonexistent") is False

    def test_archive_old_by_days(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        # Save old session (60 days ago)
        old = _make_session("old", _basic_messages(1), days_ago=60)
        _save_session(sm, old)
        # Save recent session
        recent = _make_session("recent", _basic_messages(1), days_ago=0)
        _save_session(sm, recent)

        archived = sm.archive_old(days=30)
        assert "old" in archived
        assert "recent" not in archived

    def test_archive_old_keeps_recent(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("fresh", _basic_messages(1), days_ago=0))
        archived = sm.archive_old(days=30)
        assert archived == []

    def test_archive_old_empty_dir(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        archived = sm.archive_old(days=30)
        assert archived == []

    def test_list_archived(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("a1", _basic_messages(1)))
        _save_session(sm, _make_session("a2", _basic_messages(1)))
        sm.archive("a1")
        sm.archive("a2")
        keys = sm.list_archived()
        assert sorted(keys) == ["a1", "a2"]

    def test_search_excludes_archived(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("active", [
            {"role": "user", "content": "find this keyword"},
        ]))
        _save_session(sm, _make_session("archived", [
            {"role": "user", "content": "find this keyword too"},
        ]))
        sm.archive("archived")
        hits = sm.search("keyword")
        assert len(hits) == 1
        assert hits[0].session_key == "active"

    def test_archive_creates_subdir(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("sub", _basic_messages(1)))
        archive_dir = tmp_path / "sessions" / "archive"
        assert not archive_dir.exists()
        sm.archive("sub")
        assert archive_dir.exists()


class TestArchiveViaAnalyzeTool:
    @pytest.mark.asyncio
    async def test_archive_specific(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("tool:1", _basic_messages(1)))
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="archive", session_key="tool:1")
        assert "archived" in result

    @pytest.mark.asyncio
    async def test_archive_auto(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("old:1", _basic_messages(1), days_ago=60))
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="archive", days=30)
        assert "Archived 1" in result

    @pytest.mark.asyncio
    async def test_archive_nothing_old(self, tmp_path: Path) -> None:
        sm = SessionManager(tmp_path / "sessions")
        _save_session(sm, _make_session("fresh", _basic_messages(1), days_ago=0))
        tool = AnalyzeTool(sm)
        result = await tool.execute(action="archive", days=30)
        assert "No sessions" in result
