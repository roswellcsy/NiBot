"""Tests for session cache eviction write-back."""

import asyncio
import json

import pytest

from nibot.session import Session, SessionManager


class TestSessionEviction:
    def test_evicted_session_persists_to_disk(self, tmp_path):
        """When cache evicts a session, it must write it to disk first."""
        mgr = SessionManager(tmp_path / "sessions", max_cache_size=2)

        # Create 3 sessions with messages
        for i in range(3):
            s = mgr.get_or_create(f"sess_{i}")
            s.add_message("user", f"hello from session {i}")
            mgr.save(s)

        # Session 0 should have been evicted when session 2 was added
        assert "sess_0" not in mgr._cache

        # But it must still be on disk with its message
        reloaded = mgr._load("sess_0")
        assert reloaded is not None
        assert len(reloaded.messages) == 1
        assert reloaded.messages[0]["content"] == "hello from session 0"

    def test_eviction_preserves_unsaved_mutations(self, tmp_path):
        """Mutations made after save() are preserved on eviction."""
        mgr = SessionManager(tmp_path / "sessions", max_cache_size=2)

        s0 = mgr.get_or_create("sess_0")
        s0.add_message("user", "msg1")
        mgr.save(s0)

        # Add another message WITHOUT calling save()
        s0.add_message("assistant", "reply1")

        # Force eviction by adding 2 more sessions
        for i in range(1, 3):
            s = mgr.get_or_create(f"sess_{i}")
            mgr.save(s)

        # sess_0 evicted -- check disk has both messages
        reloaded = mgr._load("sess_0")
        assert reloaded is not None
        assert len(reloaded.messages) == 2
        assert reloaded.messages[1]["content"] == "reply1"

    def test_save_then_evict_no_data_loss(self, tmp_path):
        """Save → evict → reload cycle preserves all data."""
        mgr = SessionManager(tmp_path / "sessions", max_cache_size=2)

        s = mgr.get_or_create("target")
        for j in range(5):
            s.add_message("user", f"msg_{j}")
        mgr.save(s)

        # Evict by filling cache
        for i in range(2):
            filler = mgr.get_or_create(f"filler_{i}")
            mgr.save(filler)

        assert "target" not in mgr._cache

        # Reload from disk
        reloaded = mgr.get_or_create("target")
        assert len(reloaded.messages) == 5
        for j in range(5):
            assert reloaded.messages[j]["content"] == f"msg_{j}"

    def test_eviction_does_not_break_lock(self, tmp_path):
        """lock_for() still works after cache entry is evicted."""
        mgr = SessionManager(tmp_path / "sessions", max_cache_size=2)

        s = mgr.get_or_create("locktest")
        s.add_message("user", "hi")
        mgr.save(s)

        lock = mgr.lock_for("locktest")

        # Evict
        for i in range(2):
            mgr.get_or_create(f"other_{i}")

        assert "locktest" not in mgr._cache
        # Lock must still be the same object and usable
        assert mgr.lock_for("locktest") is lock

    def test_eviction_of_empty_session(self, tmp_path):
        """Empty sessions are also written to disk on eviction."""
        mgr = SessionManager(tmp_path / "sessions", max_cache_size=1)

        mgr.get_or_create("empty_sess")
        # No messages, no save -- just in cache

        # Evict by inserting another
        mgr.get_or_create("newer_sess")

        # File should exist with metadata
        path = mgr._path_for("empty_sess")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert data["_type"] == "metadata"
        assert data["key"] == "empty_sess"
