"""Memory FIFO rotation tests (Phase 4 v1.4)."""
from __future__ import annotations

from nibot.memory import MemoryStore, _MEMORY_MAX_LINES


class TestMemoryRotation:
    def test_rotation_caps_at_max(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        for i in range(1200):
            store.append_memory(f"line-{i}")
        lines = store.read_memory().splitlines()
        assert len(lines) == _MEMORY_MAX_LINES

    def test_rotation_keeps_newest(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        for i in range(1100):
            store.append_memory(f"line-{i}")
        lines = store.read_memory().splitlines()
        # Oldest kept should be line-100 (1100 - 1000)
        assert lines[0] == "line-100"
        assert lines[-1] == "line-1099"

    def test_small_file_untouched(self, tmp_path):
        store = MemoryStore(tmp_path / "mem")
        for i in range(50):
            store.append_memory(f"line-{i}")
        lines = store.read_memory().splitlines()
        assert len(lines) == 50
        assert lines[0] == "line-0"
