"""Dual-layer memory -- MEMORY.md (long-term) + daily notes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

_MEMORY_MAX_LINES = 1000


class MemoryStore:
    """Workspace memory: MEMORY.md for persistent facts, YYYY-MM-DD.md for daily notes."""

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _memory_file(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    def _daily_file(self, date: datetime | None = None) -> Path:
        d = date or datetime.now()
        return self.memory_dir / f"{d.strftime('%Y-%m-%d')}.md"

    def get_context(self) -> str:
        """Build memory context for system prompt injection."""
        parts: list[str] = []
        if self._memory_file.exists():
            content = self._memory_file.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"## Long-term Memory\n{content}")
        daily = self._daily_file()
        if daily.exists():
            content = daily.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"## Today's Notes ({datetime.now().strftime('%Y-%m-%d')})\n{content}")
        return "\n\n".join(parts)

    def read_memory(self) -> str:
        if self._memory_file.exists():
            return self._memory_file.read_text(encoding="utf-8")
        return ""

    def write_memory(self, content: str) -> None:
        self._memory_file.write_text(content, encoding="utf-8")

    def append_memory(self, line: str) -> None:
        lines: list[str] = []
        if self._memory_file.exists():
            lines = self._memory_file.read_text(encoding="utf-8").splitlines()
        lines.append(line.rstrip("\n"))
        if len(lines) > _MEMORY_MAX_LINES:
            lines = lines[-_MEMORY_MAX_LINES:]
        self._memory_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def read_daily(self, date: datetime | None = None) -> str:
        path = self._daily_file(date)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def append_daily(self, line: str, date: datetime | None = None) -> None:
        path = self._daily_file(date)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
