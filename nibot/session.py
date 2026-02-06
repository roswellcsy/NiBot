"""Session persistence -- JSONL storage with in-memory cache."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nibot.log import logger


@dataclass
class Session:
    """Conversation session. Empty session == new session, no special-casing."""

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self.messages.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **kwargs}
        )
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 50) -> list[dict[str, str]]:
        """Return recent messages in LLM dict format."""
        recent = self.messages[-max_messages:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self) -> None:
        self.messages.clear()
        self.updated_at = datetime.now()


class SessionManager:
    """JSONL-backed session store with in-memory cache."""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]
        session = self._load(key) or Session(key=key)
        self._cache[key] = session
        return session

    def save(self, session: Session) -> None:
        path = self._path_for(session.key)
        with open(path, "w", encoding="utf-8") as f:
            meta = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._cache[session.key] = session

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)
        path = self._path_for(key)
        if path.exists():
            path.unlink()

    def _load(self, key: str) -> Session | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        messages: list[dict[str, Any]] = []
        created_at = None
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "metadata":
                    created_at = datetime.fromisoformat(data["created_at"])
                else:
                    messages.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt session file {path}: {e}")
            return None
        return Session(key=key, messages=messages, created_at=created_at or datetime.now())

    def _path_for(self, key: str) -> Path:
        safe = key.replace(":", "_").replace("/", "_").replace("\\", "_")
        return self.sessions_dir / f"{safe}.jsonl"
