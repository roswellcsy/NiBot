"""Session persistence -- JSONL storage with in-memory cache."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
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

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """Return recent messages in LLM dict format, preserving tool_calls/tool_call_id."""
        recent = self.messages[-max_messages:]
        return [{k: v for k, v in m.items() if k != "timestamp"} for m in recent]

    def clear(self) -> None:
        self.messages.clear()
        self.updated_at = datetime.now()


class SessionManager:
    """JSONL-backed session store with in-memory cache."""

    def __init__(self, sessions_dir: Path, max_cache_size: int = 200) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._max_cache_size = max_cache_size
        self._cache: OrderedDict[str, Session] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    def lock_for(self, key: str) -> asyncio.Lock:
        """Return a per-session lock for concurrent access protection."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _cache_put(self, key: str, session: Session) -> None:
        """Insert/update cache entry, evicting oldest if over limit.

        Note: locks are NOT evicted here. A lock may be held by a coroutine
        while its cache entry is evicted; cleaning it would break mutual exclusion.
        Lock objects are tiny (~64 bytes) so unbounded growth is acceptable.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = session
        while len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        session = self._load(key) or Session(key=key)
        self._cache_put(key, session)
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
        self._cache_put(session.key, session)

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
        updated_at = None
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "metadata":
                    created_at = datetime.fromisoformat(data["created_at"])
                    if "updated_at" in data:
                        updated_at = datetime.fromisoformat(data["updated_at"])
                else:
                    messages.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt session file {path}: {e}")
            return None
        ca = created_at or datetime.now()
        return Session(key=key, messages=messages, created_at=ca, updated_at=updated_at or ca)

    def iter_recent_from_disk(self, limit: int = 50) -> list[Session]:
        """Read sessions from disk sorted by mtime, WITHOUT caching them.

        Used by analytics/evolution code that needs a snapshot but shouldn't
        bloat the cache.  Returns from cache when a session is already loaded.
        """
        paths = sorted(
            self.sessions_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        results: list[Session] = []
        for path in paths:
            key = self._read_key_from_file(path)
            if not key:
                continue
            if key in self._cache:
                results.append(self._cache[key])
            else:
                session = self._load(key)
                if session:
                    results.append(session)
        return results

    def archive(self, key: str) -> bool:
        """Move a session file to the archive subdirectory. Remove from cache."""
        src = self._path_for(key)
        if not src.exists():
            return False
        archive_dir = self.sessions_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        dst = archive_dir / src.name
        src.rename(dst)
        self._cache.pop(key, None)
        return True

    def archive_old(self, days: int = 30) -> list[str]:
        """Archive sessions not updated in `days` days."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        archived: list[str] = []
        for path in list(self.sessions_dir.glob("*.jsonl")):
            key = self._read_key_from_file(path)
            if not key:
                continue
            updated_at = self._read_updated_at(path)
            if updated_at and updated_at < cutoff:
                if self.archive(key):
                    archived.append(key)
        return archived

    def _read_updated_at(self, path: Path) -> datetime | None:
        """Read updated_at from the metadata line without loading full session."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            data = json.loads(first_line)
            if data.get("_type") == "metadata" and "updated_at" in data:
                return datetime.fromisoformat(data["updated_at"])
        except (json.JSONDecodeError, ValueError, OSError):
            pass
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            return None

    def list_archived(self) -> list[str]:
        """List keys of archived sessions."""
        archive_dir = self.sessions_dir / "archive"
        if not archive_dir.exists():
            return []
        keys: list[str] = []
        for path in archive_dir.glob("*.jsonl"):
            key = self._read_key_from_file(path)
            if key:
                keys.append(key)
        return keys

    def search(self, query: str, max_results: int = 20) -> list[SearchHit]:
        """Search across all sessions for a keyword."""
        return search_sessions(self.sessions_dir, query, max_results=max_results)

    def iter_all_from_disk(self) -> list[Session]:
        """Read ALL sessions from disk. Does NOT pollute cache.

        Used by analytics that need complete history (usage stats).
        Returns cached sessions when available.
        """
        results: list[Session] = []
        for path in self.sessions_dir.glob("*.jsonl"):
            key = self._read_key_from_file(path)
            if not key:
                continue
            if key in self._cache:
                results.append(self._cache[key])
            else:
                session = self._load(key)
                if session:
                    results.append(session)
        return results

    def query_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return summaries of recent sessions with structured metrics."""
        from nibot.metrics import compute_session_metrics

        sessions = self.iter_recent_from_disk(limit)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        result: list[dict[str, Any]] = []
        for s in sessions[:limit]:
            metrics = compute_session_metrics(s.messages)
            last_user = next(
                (m["content"][:200] for m in reversed(s.messages) if m.get("role") == "user"),
                "",
            )
            result.append({
                "key": s.key,
                "messages": metrics.message_count,
                "tool_calls": metrics.tool_calls,
                "errors": metrics.tool_errors,
                "updated_at": s.updated_at.isoformat(),
                "last_user_msg": last_user,
                "metrics": metrics.to_dict(),
            })
        return result

    def get_session_messages(self, key: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return the last N messages from a session."""
        session = self._cache.get(key)
        if not session:
            session = self._load(key)
            if not session:
                return []
        return session.messages[-limit:]

    def _load_all(self) -> None:
        """Load session files into cache (respects max_cache_size).

        Prefer iter_recent_from_disk() for analytics queries.
        """
        for path in self.sessions_dir.glob("*.jsonl"):
            key = self._read_key_from_file(path)
            if key and key not in self._cache:
                session = self._load(key)
                if session:
                    self._cache_put(key, session)

    def _read_key_from_file(self, path: Path) -> str | None:
        """Read session key from the metadata line of a JSONL file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            data = json.loads(first_line)
            if data.get("_type") == "metadata":
                return data.get("key") or None
        except (json.JSONDecodeError, ValueError, OSError):
            pass
        return None

    def _path_for(self, key: str) -> Path:
        safe = key
        for ch in r':/<>\|"?*':
            safe = safe.replace(ch, "_")
        return self.sessions_dir / f"{safe}.jsonl"


# ---- Cross-session search (v0.10.0) ----


@dataclass
class SearchHit:
    """A single search result."""

    session_key: str
    role: str
    content_preview: str
    timestamp: str
    message_index: int


def search_sessions(
    sessions_dir: Path,
    query: str,
    max_results: int = 20,
) -> list[SearchHit]:
    """Full-text search across all session JSONL files.

    Scans files line-by-line without loading entire sessions into memory.
    Case-insensitive substring match. Early exit on max_results.
    """
    if not query:
        return []
    query_lower = query.lower()
    hits: list[SearchHit] = []

    for path in sessions_dir.glob("*.jsonl"):
        session_key: str | None = None
        msg_idx = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        session_key = data.get("key", path.stem)
                        continue
                    content = data.get("content") or ""
                    if query_lower in content.lower():
                        hits.append(SearchHit(
                            session_key=session_key or path.stem,
                            role=data.get("role", "?"),
                            content_preview=content[:200],
                            timestamp=data.get("timestamp", ""),
                            message_index=msg_idx,
                        ))
                        if len(hits) >= max_results:
                            return hits
                    msg_idx += 1
        except (json.JSONDecodeError, OSError):
            continue
    return hits


# ---- Session export (v0.10.0) ----


def format_session_export(session: Session, fmt: str = "markdown") -> str:
    """Export a session to a readable format. Pure function.

    Formats: "markdown", "json", "html".
    """
    if fmt == "json":
        return _export_json(session)
    if fmt == "html":
        return _export_html(session)
    return _export_markdown(session)


def _export_markdown(session: Session) -> str:
    lines: list[str] = [
        f"# Session: {session.key}",
        f"Created: {session.created_at.isoformat()}",
        f"Updated: {session.updated_at.isoformat()}",
        f"Messages: {len(session.messages)}",
        "", "---", "",
    ]
    for msg in session.messages:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        ts = msg.get("timestamp", "")
        header = f"**{role}**"
        if ts:
            header += f" ({ts})"
        lines.append(header)
        lines.append("")
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", tc.get("name", "?"))
                args = func.get("arguments", tc.get("arguments", ""))
                lines.append(f"Tool call: `{name}`")
                lines.append(f"```json\n{args}\n```")
        elif role == "tool":
            tool_name = msg.get("name", "")
            lines.append(f"Tool result ({tool_name}):")
            lines.append(f"```\n{content[:2000]}\n```")
        else:
            lines.append(content)
        lines.extend(["", "---", ""])
    return "\n".join(lines)


def _export_json(session: Session) -> str:
    data = {
        "key": session.key,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "message_count": len(session.messages),
        "messages": session.messages,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _export_html(session: Session) -> str:
    """Minimal HTML export. No external dependencies."""
    from html import escape
    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Session: {escape(session.key)}</title>",
        "<style>body{font-family:sans-serif;max-width:800px;margin:auto;padding:20px}",
        ".msg{border-bottom:1px solid #eee;padding:10px 0}",
        ".role{font-weight:bold}.user{color:#0066cc}",
        ".assistant{color:#009933}.tool{color:#996600}",
        "pre{background:#f5f5f5;padding:10px;overflow-x:auto}</style></head><body>",
        f"<h1>Session: {escape(session.key)}</h1>",
        f"<p>Created: {escape(session.created_at.isoformat())} | "
        f"Updated: {escape(session.updated_at.isoformat())} | "
        f"Messages: {len(session.messages)}</p><hr>",
    ]
    for msg in session.messages:
        role = msg.get("role", "unknown")
        content = escape(msg.get("content") or "")[:2000]
        ts = msg.get("timestamp", "")
        parts.append(f'<div class="msg"><span class="role {escape(role)}">{escape(role)}</span>')
        if ts:
            parts.append(f" <small>({escape(ts)})</small>")
        parts.append(f"<p>{content}</p></div>")
    parts.append("</body></html>")
    return "".join(parts)
