"""Context builder -- composable system prompt assembly."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

from nibot.config import NiBotConfig
from nibot.log import logger
from nibot.memory import MemoryStore
from nibot.session import Session, SessionManager
from nibot.skills import SkillsLoader
from nibot.types import Envelope


def _estimate_tokens(messages: list[dict[str, Any]], model: str = "") -> int:
    """Estimate token count. Use litellm if available, else rough 4-chars-per-token."""
    try:
        from litellm import token_counter
        return token_counter(model=model, messages=messages)
    except Exception:
        total = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
        return total // 4


class ContextBuilder:
    """Assemble LLM message list from session history, memory, and skills."""

    COMPACT_DROP_THRESHOLD = 10  # trigger compact when this many messages are dropped

    def __init__(
        self,
        config: NiBotConfig,
        memory: MemoryStore,
        skills: SkillsLoader,
        workspace: Path,
        provider: Any | None = None,
        sessions: SessionManager | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.skills = skills
        self.workspace = workspace
        self._provider = provider
        self._sessions = sessions
        self._compact_tasks: set[asyncio.Task[Any]] = set()
        self._compacting_sessions: set[str] = set()

    def build(self, session: Session, current: Envelope) -> list[dict[str, Any]]:
        system_msg = {
            "role": "system",
            "content": self._build_system_prompt(current.channel, current.chat_id),
        }
        user_msg = {"role": "user", "content": self._build_user_content(current)}

        # Token budget: context_window - reserve - system - user = budget for history
        model = self.config.agent.model
        budget = self.config.agent.context_window - self.config.agent.context_reserve
        fixed_cost = _estimate_tokens([system_msg, user_msg], model)
        history_budget = max(0, budget - fixed_cost)

        # Fill history from most recent backwards until budget exhausted
        all_history = session.get_history()
        kept: list[dict[str, Any]] = []
        used = 0
        for msg in reversed(all_history):
            cost = _estimate_tokens([msg], model)
            if used + cost > history_budget:
                break
            kept.append(msg)
            used += cost
        kept.reverse()

        dropped_count = len(all_history) - len(kept)

        result = [system_msg]
        if session.compacted_summary:
            result.append({
                "role": "system",
                "content": f"[Earlier conversation summary]\n{session.compacted_summary}",
            })
        result += kept + [user_msg]

        # Schedule background compaction when many messages are silently dropped
        if (
            dropped_count > self.COMPACT_DROP_THRESHOLD
            and not session.compacted_summary
            and self._provider
            and session.key not in self._compacting_sessions
        ):
            dropped = all_history[:dropped_count]
            self._schedule_compact(session.key, dropped)

        return result

    def _build_system_prompt(self, channel: str = "", chat_id: str = "") -> str:
        sections: list[str] = []

        # Layer 1: Identity (bootstrap files)
        for fname in self.config.agent.bootstrap_files:
            path = self.workspace / fname
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    sections.append(content)

        # Layer 2: Runtime context
        sections.append(f"Current time: {datetime.now().isoformat()}")
        if channel:
            sections.append(f"Current session: {channel}:{chat_id}")

        # Layer 3: Memory
        mem = self.memory.get_context()
        if mem:
            sections.append(mem)

        # Layer 4: Skills (always-skills inline, others as summary)
        for skill in self.skills.get_always_skills():
            sections.append(f"## Skill: {skill.name}\n{skill.body}")
        summary = self.skills.build_summary()
        if summary:
            sections.append(
                "## Available Skills\n"
                "To use a skill, read its SKILL.md with read_file.\n" + summary
            )

        # Layer 5: Shared thoughts (inter-agent context)
        thoughts = self._read_thoughts()
        if thoughts:
            sections.append("## Shared Context (thoughts/)\n" + thoughts)

        return "\n\n---\n\n".join(sections)

    def _build_user_content(self, envelope: Envelope) -> Any:
        if not envelope.media:
            return envelope.content
        parts: list[dict[str, Any]] = []
        for media_path in envelope.media:
            b64 = self._encode_media(media_path)
            if b64:
                mime = mimetypes.guess_type(media_path)[0] or "image/png"
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
        parts.append({"type": "text", "text": envelope.content})
        return parts

    def _read_thoughts(self) -> str:
        """Read recent thoughts/*.md files for inter-agent context sharing."""
        thoughts_dir = self.workspace / "thoughts"
        if not thoughts_dir.exists():
            return ""
        files = sorted(thoughts_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
        parts: list[str] = []
        budget = 6000
        for f in files[:8]:
            try:
                content = f.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if len(content) > budget:
                content = content[:budget] + "...(truncated)"
            parts.append(f"### {f.stem}\n{content}")
            budget -= len(content)
            if budget <= 0:
                break
        return "\n\n".join(parts)

    def _schedule_compact(self, session_key: str, dropped: list[dict[str, Any]]) -> None:
        """Fire-and-forget: summarize dropped messages and save to session."""
        self._compacting_sessions.add(session_key)

        async def _do_compact() -> None:
            try:
                from nibot.compact import compact_messages

                summary = await compact_messages(dropped, self._provider)
                if not summary:
                    return
                if not self._sessions:
                    return
                session = self._sessions.get_or_create(session_key)
                session.compacted_summary = summary
                self._sessions.save(session)
                logger.info(f"Auto-compact: saved summary for {session_key} ({len(summary)} chars)")
            finally:
                self._compacting_sessions.discard(session_key)

        task = asyncio.create_task(_do_compact())
        self._compact_tasks.add(task)
        task.add_done_callback(self._compact_tasks.discard)

    def _encode_media(self, path: str) -> str | None:
        p = Path(path)
        if not p.exists():
            return None
        return base64.b64encode(p.read_bytes()).decode("ascii")
