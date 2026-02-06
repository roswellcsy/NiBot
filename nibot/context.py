"""Context builder -- composable system prompt assembly."""

from __future__ import annotations

import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

from nibot.config import NiBotConfig
from nibot.memory import MemoryStore
from nibot.session import Session
from nibot.skills import SkillsLoader
from nibot.types import Envelope

BOOTSTRAP_FILES = ["IDENTITY.md", "AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]


class ContextBuilder:
    """Assemble LLM message list from session history, memory, and skills."""

    def __init__(
        self,
        config: NiBotConfig,
        memory: MemoryStore,
        skills: SkillsLoader,
        workspace: Path,
    ) -> None:
        self.config = config
        self.memory = memory
        self.skills = skills
        self.workspace = workspace

    def build(self, session: Session, current: Envelope) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        messages.append({
            "role": "system",
            "content": self._build_system_prompt(current.channel, current.chat_id),
        })
        messages.extend(session.get_history())
        messages.append({"role": "user", "content": self._build_user_content(current)})
        return messages

    def _build_system_prompt(self, channel: str = "", chat_id: str = "") -> str:
        sections: list[str] = []

        # Layer 1: Identity (bootstrap files)
        for fname in BOOTSTRAP_FILES:
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

    def _encode_media(self, path: str) -> str | None:
        p = Path(path)
        if not p.exists():
            return None
        return base64.b64encode(p.read_bytes()).decode("ascii")
