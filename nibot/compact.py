"""Auto-compact -- LLM-based summarization of old conversation messages."""

from __future__ import annotations

from typing import Any

from nibot.log import logger
from nibot.provider import LLMProvider

COMPACT_PROMPT = (
    "Summarize the following conversation in 200-300 words. "
    "Preserve: key decisions, user preferences, task context, "
    "code/file references. Omit: greetings, acknowledgments, "
    "tool call details."
)


async def compact_messages(
    messages: list[dict[str, Any]],
    provider: LLMProvider,
    model: str = "",
) -> str:
    """Summarize old messages using a lightweight LLM call."""
    text = "\n".join(
        f"[{m.get('role', '?')}]: {(m.get('content') or '')[:500]}"
        for m in messages if m.get("content")
    )
    if not text.strip():
        return ""
    try:
        resp = await provider.chat(
            messages=[
                {"role": "system", "content": COMPACT_PROMPT},
                {"role": "user", "content": text},
            ],
            model=model,
            max_tokens=1024,
        )
        return (resp.content or "").strip()
    except Exception as e:
        logger.warning(f"Compact summarization failed: {e}")
        return ""
