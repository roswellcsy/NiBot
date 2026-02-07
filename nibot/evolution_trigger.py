"""Event-driven evolution trigger -- fires evolution when error rate exceeds threshold."""

from __future__ import annotations

import time
from typing import Any

from nibot.log import logger
from nibot.metrics import SessionMetrics, compute_session_metrics, should_trigger_evolution


class EvolutionTrigger:
    """Check recent session metrics after each conversation; fire evolution if error rate is high.

    Integrates with AgentLoop: called after session save via ``asyncio.create_task(trigger.check())``.
    """

    def __init__(
        self,
        bus: Any,
        sessions: Any,
        cooldown_seconds: int = 3600,
        error_rate_threshold: float = 0.3,
        min_sessions: int = 5,
        enabled: bool = False,
        window: int = 20,
    ) -> None:
        self._bus = bus
        self._sessions = sessions
        self._cooldown = cooldown_seconds
        self._threshold = error_rate_threshold
        self._min_sessions = min_sessions
        self.enabled = enabled
        self._window = window
        self._last_trigger: float = 0.0

    async def check(self) -> bool:
        """Evaluate recent metrics and maybe publish an evolution request.

        Returns True if evolution was triggered.
        """
        if not self.enabled:
            return False
        now = time.monotonic()
        if now - self._last_trigger < self._cooldown:
            return False

        try:
            recent = self._sessions.iter_recent_from_disk(limit=self._window)
            recent.sort(key=lambda s: s.updated_at, reverse=True)
            per_session: list[SessionMetrics] = [
                compute_session_metrics(s.messages) for s in recent
            ]
            triggered, reason = should_trigger_evolution(
                per_session,
                error_rate_threshold=self._threshold,
                min_sessions=self._min_sessions,
            )
        except Exception as e:
            logger.warning(f"EvolutionTrigger.check() failed: {e}")
            return False

        if not triggered:
            return False

        self._last_trigger = now
        logger.info(f"EvolutionTrigger fired: {reason}")

        from nibot.types import Envelope
        await self._bus.publish_inbound(Envelope(
            channel="evolution_trigger",
            sender_id="system",
            chat_id="evolution",
            content=(
                f"[Auto-triggered evolution] Reason: {reason}\n\n"
                "Analyze errors, review skill inventory, and take action. "
                "Log your decision via analyze(action='log_decision', ...)."
            ),
        ))
        return True
