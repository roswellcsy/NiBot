"""Structured event log -- JSONL append-only for operational analytics."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLog:
    """Append-only JSONL event log for cost, latency, and decision tracking.

    Four event types:
      llm_call       -- per-provider API call (tokens, latency, success)
      tool_call      -- tool execution (duration, success)
      provider_switch -- provider selection decision (chain, skipped)
      request        -- end-to-end request processing
    """

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self._path = path
        self._enabled = enabled
        self._lock = threading.Lock()

    def log_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        error: str = "",
    ) -> None:
        data: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
            "success": success,
        }
        if error:
            data["error"] = error
        self._append("llm_call", data)

    def log_tool_call(
        self,
        tool: str,
        duration_ms: float,
        success: bool,
        error: str = "",
    ) -> None:
        data: dict[str, Any] = {
            "tool": tool,
            "duration_ms": round(duration_ms, 1),
            "success": success,
        }
        if error:
            data["error"] = error
        self._append("tool_call", data)

    def log_provider_switch(
        self,
        chain: list[str],
        selected: str,
        skipped: list[str],
        reason: str,
    ) -> None:
        self._append("provider_switch", {
            "chain": chain,
            "selected": selected,
            "skipped": skipped,
            "reason": reason,
        })

    def log_request(
        self,
        channel: str,
        session_key: str,
        latency_ms: float,
        tool_count: int,
        total_tokens: int,
        provider: str,
    ) -> None:
        self._append("request", {
            "channel": channel,
            "session_key": session_key,
            "latency_ms": round(latency_ms, 1),
            "tool_count": tool_count,
            "total_tokens": total_tokens,
            "provider": provider,
        })

    def _append(self, event_type: str, data: dict[str, Any]) -> None:
        if not self._enabled:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "type": event_type,
            **data,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # never crash the hot path for logging
