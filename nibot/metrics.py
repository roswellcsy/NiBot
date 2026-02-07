"""Session metrics -- structured analysis of conversation data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SessionMetrics:
    """Computed metrics for a single session."""

    message_count: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    unique_tools: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    error_rate: float = 0.0
    avg_response_length: float = 0.0
    tool_diversity: int = 0
    conversation_turns: int = 0
    first_message_at: str = ""
    last_message_at: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_count": self.message_count,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "unique_tools": self.unique_tools,
            "error_rate": round(self.error_rate, 3),
            "avg_response_length": round(self.avg_response_length, 1),
            "tool_diversity": self.tool_diversity,
            "conversation_turns": self.conversation_turns,
            "duration_seconds": round(self.duration_seconds, 1),
        }


def compute_session_metrics(messages: list[dict[str, Any]]) -> SessionMetrics:
    """Compute structured metrics from a list of session messages.

    Pure function. No I/O. No side effects.
    """
    m = SessionMetrics()
    m.message_count = len(messages)

    tools_seen: set[str] = set()
    assistant_lengths: list[int] = []
    timestamps: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        ts = msg.get("timestamp", "")
        if ts:
            timestamps.append(ts)

        if role == "user":
            m.user_messages += 1
        elif role == "assistant":
            m.assistant_messages += 1
            if content:
                assistant_lengths.append(len(content))
        elif role == "tool":
            m.tool_calls += 1
            tool_name = msg.get("name", "")
            if tool_name:
                tools_seen.add(tool_name)
            if "error" in content.lower():
                m.tool_errors += 1
                m.error_messages.append(content[:200])

    m.unique_tools = sorted(tools_seen)
    m.tool_diversity = len(tools_seen)
    m.conversation_turns = min(m.user_messages, m.assistant_messages)

    if m.tool_calls > 0:
        m.error_rate = m.tool_errors / m.tool_calls
    if assistant_lengths:
        m.avg_response_length = sum(assistant_lengths) / len(assistant_lengths)

    if timestamps:
        m.first_message_at = timestamps[0]
        m.last_message_at = timestamps[-1]
        try:
            t0 = datetime.fromisoformat(timestamps[0])
            t1 = datetime.fromisoformat(timestamps[-1])
            m.duration_seconds = (t1 - t0).total_seconds()
        except (ValueError, TypeError):
            pass

    return m


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple sessions."""

    session_count: int = 0
    total_messages: int = 0
    total_tool_calls: int = 0
    total_tool_errors: int = 0
    overall_error_rate: float = 0.0
    avg_turns_per_session: float = 0.0
    avg_duration_seconds: float = 0.0
    tool_usage: dict[str, int] = field(default_factory=dict)
    top_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_count": self.session_count,
            "total_messages": self.total_messages,
            "total_tool_calls": self.total_tool_calls,
            "total_tool_errors": self.total_tool_errors,
            "overall_error_rate": round(self.overall_error_rate, 3),
            "avg_turns_per_session": round(self.avg_turns_per_session, 1),
            "avg_duration_seconds": round(self.avg_duration_seconds, 1),
            "tool_usage": dict(sorted(self.tool_usage.items(), key=lambda x: -x[1])),
            "top_errors": self.top_errors[:10],
        }


def aggregate_metrics(per_session: list[SessionMetrics]) -> AggregateMetrics:
    """Aggregate metrics across multiple sessions.

    Pure function. No I/O.
    """
    agg = AggregateMetrics()
    agg.session_count = len(per_session)
    if not per_session:
        return agg

    tool_counts: dict[str, int] = {}
    error_strs: list[str] = []
    durations: list[float] = []
    turns: list[int] = []

    for sm in per_session:
        agg.total_messages += sm.message_count
        agg.total_tool_calls += sm.tool_calls
        agg.total_tool_errors += sm.tool_errors
        for t in sm.unique_tools:
            tool_counts[t] = tool_counts.get(t, 0) + 1
        error_strs.extend(sm.error_messages)
        if sm.duration_seconds > 0:
            durations.append(sm.duration_seconds)
        turns.append(sm.conversation_turns)

    if agg.total_tool_calls > 0:
        agg.overall_error_rate = agg.total_tool_errors / agg.total_tool_calls
    if turns:
        agg.avg_turns_per_session = sum(turns) / len(turns)
    if durations:
        agg.avg_duration_seconds = sum(durations) / len(durations)

    agg.tool_usage = tool_counts

    seen: set[str] = set()
    for e in error_strs:
        prefix = e[:80]
        if prefix not in seen:
            seen.add(prefix)
            agg.top_errors.append(prefix)

    return agg


@dataclass
class UsageBucket:
    """Usage statistics for a single time period."""

    period: str = ""   # "2026-02-07", "2026-W06", "2026-02"
    session_count: int = 0
    total_messages: int = 0
    total_tool_calls: int = 0
    total_tool_errors: int = 0
    error_rate: float = 0.0
    avg_duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "session_count": self.session_count,
            "total_messages": self.total_messages,
            "total_tool_calls": self.total_tool_calls,
            "total_tool_errors": self.total_tool_errors,
            "error_rate": round(self.error_rate, 3),
            "avg_duration_seconds": round(self.avg_duration_seconds, 1),
        }


def compute_usage_stats(
    sessions: list[tuple[datetime, SessionMetrics]],
    granularity: str = "day",
) -> list[UsageBucket]:
    """Group session metrics by time period.

    Pure function. No I/O.
    sessions: list of (updated_at, metrics) tuples.
    granularity: "day", "week", or "month".
    """
    buckets: dict[str, list[SessionMetrics]] = {}
    for updated_at, sm in sessions:
        if granularity == "week":
            iso = updated_at.isocalendar()
            key = f"{iso[0]}-W{iso[1]:02d}"
        elif granularity == "month":
            key = updated_at.strftime("%Y-%m")
        else:  # day
            key = updated_at.strftime("%Y-%m-%d")
        buckets.setdefault(key, []).append(sm)

    result: list[UsageBucket] = []
    for period in sorted(buckets):
        items = buckets[period]
        agg = aggregate_metrics(items)
        result.append(UsageBucket(
            period=period,
            session_count=agg.session_count,
            total_messages=agg.total_messages,
            total_tool_calls=agg.total_tool_calls,
            total_tool_errors=agg.total_tool_errors,
            error_rate=agg.overall_error_rate,
            avg_duration_seconds=agg.avg_duration_seconds,
        ))
    return result


def should_trigger_evolution(
    per_session: list[SessionMetrics],
    error_rate_threshold: float = 0.3,
    min_sessions: int = 5,
) -> tuple[bool, str]:
    """Decide whether error rate warrants an evolution run.

    Pure function. No I/O. Returns (should_trigger, reason).
    """
    if len(per_session) < min_sessions:
        return False, f"insufficient data ({len(per_session)}/{min_sessions} sessions)"
    agg = aggregate_metrics(per_session)
    if agg.overall_error_rate >= error_rate_threshold:
        return True, f"error_rate={agg.overall_error_rate:.2f} >= {error_rate_threshold}"
    return False, "metrics within normal range"
