"""Analyze tool -- conversation analysis for evolution agents."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from nibot.metrics import aggregate_metrics, compute_session_metrics
from nibot.registry import Tool
from nibot.session import SessionManager


class AnalyzeTool(Tool):
    """Analyze recent conversations to identify patterns, errors, and skill gaps."""

    def __init__(self, sessions: SessionManager, skills: Any = None,
                 evolution_log: Any = None) -> None:
        self._sessions = sessions
        self._skills = skills
        self._evolution_log = evolution_log

    @property
    def name(self) -> str:
        return "analyze"

    @property
    def description(self) -> str:
        return (
            "Analyze recent conversations: summarize sessions, list errors, "
            "compute metrics, or measure skill impact."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["summary", "errors", "session_detail", "metrics", "skill_impact",
                            "log_decision", "decision_history",
                            "export", "search", "archive", "usage_stats"],
                    "description": (
                        "summary: overview of recent sessions, "
                        "errors: list error messages, "
                        "session_detail: full messages for a session, "
                        "metrics: aggregated structured metrics, "
                        "skill_impact: compare metrics before/after a skill was created"
                    ),
                },
                "session_key": {"type": "string", "description": "Session key (for session_detail)"},
                "limit": {"type": "integer", "description": "Max items to return (default 20)"},
                "skill_name": {"type": "string", "description": "Skill name (for skill_impact)"},
                "trigger": {"type": "string", "description": "Trigger type (for log_decision): cron|error_rate|manual"},
                "decision_action": {"type": "string", "description": "Action taken (for log_decision): create_skill|update_skill|disable_skill|delete_skill|skip"},
                "reasoning": {"type": "string", "description": "Reasoning for the decision (for log_decision)"},
                "outcome": {"type": "string", "description": "Outcome (for log_decision): success|error|skipped"},
                "granularity": {"type": "string", "enum": ["day", "week", "month"], "description": "Time period granularity (for usage_stats, default: day)"},
                "format": {"type": "string", "enum": ["markdown", "json", "html"], "description": "Export format (for export, default: markdown)"},
                "query": {"type": "string", "description": "Search query (for search)"},
                "days": {"type": "integer", "description": "Archive sessions older than N days (for archive, default: 30)"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs["action"]
        limit = kwargs.get("limit", 20)

        if action == "summary":
            return self._summary(limit)
        if action == "errors":
            return self._errors(limit)
        if action == "session_detail":
            return self._session_detail(kwargs.get("session_key", ""), limit)
        if action == "metrics":
            return self._metrics(limit)
        if action == "skill_impact":
            return self._skill_impact(kwargs.get("skill_name", ""), limit)
        if action == "log_decision":
            return self._log_decision(kwargs)
        if action == "decision_history":
            return self._decision_history(limit)
        if action == "usage_stats":
            return self._usage_stats(kwargs.get("granularity", "day"), limit)
        if action == "export":
            return self._export(kwargs.get("session_key", ""), kwargs.get("format", "markdown"))
        if action == "search":
            return self._search(kwargs.get("query", ""), limit)
        if action == "archive":
            return self._archive(kwargs)
        return f"Unknown action: {action}"

    def _summary(self, limit: int) -> str:
        sessions = self._sessions.query_recent(limit=limit)
        if not sessions:
            return "No sessions found."
        lines = []
        total_msgs = 0
        total_errors = 0
        for s in sessions:
            total_msgs += s["messages"]
            total_errors += s["errors"]
            preview = s["last_user_msg"][:80] if s["last_user_msg"] else "(empty)"
            lines.append(
                f"  {s['key']}: {s['messages']} msgs, {s['tool_calls']} tools, "
                f"{s['errors']} errors | {preview}"
            )
        header = f"Sessions: {len(sessions)} | Total messages: {total_msgs} | Total errors: {total_errors}\n"
        return header + "\n".join(lines)

    def _errors(self, limit: int) -> str:
        sessions = self._sessions.query_recent(limit=50)
        errors: list[str] = []
        for s in sessions:
            if s["errors"] == 0:
                continue
            msgs = self._sessions.get_session_messages(s["key"], limit=100)
            for m in msgs:
                content = m.get("content") or ""
                if "error" in content.lower() and m.get("role") in ("tool", "assistant"):
                    errors.append(f"[{s['key']}] {content[:200]}")
                    if len(errors) >= limit:
                        break
            if len(errors) >= limit:
                break
        if not errors:
            return "No errors found in recent sessions."
        return f"Errors ({len(errors)}):\n" + "\n".join(errors)

    def _session_detail(self, key: str, limit: int) -> str:
        if not key:
            return "Error: 'session_key' is required for session_detail."
        msgs = self._sessions.get_session_messages(key, limit=limit)
        if not msgs:
            return f"Session '{key}' not found or empty."
        lines = []
        for m in msgs:
            role = m.get("role", "?")
            content = (m.get("content") or "")[:300]
            lines.append(f"[{role}] {content}")
        return f"Session {key} ({len(msgs)} messages):\n" + "\n".join(lines)

    def _metrics(self, limit: int) -> str:
        """Return aggregated metrics across recent sessions."""
        sessions = self._sessions.iter_recent_from_disk(limit=limit)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        per_session = [compute_session_metrics(s.messages) for s in sessions[:limit]]
        agg = aggregate_metrics(per_session)
        return json.dumps(agg.to_dict(), indent=2)

    def _skill_impact(self, skill_name: str, limit: int) -> str:
        """Compare metrics before/after a skill was created."""
        if not skill_name:
            return "Error: 'skill_name' is required for skill_impact."
        if not self._skills:
            return "Error: skills loader not available."

        spec = self._skills.get(skill_name)
        if not spec:
            return f"Skill '{skill_name}' not found."
        if not spec.created_at:
            return f"Skill '{skill_name}' has no created_at timestamp. Cannot measure impact."

        try:
            skill_created = datetime.fromisoformat(spec.created_at)
        except (ValueError, TypeError):
            return f"Invalid created_at for skill '{skill_name}': {spec.created_at}"

        all_sessions = self._sessions.iter_recent_from_disk(limit=limit * 2)
        all_sessions.sort(key=lambda s: s.updated_at, reverse=True)
        before: list = []
        after: list = []
        sessions = all_sessions[:limit * 2]

        for s in sessions:
            sm = compute_session_metrics(s.messages)
            if s.updated_at < skill_created:
                before.append(sm)
            else:
                after.append(sm)

        agg_before = aggregate_metrics(before)
        agg_after = aggregate_metrics(after)

        result = {
            "skill": skill_name,
            "created_at": spec.created_at,
            "before": {
                "sessions": agg_before.session_count,
                "error_rate": agg_before.overall_error_rate,
                "avg_turns": agg_before.avg_turns_per_session,
            },
            "after": {
                "sessions": agg_after.session_count,
                "error_rate": agg_after.overall_error_rate,
                "avg_turns": agg_after.avg_turns_per_session,
            },
        }
        return json.dumps(result, indent=2)

    def _log_decision(self, kwargs: dict[str, Any]) -> str:
        if not self._evolution_log:
            return "Error: evolution log not available."
        from nibot.evolution_log import EvolutionDecision
        decision = EvolutionDecision(
            trigger=kwargs.get("trigger", ""),
            action=kwargs.get("decision_action", ""),
            skill_name=kwargs.get("skill_name", ""),
            reasoning=kwargs.get("reasoning", ""),
            outcome=kwargs.get("outcome", ""),
        )
        self._evolution_log.append(decision)
        return f"Decision logged: {decision.action} ({decision.outcome})"

    def _decision_history(self, limit: int) -> str:
        if not self._evolution_log:
            return "Error: evolution log not available."
        return self._evolution_log.summary(limit)

    # ---- v0.10.0: session management enhancements ----

    def _usage_stats(self, granularity: str, limit: int) -> str:
        """Usage statistics grouped by day/week/month."""
        from nibot.metrics import compute_usage_stats

        sessions = self._sessions.iter_all_from_disk()
        if not sessions:
            return "No sessions found."
        pairs = [(s.updated_at, compute_session_metrics(s.messages)) for s in sessions]
        buckets = compute_usage_stats(pairs, granularity=granularity)
        buckets = buckets[-limit:]  # most recent periods
        return json.dumps([b.to_dict() for b in buckets], indent=2)

    def _export(self, session_key: str, fmt: str) -> str:
        """Export a session to readable format."""
        if not session_key:
            return "Error: 'session_key' is required for export."
        from nibot.session import format_session_export

        session = self._sessions._cache.get(session_key)
        if not session:
            session = self._sessions._load(session_key)
        if not session:
            return f"Session '{session_key}' not found."
        result = format_session_export(session, fmt=fmt)
        if len(result) > 50000:
            result = result[:50000] + "\n\n... (truncated, session too large)"
        return result

    def _search(self, query: str, limit: int) -> str:
        """Search across all sessions."""
        if not query:
            return "Error: 'query' is required for search."
        hits = self._sessions.search(query, max_results=limit)
        if not hits:
            return f"No results found for '{query}'."
        lines = [f"Search results for '{query}' ({len(hits)} hits):"]
        for h in hits:
            ts = f" ({h.timestamp})" if h.timestamp else ""
            lines.append(f"  [{h.session_key}] {h.role}{ts}: {h.content_preview[:100]}")
        return "\n".join(lines)

    def _archive(self, kwargs: dict[str, Any]) -> str:
        """Archive old sessions or a specific session."""
        session_key = kwargs.get("session_key", "")
        days = kwargs.get("days", 30)
        if session_key:
            ok = self._sessions.archive(session_key)
            if ok:
                return f"Session '{session_key}' archived."
            return f"Session '{session_key}' not found."
        archived = self._sessions.archive_old(days=days)
        if not archived:
            return f"No sessions older than {days} days to archive."
        return f"Archived {len(archived)} sessions: {', '.join(archived[:10])}"
