"""Evolution decision log -- append-only JSONL journal for evolution agent decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class EvolutionDecision:
    """A single evolution decision record."""

    timestamp: str = ""
    trigger: str = ""       # "cron" | "error_rate" | "manual"
    action: str = ""        # "create_skill" | "update_skill" | "disable_skill" | "delete_skill" | "skip"
    skill_name: str = ""
    reasoning: str = ""
    metrics_snapshot: dict = field(default_factory=dict)
    outcome: str = ""       # "success" | "error" | "skipped"


class EvolutionLog:
    """Append-only JSONL log of evolution decisions.

    File: ``workspace/evolution/decisions.jsonl``
    Same pattern as SessionManager -- one JSON object per line, no locking needed
    for single-writer append.
    """

    def __init__(self, workspace: Path) -> None:
        self._dir = workspace / "evolution"
        self._path = self._dir / "decisions.jsonl"

    def append(self, decision: EvolutionDecision) -> None:
        if not decision.timestamp:
            decision.timestamp = datetime.now().isoformat()
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")

    def read_recent(self, limit: int = 20) -> list[EvolutionDecision]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        result: list[EvolutionDecision] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                result.append(EvolutionDecision(**{k: v for k, v in data.items()
                                                   if k in EvolutionDecision.__dataclass_fields__}))
            except (json.JSONDecodeError, TypeError):
                continue
            if len(result) >= limit:
                break
        return result

    def summary(self, limit: int = 10) -> str:
        decisions = self.read_recent(limit)
        if not decisions:
            return "(no evolution decisions yet)"
        lines: list[str] = []
        for d in decisions:
            lines.append(f"- [{d.timestamp[:16]}] {d.trigger}/{d.action}: "
                         f"{d.skill_name or '-'} -> {d.outcome} | {d.reasoning[:80]}")
        return "\n".join(lines)
