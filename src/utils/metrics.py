"""Run-level metrics aggregation. Leaderboard-shaped output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..router.base import RoutingTrace


@dataclass
class RunMetrics:
    n: int = 0
    n_correct: int = 0
    total_remote_tokens: int = 0
    total_local_tokens: int = 0
    n_local_only: int = 0  # routed to local, no remote
    n_remote_used: int = 0
    n_verify_mode: int = 0
    per_task_type: dict[str, dict[str, int]] = field(default_factory=dict)
    traces: list[RoutingTrace] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def avg_remote_tokens(self) -> float:
        return self.total_remote_tokens / self.n if self.n else 0.0

    @property
    def local_rate(self) -> float:
        return self.n_local_only / self.n if self.n else 0.0

    def add(self, trace: RoutingTrace, correct: bool) -> None:
        self.n += 1
        self.n_correct += int(correct)
        self.total_remote_tokens += trace.remote_tokens
        self.total_local_tokens += trace.local_tokens
        if trace.remote_tokens == 0:
            self.n_local_only += 1
        else:
            self.n_remote_used += 1
        if trace.metadata.get("verify_mode"):
            self.n_verify_mode += 1

        features = trace.metadata.get("features")
        if features is not None:
            t = str(features.type)
            bucket = self.per_task_type.setdefault(
                t, {"n": 0, "correct": 0, "remote_tokens": 0}
            )
            bucket["n"] += 1
            bucket["correct"] += int(correct)
            bucket["remote_tokens"] += trace.remote_tokens

        self.traces.append(trace)

    def summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "total_remote_tokens": self.total_remote_tokens,
            "avg_remote_tokens": round(self.avg_remote_tokens, 1),
            "local_rate": round(self.local_rate, 3),
            "n_remote_used": self.n_remote_used,
            "n_verify_mode": self.n_verify_mode,
            "per_task_type": self.per_task_type,
        }
