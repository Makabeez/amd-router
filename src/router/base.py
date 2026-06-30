"""Router base — orchestrates classifier → local → confidence → escalation → remote."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import GenerationResult


@dataclass
class RoutingTrace:
    """Full audit trail for one query. Used by metrics + leaderboard scoring."""

    prompt: str
    final_text: str
    final_backend: str
    remote_tokens: int  # the score-relevant number
    local_tokens: int
    decisions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Router(ABC):
    @abstractmethod
    def route(self, prompt: str, **kwargs: Any) -> RoutingTrace:
        """Process one query end-to-end."""

    def batch(self, prompts: list[str], **kwargs: Any) -> list[RoutingTrace]:
        return [self.route(p, **kwargs) for p in prompts]
