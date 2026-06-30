"""Backend interface. Every model wrapper implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """Result of a single generation call.

    `remote_tokens` is the only field that counts toward score.
    Local tokens are free per Track 1 rules.
    """

    text: str
    remote_input_tokens: int = 0
    remote_output_tokens: int = 0
    local_input_tokens: int = 0
    local_output_tokens: int = 0
    logprobs: list[float] | None = None  # per-token logprobs if available
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def remote_tokens(self) -> int:
        return self.remote_input_tokens + self.remote_output_tokens

    @property
    def total_tokens(self) -> int:
        return (
            self.remote_input_tokens
            + self.remote_output_tokens
            + self.local_input_tokens
            + self.local_output_tokens
        )

    @property
    def min_logprob(self) -> float | None:
        return min(self.logprobs) if self.logprobs else None

    @property
    def mean_logprob(self) -> float | None:
        if not self.logprobs:
            return None
        return sum(self.logprobs) / len(self.logprobs)


class Backend(ABC):
    """A generation backend. Either local (free tokens) or remote (counts)."""

    name: str
    is_remote: bool

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        return_logprobs: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a completion."""

    def generate_n(
        self,
        prompt: str,
        n: int,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> list[GenerationResult]:
        """Generate n independent samples (for self-consistency).

        Default impl is sequential. Override for batched local backends.
        """
        return [
            self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
            for _ in range(n)
        ]
