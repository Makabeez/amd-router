"""Fireworks AI backend — harness-aware.

AMD Track 1 contract (Participant Guide):
  - FIREWORKS_API_KEY: injected by harness at eval time (use it, not your own)
  - FIREWORKS_BASE_URL: ALL calls must route through this or they score zero
  - ALLOWED_MODELS: comma-separated model IDs, published launch day

For local dev, set these in .env to your own key + the public base URL.
At eval time the harness overrides them. NEVER hardcode model IDs — read
from ALLOWED_MODELS.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import Backend, GenerationResult


def get_allowed_models() -> list[str]:
    """Parse ALLOWED_MODELS env var into a list. Empty list if unset."""
    raw = os.environ.get("ALLOWED_MODELS", "").strip()
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


class FireworksBackend(Backend):
    """OpenAI-compatible client for Fireworks AI, routed through the harness URL.

    Every token here counts toward the leaderboard score, so we log call
    counts and surface exact input/output token usage on each result.
    """

    is_remote = True

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = f"fireworks:{model}"
        self.model = model
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY not set (harness injects this at eval)")

        # MUST route through FIREWORKS_BASE_URL. Fall back to public URL for dev.
        self.base_url = (
            base_url
            or os.environ.get("FIREWORKS_BASE_URL")
            or "https://api.fireworks.ai/inference/v1"
        ).rstrip("/")

        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        return_logprobs: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        if return_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

        r = self.client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()

        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        usage = data.get("usage", {})

        logprobs: list[float] | None = None
        if return_logprobs and choice.get("logprobs"):
            content = choice["logprobs"].get("content", [])
            logprobs = [tok["logprob"] for tok in content if "logprob" in tok]

        return GenerationResult(
            text=text,
            remote_input_tokens=usage.get("prompt_tokens", 0),
            remote_output_tokens=usage.get("completion_tokens", 0),
            logprobs=logprobs,
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    def __del__(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


class MockBackend(Backend):
    """Deterministic mock backend for unit tests and harness dry-runs."""

    is_remote = False

    def __init__(self, name: str = "mock", canned: str = "MOCK_RESPONSE") -> None:
        self.name = name
        self.canned = canned

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        return_logprobs: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        return GenerationResult(
            text=self.canned,
            local_input_tokens=len(prompt.split()),
            local_output_tokens=len(self.canned.split()),
            logprobs=[-0.1] * len(self.canned.split()) if return_logprobs else None,
        )
