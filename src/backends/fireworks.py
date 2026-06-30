"""Fireworks AI API backend. Every token here counts toward score."""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import Backend, GenerationResult


class FireworksBackend(Backend):
    """OpenAI-compatible client for Fireworks AI.

    Remote tokens are the only ones that count toward score, so we
    log every call and surface input/output token counts on the result.
    """

    is_remote = True

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        timeout: float = 60.0,
    ) -> None:
        self.name = f"fireworks:{model}"
        self.model = model
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY not set")
        self.base_url = base_url.rstrip("/")
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
        text = choice["message"]["content"]
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
