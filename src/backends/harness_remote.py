"""Harness remote backend — routes escalations across ALLOWED_MODELS.

Replaces the old hardcoded TieredRemoteBackend. Reads the allowed Fireworks
model list at construction, and per-call picks the cheapest-capable model
for the task type via ModelSelector. All calls go through FIREWORKS_BASE_URL
(enforced by FireworksBackend), so tokens are recorded by the judging proxy.
"""

from __future__ import annotations

from typing import Any

from .base import Backend, GenerationResult
from .fireworks import FireworksBackend
from .model_selector import ModelSelector


class HarnessRemoteBackend(Backend):
    is_remote = True

    def __init__(self, allowed_models: list[str], api_key: str | None = None) -> None:
        self.name = "fireworks:harness"
        self.selector = ModelSelector(allowed_models)
        self._api_key = api_key
        # Cache one FireworksBackend per model_id (client reuse).
        self._clients: dict[str, FireworksBackend] = {}

    def _client(self, model_id: str) -> FireworksBackend:
        if model_id not in self._clients:
            self._clients[model_id] = FireworksBackend(
                model=model_id, api_key=self._api_key
            )
        return self._clients[model_id]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        return_logprobs: bool = False,
        tier: str | None = None,          # accepted for interface compat, unused
        task_type: Any = None,            # TaskType, if the router passes it
        difficulty: float = 0.5,
        **kwargs: Any,
    ) -> GenerationResult:
        # Pick a model from the allowed list for this task.
        if task_type is not None:
            model_id = self.selector.select(task_type, difficulty)
        else:
            model_id = self.selector.cheapest

        result = self._client(model_id).generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            return_logprobs=return_logprobs,
        )
        result.raw["selected_model"] = model_id
        return result
