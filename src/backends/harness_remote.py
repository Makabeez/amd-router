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
        self._dead: set[str] = set()   # models that 404/400 - never retry

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
        # AMD_MODEL_FALLBACK: ordered candidates. A model may be undeployed
        # (Fireworks on-demand -> 404). Never let that collapse to local.
        if task_type is not None:
            first = self.selector.select(task_type, difficulty)
        else:
            first = self.selector.cheapest

        candidates = [first] + [
            m.model_id for m in self.selector._by_size if m.model_id != first
        ]

        last_err: Exception | None = None
        for model_id in candidates:
            if model_id in self._dead:
                continue
            try:
                result = self._client(model_id).generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    return_logprobs=return_logprobs,
                )
            except Exception as e:
                last_err = e
                code = getattr(getattr(e, "response", None), "status_code", None)
                # 404/400 = model not deployed or unknown -> permanently skip it.
                if code in (400, 404):
                    self._dead.add(model_id)
                    print(f"[router] model {model_id} unavailable ({code}); "
                          f"falling back", flush=True)
                    continue
                # 401/403 = auth problem, no other model will help.
                if code in (401, 403):
                    raise
                # transient (429/5xx/timeout): try the next model rather than die.
                print(f"[router] model {model_id} failed ({type(e).__name__}); "
                      f"trying next", flush=True)
                continue

            result.raw["selected_model"] = model_id
            return result

        raise RuntimeError(
            f"all remote models exhausted ({len(candidates)} tried); "
            f"last error: {type(last_err).__name__}: {last_err}"
        )
