"""Model selection across ALLOWED_MODELS.

The harness publishes the allowed model IDs at runtime via ALLOWED_MODELS.
We can't hardcode them, but we CAN classify them by name pattern into
capability tiers, then pick the cheapest-capable model per task type.

Launch-day allowed list (Track 1), for reference — but always read the env:
  minimax-m3
  kimi-k2p7-code
  gemma-4-31b-it
  gemma-4-26b-a4b-it
  gemma-4-31b-it-nvfp4

Strategy: Gemma models are 3 of 5 and the general-purpose workhorses (also
unlock the +$1k Gemma bonus). kimi-k2p7-code is code-specialized. minimax-m3
is a capable generalist. We prefer the smallest Gemma for easy tasks, a
code model for code tasks, and a larger model only when needed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from ..classifiers.heuristic import TaskType


@dataclass
class ModelInfo:
    model_id: str
    family: str          # gemma | kimi | minimax | unknown
    is_code: bool        # code-specialized
    size_hint: float     # rough param-size proxy for ordering (lower = cheaper)


def _classify_model(model_id: str) -> ModelInfo:
    """Infer family/specialization/size from the model ID string."""
    low = model_id.lower()

    family = "unknown"
    if "gemma" in low:
        family = "gemma"
    elif "kimi" in low:
        family = "kimi"
    elif "minimax" in low:
        family = "minimax"
    elif "deepseek" in low:
        family = "deepseek"
    elif "llama" in low:
        family = "llama"

    is_code = "code" in low

    # Size hint: extract a number followed by 'b' (billions), else heuristic.
    size_hint = 30.0  # default mid
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", low)
    if m:
        size_hint = float(m.group(1))
    # a4b / MoE-active-param hints: "26b-a4b" means 26B total but 4B active — cheaper
    a = re.search(r"a(\d+(?:\.\d+)?)b", low)
    if a:
        size_hint = float(a.group(1))  # active params dominate cost
    # nvfp4 quantization → cheaper to run, nudge down
    if "nvfp4" in low or "fp4" in low or "fp8" in low:
        size_hint *= 0.9

    return ModelInfo(model_id=model_id, family=family, is_code=is_code, size_hint=size_hint)


class ModelSelector:
    """Chooses a model ID from the allowed list for a given task type."""

    def __init__(self, allowed_models: list[str]) -> None:
        if not allowed_models:
            raise ValueError("no allowed models provided")
        self.models = [_classify_model(m) for m in allowed_models]
        # Precompute orderings
        self._by_size = sorted(self.models, key=lambda m: m.size_hint)
        self._code_models = [m for m in self.models if m.is_code]
        self._gemma_models = sorted(
            [m for m in self.models if m.family == "gemma"], key=lambda m: m.size_hint
        )

    @property
    def cheapest(self) -> str:
        return self._by_size[0].model_id

    @property
    def largest(self) -> str:
        return self._by_size[-1].model_id

    def cheapest_gemma(self) -> str | None:
        return self._gemma_models[0].model_id if self._gemma_models else None

    def select(self, task_type: TaskType, difficulty: float = 0.5) -> str:
        """Pick a model ID for this task.

        - Code tasks → prefer a code-specialized model if available.
        - Everything else → prefer Gemma (unlocks bonus, strong generalist);
          cheapest Gemma for easy, a larger model for hard.
        - Fall back to cheapest overall if no Gemma present.
        """
        # Code tasks: use code model if we have one
        if task_type in (TaskType.CODE,) and self._code_models:
            return self._code_models[0].model_id

        # Gemma is on-demand on Fireworks: undeployed -> 404. Chasing the bonus
        # by default risks the 80% accuracy gate. Opt in with PREFER_GEMMA=1.
        if self._gemma_models and os.environ.get("PREFER_GEMMA") == "1":
            if difficulty >= 0.7 and len(self._gemma_models) > 1:
                # harder task → larger Gemma
                return self._gemma_models[-1].model_id
            return self._gemma_models[0].model_id

        # No Gemma → size-based pick
        if difficulty >= 0.7:
            return self.largest
        return self.cheapest
