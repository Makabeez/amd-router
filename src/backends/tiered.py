"""Tiered remote backend — model ladder for cost-optimized escalation.

Wraps multiple Fireworks models behind one Backend interface. The router
calls `generate(prompt, tier=...)` and the wrapper dispatches to the right
underlying model.

Tiers (calibrated against Jul-2026 Fireworks catalog — reasoning-model era):
  small  → cheapest per-answer overhead. Use for verification of local drafts,
           simple classification escalations, extraction confirmations.
  medium → balanced. Use for math, reasoning where "small" might miss.
  large  → highest capacity. Use only when medium also failed.

Empirical basis: DeepSeek V4-Pro burns ~36 output tokens on a 1-word answer,
Kimi K2p6 burns ~54, GLM 5p2 burns ~87. Numbers will differ per task shape
but this ordering held on smoke tests. Recalibrate via calibrate_thresholds.py
once real tasks are seen on Jul 6.

In trading terms: small = tight-spread liquid market, large = expensive
limit order with slippage. Match instrument to trade.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from .base import Backend, GenerationResult
from .fireworks import FireworksBackend

Tier = Literal["small", "medium", "large"]


class TieredRemoteBackend(Backend):
    """Multi-model remote backend with tier-based dispatch.

    Defaults reflect the reasoning-model catalog available Jul 2026.
    Override via env vars for hackathon-day model lineup:
      FIREWORKS_MODEL_SMALL, FIREWORKS_MODEL_MEDIUM, FIREWORKS_MODEL_LARGE
    """

    is_remote = True

    DEFAULTS: dict[Tier, str] = {
        "small": "accounts/fireworks/models/deepseek-v4-pro",   # lowest overhead observed
        "medium": "accounts/fireworks/models/kimi-k2p6",         # balanced
        "large": "accounts/fireworks/models/glm-5p2",            # highest capacity, most overhead
    }

    def __init__(
        self,
        models: dict[Tier, str] | None = None,
        api_key: str | None = None,
        default_tier: Tier = "medium",
    ) -> None:
        self.name = "fireworks:tiered"
        self.default_tier = default_tier

        m = dict(self.DEFAULTS)
        if models:
            m.update(models)
        # Env overrides
        for tier in ("small", "medium", "large"):
            env_key = f"FIREWORKS_MODEL_{tier.upper()}"
            if env_key in os.environ:
                m[tier] = os.environ[env_key]  # type: ignore[index]

        self._backends: dict[Tier, FireworksBackend] = {
            tier: FireworksBackend(model=model_id, api_key=api_key)
            for tier, model_id in m.items()
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        return_logprobs: bool = False,
        tier: Tier | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        chosen_tier: Tier = tier or self.default_tier
        backend = self._backends[chosen_tier]
        result = backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            return_logprobs=return_logprobs,
            **kwargs,
        )
        # Annotate which tier was used (visible in metrics)
        result.raw["tier"] = chosen_tier
        return result
