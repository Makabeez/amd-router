"""Hybrid local/remote router. The main thing.

Flow:
  1. classify (free)
  2. preflight: should we even try local?  -> if no, go straight to remote
  3. local generate with logprobs
  4. (optional) self-consistency: n samples at temp>0
  5. confidence assessment
  6. postlocal: escalate?  -> if yes, remote call (optionally with local as draft)
  7. return final
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..backends.base import Backend, GenerationResult
from ..classifiers.confidence import ConfidenceReport, assess
from ..classifiers.heuristic import TaskFeatures, classify
from ..escalation.policies import EscalationPolicy, ThresholdPolicy
from .base import Router, RoutingTrace

AnswerExtractor = Callable[[GenerationResult], str]


@dataclass
class HybridConfig:
    # Local generation
    local_max_tokens: int = 512
    local_temperature: float = 0.0
    local_return_logprobs: bool = True

    # Self-consistency (set n_samples=0 to disable)
    n_samples: int = 0
    samples_temperature: float = 0.7
    samples_max_tokens: int = 256  # samples can be shorter than primary

    # Remote generation
    remote_max_tokens: int = 512
    remote_temperature: float = 0.0

    # Verification mode: when escalating with local-as-context,
    # ask remote to verify+correct rather than redo from scratch
    verify_prompt_template: str = (
        "Question: {prompt}\n\n"
        "Draft answer: {draft}\n\n"
        "If the draft is correct, repeat it exactly. "
        "If incorrect, provide the correct answer only. No explanation."
    )


class HybridRouter(Router):
    def __init__(
        self,
        local: Backend,
        remote: Backend,
        policy: EscalationPolicy | None = None,
        config: HybridConfig | None = None,
        answer_extractor: AnswerExtractor | None = None,
        classifier: Callable[[str], TaskFeatures] = classify,
    ) -> None:
        assert not local.is_remote, "local backend must have is_remote=False"
        assert remote.is_remote, "remote backend must have is_remote=True"
        self.local = local
        self.remote = remote
        self.policy = policy or ThresholdPolicy()
        self.config = config or HybridConfig()
        self.answer_extractor = answer_extractor or (lambda r: r.text)
        self.classifier = classifier

    def route(self, prompt: str, **kwargs: Any) -> RoutingTrace:
        cfg = self.config
        decisions: list[str] = []

        # 1. Classify
        features = self.classifier(prompt)
        decisions.append(
            f"classified: type={features.type} difficulty={features.difficulty:.2f}"
        )

        # 2. Preflight
        pre = self.policy.decide_preflight(features)
        decisions.append(f"preflight: escalate={pre.escalate} ({pre.reason})")

        if pre.escalate:
            remote_result = self.remote.generate(
                prompt,
                max_tokens=cfg.remote_max_tokens,
                temperature=cfg.remote_temperature,
            )
            return RoutingTrace(
                prompt=prompt,
                final_text=remote_result.text,
                final_backend=self.remote.name,
                remote_tokens=remote_result.remote_tokens,
                local_tokens=0,
                decisions=decisions,
                metadata={"features": features, "skipped_local": True},
            )

        # 3. Local primary generation
        local_primary = self.local.generate(
            prompt,
            max_tokens=cfg.local_max_tokens,
            temperature=cfg.local_temperature,
            return_logprobs=cfg.local_return_logprobs,
        )
        decisions.append(
            f"local: {local_primary.local_output_tokens} tokens, "
            f"mean_logprob={local_primary.mean_logprob}"
        )

        # 4. Optional self-consistency
        samples: list[GenerationResult] | None = None
        if cfg.n_samples > 0:
            samples = self.local.generate_n(
                prompt,
                n=cfg.n_samples,
                max_tokens=cfg.samples_max_tokens,
                temperature=cfg.samples_temperature,
            )
            decisions.append(f"self-consistency: n={len(samples)}")

        # 5. Confidence assessment
        confidence = assess(local_primary, samples, self.answer_extractor)
        decisions.append(
            f"confidence: score={confidence.score:.2f} "
            f"agreement={confidence.agreement} "
            f"top_answer={confidence.top_answer!r}"
        )

        # If self-consistency gave a clear winner, prefer the voted answer
        local_text = local_primary.text
        if confidence.top_answer and confidence.agreement and confidence.agreement >= 0.6:
            local_text = confidence.top_answer

        # 6. Postlocal escalation decision
        post = self.policy.decide_postlocal(features, confidence)
        decisions.append(f"postlocal: escalate={post.escalate} ({post.reason})")

        if not post.escalate:
            local_tokens = local_primary.local_output_tokens + sum(
                s.local_output_tokens for s in (samples or [])
            )
            return RoutingTrace(
                prompt=prompt,
                final_text=local_text,
                final_backend=self.local.name,
                remote_tokens=0,
                local_tokens=local_tokens,
                decisions=decisions,
                metadata={
                    "features": features,
                    "confidence": confidence,
                },
            )

        # 7. Escalate to remote
        if post.use_local_as_context:
            remote_prompt = cfg.verify_prompt_template.format(
                prompt=prompt, draft=local_text
            )
            decisions.append("remote: verify-mode (draft attached)")
        else:
            remote_prompt = prompt
            decisions.append("remote: fresh prompt")

        remote_result = self.remote.generate(
            remote_prompt,
            max_tokens=cfg.remote_max_tokens,
            temperature=cfg.remote_temperature,
        )
        decisions.append(
            f"remote: in={remote_result.remote_input_tokens} "
            f"out={remote_result.remote_output_tokens}"
        )

        return RoutingTrace(
            prompt=prompt,
            final_text=remote_result.text,
            final_backend=self.remote.name,
            remote_tokens=remote_result.remote_tokens,
            local_tokens=local_primary.local_output_tokens
            + sum(s.local_output_tokens for s in (samples or [])),
            decisions=decisions,
            metadata={
                "features": features,
                "confidence": confidence,
                "verify_mode": post.use_local_as_context,
            },
        )
