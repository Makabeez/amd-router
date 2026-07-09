"""Escalation policies.

The router asks a policy: "given this task and this local attempt, escalate?"
Policies are pluggable so we can A/B them on Day 1 once tasks are known.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..classifiers.confidence import ConfidenceReport
from ..classifiers.heuristic import TaskFeatures, TaskType


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str
    use_local_as_context: bool = False  # pass local attempt to remote (cheaper than fresh)


class EscalationPolicy(ABC):
    @abstractmethod
    def decide_preflight(self, features: TaskFeatures) -> EscalationDecision:
        """Called BEFORE any local generation. Skip local entirely if very hard."""

    @abstractmethod
    def decide_postlocal(
        self,
        features: TaskFeatures,
        confidence: ConfidenceReport,
    ) -> EscalationDecision:
        """Called AFTER local attempt. Decide whether to escalate to remote."""


@dataclass
class ThresholdPolicy(EscalationPolicy):
    """Pure-threshold policy. Calibrate thresholds per task type after a dry run.

    Defaults are conservative — tune via scripts/calibrate_thresholds.py once we
    see the real task distribution on Jul 6.
    """

    # Preflight: tasks with difficulty >= this skip local entirely
    preflight_skip_local_above: float = 0.85

    # Postlocal: confidence below this triggers escalation
    min_confidence: float = 0.65

    # Per-task overrides — math/code routinely fail on small local models
    per_task_min_confidence: dict[TaskType, float] = field(
        default_factory=lambda: {
            TaskType.MATH: 0.80,
            TaskType.CODE: 0.80,
            TaskType.REASONING: 0.75,
            TaskType.EXTRACTION: 0.55,
            TaskType.CLASSIFICATION: 0.55,
            TaskType.SHORT_QA: 0.60,
            TaskType.LONG_GEN: 0.65,
        }
    )

    # If escalating, should we feed the local attempt to remote as draft?
    use_local_as_context_default: bool = True

    # Task types where the small local model is too unreliable to trust even at
    # high self-reported confidence (it can be confidently wrong — e.g. Qwen 0.5B
    # hallucinating "task unrelated" on code debugging at logprob-confidence 0.93).
    # These skip local and go straight to remote. Code is the main offender;
    # these are also the highest-value categories where a wrong answer fails the
    # accuracy gate entirely.
    # AMD_ALWAYS_ESCALATE_ENV: set ALWAYS_ESCALATE="" to disable entirely,
    # or a comma list of TaskType values (e.g. "code"). Measured: Qwen-0.5B
    # passes 3/3 local code tasks, so preflight-escalating CODE burns ~1.2k
    # remote tokens for answers we already had.
    always_escalate: set[TaskType] = field(
        default_factory=lambda: {TaskType.CODE}
    )

    def decide_preflight(self, features: TaskFeatures) -> EscalationDecision:
        if features.type in self.always_escalate:
            return EscalationDecision(
                escalate=True,
                reason=f"preflight: {features.type.value} always escalates (local unreliable)",
                use_local_as_context=False,
            )
        if features.difficulty >= self.preflight_skip_local_above:
            return EscalationDecision(
                escalate=True,
                reason=f"preflight: difficulty {features.difficulty:.2f} >= {self.preflight_skip_local_above}",
                use_local_as_context=False,
            )
        return EscalationDecision(escalate=False, reason="preflight: local first")

    def decide_postlocal(
        self,
        features: TaskFeatures,
        confidence: ConfidenceReport,
    ) -> EscalationDecision:
        threshold = self.per_task_min_confidence.get(features.type, self.min_confidence)
        if confidence.score < threshold:
            return EscalationDecision(
                escalate=True,
                reason=(
                    f"postlocal: confidence {confidence.score:.2f} < {threshold:.2f} "
                    f"(task={features.type}, agreement={confidence.agreement}, "
                    f"mean_logprob={confidence.mean_logprob})"
                ),
                use_local_as_context=self.use_local_as_context_default,
            )
        return EscalationDecision(
            escalate=False,
            reason=f"postlocal: confident ({confidence.score:.2f} >= {threshold:.2f})",
        )


@dataclass
class AlwaysLocalPolicy(EscalationPolicy):
    """Baseline: never escalate. For measuring local-only floor accuracy."""

    def decide_preflight(self, features: TaskFeatures) -> EscalationDecision:
        return EscalationDecision(escalate=False, reason="always_local")

    def decide_postlocal(self, features, confidence) -> EscalationDecision:
        return EscalationDecision(escalate=False, reason="always_local")


@dataclass
class AlwaysRemotePolicy(EscalationPolicy):
    """Baseline: always escalate. For measuring remote-only ceiling accuracy and cost."""

    def decide_preflight(self, features: TaskFeatures) -> EscalationDecision:
        return EscalationDecision(escalate=True, reason="always_remote")

    def decide_postlocal(self, features, confidence) -> EscalationDecision:
        return EscalationDecision(escalate=True, reason="always_remote")
