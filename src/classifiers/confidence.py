"""Confidence signals on local generation results.

LOCAL TOKENS ARE FREE — burn them generously here to avoid remote calls.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from ..backends.base import GenerationResult


@dataclass
class ConfidenceReport:
    """Aggregated confidence signal across all available checks."""

    score: float  # 0.0 → 1.0
    min_logprob: float | None
    mean_logprob: float | None
    agreement: float | None  # self-consistency agreement, 0.0-1.0
    top_answer: str | None  # most frequent canonical answer if voted
    n_samples: int

    @property
    def is_confident(self) -> bool:
        return self.score >= 0.7


def _canonicalize(text: str) -> str:
    """Normalize an answer for vote comparison.

    Strips whitespace, lowercases, collapses inner spaces, removes trailing
    punctuation. For numeric answers, parses to canonical numeric form.
    """
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".!?,;:")

    # Try numeric canonicalization
    num_match = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if num_match and num_match.group() == s.replace(",", "").replace(" ", ""):
        try:
            f = float(num_match.group())
            return str(int(f)) if f.is_integer() else f"{f:.6g}"
        except ValueError:
            pass

    return s


def logprob_confidence(result: GenerationResult) -> float | None:
    """Map mean logprob to a confidence score in [0,1].

    -0.1 mean logprob → ~0.9 confidence
    -1.0 mean logprob → ~0.37 confidence
    -3.0 mean logprob → ~0.05 confidence
    """
    if result.mean_logprob is None:
        return None
    return math.exp(result.mean_logprob)


def self_consistency(
    results: list[GenerationResult],
    answer_extractor=None,
) -> tuple[float, str]:
    """Vote across n samples. Returns (agreement_ratio, top_answer)."""
    if not results:
        return 0.0, ""
    if answer_extractor is None:
        answer_extractor = lambda r: r.text  # noqa: E731

    canon = [_canonicalize(answer_extractor(r)) for r in results]
    counts = Counter(canon)
    top, top_count = counts.most_common(1)[0]
    return top_count / len(results), top


def assess(
    primary: GenerationResult,
    samples: list[GenerationResult] | None = None,
    answer_extractor=None,
) -> ConfidenceReport:
    """Combine logprob + self-consistency into a single confidence report.

    Pass `samples` (independent local generations of the same prompt at
    temperature > 0) to enable self-consistency. Without samples, falls
    back to logprob-only confidence.
    """
    lp_conf = logprob_confidence(primary)
    agreement: float | None = None
    top_answer: str | None = None
    n = 1

    if samples and len(samples) >= 2:
        agreement, top_answer = self_consistency(samples, answer_extractor)
        n = len(samples)

    # Combine: agreement dominates if available, else logprob
    if agreement is not None and lp_conf is not None:
        score = 0.7 * agreement + 0.3 * lp_conf
    elif agreement is not None:
        score = agreement
    elif lp_conf is not None:
        score = lp_conf
    else:
        score = 0.5  # uninformative prior

    return ConfidenceReport(
        score=score,
        min_logprob=primary.min_logprob,
        mean_logprob=primary.mean_logprob,
        agreement=agreement,
        top_answer=top_answer,
        n_samples=n,
    )
