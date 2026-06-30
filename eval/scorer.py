"""Answer scoring.

Tracks 1 final scoring runs on standardized eval env — the official scorer is
unknown until Jul 6. This is a local proxy: exact match + numeric tolerance +
case-insensitive string compare. Replace/extend once official scorer is known.
"""

from __future__ import annotations

import re


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".,!?;:\"'()[]{}")
    return s


def _maybe_numeric(s: str) -> float | None:
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def score_answer(
    predicted: str,
    gold: str,
    task_type: str | None = None,
    numeric_atol: float = 1e-4,
    numeric_rtol: float = 1e-3,
) -> bool:
    """Return True iff predicted matches gold."""
    if predicted is None or gold is None:
        return False
    pred = str(predicted)
    g = str(gold)

    # Numeric path
    p_num = _maybe_numeric(pred)
    g_num = _maybe_numeric(g)
    if p_num is not None and g_num is not None:
        if g_num == 0:
            return abs(p_num) <= numeric_atol
        return abs(p_num - g_num) <= max(numeric_atol, numeric_rtol * abs(g_num))

    # Text path
    np_ = _normalize(pred)
    ng = _normalize(g)
    if np_ == ng:
        return True
    # Gold appears as substring of prediction (common for LLM verbosity)
    if ng and ng in np_:
        return True
    return False
