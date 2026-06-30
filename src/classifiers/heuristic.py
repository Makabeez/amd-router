"""Heuristic task classification.

Pre-routes a query before any model is touched. Free tokens, instant.
Returns a difficulty/category signal that escalation policies consume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    EXTRACTION = "extraction"  # pull a field from text
    CLASSIFICATION = "classification"  # label / multi-choice
    SHORT_QA = "short_qa"  # factual lookup, 1-2 sentence answer
    MATH = "math"  # arithmetic / word problem
    REASONING = "reasoning"  # multi-step logic
    CODE = "code"  # write or fix code
    LONG_GEN = "long_gen"  # multi-paragraph creative / summary
    UNKNOWN = "unknown"


@dataclass
class TaskFeatures:
    type: TaskType
    difficulty: float  # 0.0 easy → 1.0 hard, used by escalation policy
    input_chars: int
    has_code: bool
    has_numbers: bool
    requires_reasoning: bool

    @property
    def likely_needs_remote(self) -> bool:
        """Soft signal — final routing call belongs to the escalation policy."""
        return self.difficulty >= 0.7


# Cheap regex / keyword bank. Replace / extend on Jul 6 once real tasks are seen.
_MATH_PATTERNS = [
    r"\b\d+\s*[\+\-\*/×÷]\s*\d+",
    r"\b(sum|product|average|mean|median|percent|percentage|ratio)\b",
    r"\b(solve|calculate|compute|find\s+the\s+value)\b",
]
_CODE_PATTERNS = [
    r"```",
    r"\bdef\s+\w+\(|\bclass\s+\w+|\bfunction\s+\w+",
    r"\b(import\s+\w+|from\s+\w+\s+import)\b",
    r"\b(SELECT|INSERT|UPDATE|DELETE)\b.*\b(FROM|INTO|WHERE)\b",
]
_REASONING_PATTERNS = [
    r"\b(why|how|explain|because|therefore|reason)\b",
    r"\b(step.by.step|reasoning|think\s+through)\b",
    r"\b(if.+then|implies|deduce|infer)\b",
]
_EXTRACTION_PATTERNS = [
    r"\bextract\b",
    r"\bwhat\s+is\s+the\s+\w+\s+(in|of|from)\b",
    r"\bpull\s+out\b",
    r"\bfind\s+the\s+(name|date|number|email|address)\b",
]
_CLASSIFICATION_PATTERNS = [
    r"\b(classify|categorize|label)\b",
    r"\bwhich\s+(category|class|label|type)\b",
    r"\bsentiment\b",
    r"\b(true|false|yes|no)\?",
]


def _has(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify(prompt: str) -> TaskFeatures:
    """Pure-Python heuristic. Zero tokens, sub-millisecond."""
    text = prompt
    chars = len(text)
    has_code = _has(_CODE_PATTERNS, text)
    has_numbers = bool(re.search(r"\d", text))

    if has_code:
        ttype = TaskType.CODE
    elif _has(_MATH_PATTERNS, text):
        ttype = TaskType.MATH
    elif _has(_EXTRACTION_PATTERNS, text):
        ttype = TaskType.EXTRACTION
    elif _has(_CLASSIFICATION_PATTERNS, text):
        ttype = TaskType.CLASSIFICATION
    elif _has(_REASONING_PATTERNS, text):
        ttype = TaskType.REASONING
    elif chars > 2000:
        ttype = TaskType.LONG_GEN
    else:
        ttype = TaskType.SHORT_QA

    # Base difficulty by task type (calibrate after first eval pass)
    base = {
        TaskType.EXTRACTION: 0.25,
        TaskType.CLASSIFICATION: 0.30,
        TaskType.SHORT_QA: 0.40,
        TaskType.MATH: 0.65,
        TaskType.REASONING: 0.75,
        TaskType.CODE: 0.70,
        TaskType.LONG_GEN: 0.60,
        TaskType.UNKNOWN: 0.50,
    }[ttype]

    # Length bump — longer inputs are harder on small local models
    length_bump = min(0.25, chars / 8000)
    difficulty = min(1.0, base + length_bump)

    return TaskFeatures(
        type=ttype,
        difficulty=difficulty,
        input_chars=chars,
        has_code=has_code,
        has_numbers=has_numbers,
        requires_reasoning=ttype in {TaskType.MATH, TaskType.REASONING, TaskType.CODE},
    )
