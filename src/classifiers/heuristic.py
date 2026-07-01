"""Heuristic task classification.

Pre-routes a query before any model is touched. Free tokens, instant.
Returns a difficulty/category signal that escalation policies consume.

Design principle: fire on GENERALIZABLE features (structural cues like
"has inline source text", "has comparative ordering") not surface phrasing.
Overfitting to a specific eval set is worse than a moderately accurate
classifier — misclassifications get caught by the confidence layer downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    EXTRACTION = "extraction"  # pull a field from provided text
    CLASSIFICATION = "classification"  # label / multi-choice
    SHORT_QA = "short_qa"  # factual world-knowledge lookup
    MATH = "math"  # arithmetic / word problem
    REASONING = "reasoning"  # multi-step logic / ordering
    CODE = "code"  # write or evaluate code
    LONG_GEN = "long_gen"  # multi-paragraph creative / summary
    UNKNOWN = "unknown"


@dataclass
class TaskFeatures:
    type: TaskType
    difficulty: float  # 0.0 easy → 1.0 hard, used by escalation policy
    input_chars: int
    has_code: bool
    has_numbers: bool
    has_inline_source: bool  # true if prompt contains a text passage to work with
    requires_reasoning: bool

    @property
    def likely_needs_remote(self) -> bool:
        return self.difficulty >= 0.7


# --- Pattern banks ---

_CODE_PATTERNS = [
    r"```",
    r"\bdef\s+\w+\(|\bclass\s+\w+|\bfunction\s+\w+",
    r"\b(import\s+\w+|from\s+\w+\s+import)\b",
    r"\b(SELECT|INSERT|UPDATE|DELETE)\b.*\b(FROM|INTO|WHERE)\b",
    r"\b(what does .{0,40}return|what is the output of|what does this)\b",
    r"\bprint\s*\(|\blen\s*\(|\brange\s*\(|\bsum\s*\(",
    r"['\"][^'\"]*['\"]\.\w+\s*\(",  # 'str'.method(
    r"\[::-?\d*\]",  # slice notation
    r"\b(python|javascript|typescript|rust|golang)\s+(one-?liner|snippet|expression)\b",
    r"\bwrite\s+(a|some)\s+(python|js|javascript|rust|code)\b",
]

_MATH_ARITHMETIC_PATTERNS = [
    r"\b\d+\s*[\+\-\*/×÷]\s*\d+",
    r"\b(divided\s+by|times|multiplied\s+by|plus|minus|modulo)\b",
    r"\b(square\s+root|cube\s+root|factorial|power\s+of|to\s+the\s+power)\b",
    r"\b(sum|product|average|mean|median|percent(age)?|ratio)\s+of\b",
    r"\b(solve|calculate|compute|evaluate)\b",
    r"\b(area|volume|perimeter|circumference|diameter|radius)\s+of\b",
    r"\bx\s*[\+\-\*/=]\s*\d",  # simple equations: x + 7 = 22
    r"\b\d+\s*%\b",  # percentage
    r"\$\s*\d+",  # currency amounts
    r"\b(?:km/?h|mph|m/s|kg|lbs?)\b",  # units → likely word problem
]

_MATH_WORD_PROBLEM_PATTERNS = [
    r"\bhow\s+(many|much|far|long|old|fast)\b.*\d",
    r"\b(if|suppose|given)\b.*\d.*\?\s*$",
    r"\b(recipe|shirt|book|train|car|worker|shop|price)\b.*\d",
]

_REASONING_ORDERING_PATTERNS = [
    r"\b\w+\s+is\s+(older|younger|taller|shorter|heavier|lighter|bigger|smaller|faster|slower)\s+than\b",
    r"\b(left|right)\s+of\b",
    r"\b(in\s+front|behind|next\s+to|above|below)\b",
    r"\bwho\s+is\s+(the\s+)?(youngest|oldest|tallest|shortest|fastest|slowest|first|last)\b",
]

_REASONING_LOGIC_PATTERNS = [
    r"\b(if|when)\s+.+\s+(then|implies|must)\b",
    r"\b(implies|entails|deduce|infer|conclude)\b",
    r"\b(all|some|no|every|none)\s+\w+\s+are\b",
    r"\b(probability|chance|odds)\s+of\b",
    r"\bmust\s+(be|not\s+be)\b.*\?",
    r"\bcan\s+we\s+conclude\b",
]

_REASONING_TEMPORAL_PATTERNS = [
    r"\bif\s+today\s+is\b",
    r"\b\d+\s+days?\s+from\b",
    r"\bday\s+of\s+the\s+week\b",
]

_EXTRACTION_VERB_PATTERNS = [
    r"\bextract\b",
    r"\bpull\s+out\b",
    r"\bfind\s+the\s+(\w+\s+)?(?:in|from):\s",
    r"\bwhat\s+\w+\s+(is|are)\s+(in|mentioned\s+in|from):\s",
    r"\bwhat\s+is\s+the\s+\w+\s+(in|from):\s",
]

_CLASSIFICATION_PATTERNS = [
    r"\b(classify|categorize|label)\b",
    r"\bwhich\s+(category|class|label|type|option)\b",
    r"\bsentiment\b.*\?",
    r"\btrue\s+or\s+false\b",
    r"\byes\s+or\s+no\b",
    r"\bspam\s+or\s+ham\b",
    r"\bpositive\s+or\s+negative\b",
    r"\bquestion\s+or\s+statement\b",
    r"\bOptions?:\s",  # explicit options list
]


def _has(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _has_inline_source(text: str) -> bool:
    """True if the prompt contains an inline source text passage.

    Signals: content after a colon at least 15 chars long, or a long quoted block.
    Distinguishes 'What is the capital of France?' (short_qa) from
    'What is the city in: She moved to Lisbon in 2019.' (extraction).
    """
    if re.search(r":\s+\S.{14,}", text):
        return True
    if re.search(r'["\'][^"\']{20,}["\']', text):
        return True
    return False


def classify(prompt: str) -> TaskFeatures:
    """Pure-Python heuristic. Zero tokens, sub-millisecond."""
    text = prompt
    chars = len(text)
    has_numbers = bool(re.search(r"\d", text))
    has_code = _has(_CODE_PATTERNS, text)
    has_inline = _has_inline_source(text)

    # Priority order matters — check most specific patterns first.
    if has_code:
        ttype = TaskType.CODE

    # Extraction: strong signal is inline source text + a "what is X in" or "extract" cue.
    # Check BEFORE math because "Extract the price ($24.99)" has math-adjacent content.
    elif has_inline and (
        _has(_EXTRACTION_VERB_PATTERNS, text)
        or re.search(r"^\s*what\s+\w+", text, re.IGNORECASE)
    ):
        ttype = TaskType.EXTRACTION

    # Reasoning: logical premises, ordering, temporal deduction.
    # Check BEFORE classification because "yes or no" appears in both — logical
    # premises are the tiebreaker.
    elif (
        _has(_REASONING_ORDERING_PATTERNS, text)
        or _has(_REASONING_LOGIC_PATTERNS, text)
        or _has(_REASONING_TEMPORAL_PATTERNS, text)
    ):
        ttype = TaskType.REASONING

    elif _has(_CLASSIFICATION_PATTERNS, text):
        ttype = TaskType.CLASSIFICATION

    elif _has(_MATH_ARITHMETIC_PATTERNS, text) or _has(_MATH_WORD_PROBLEM_PATTERNS, text):
        ttype = TaskType.MATH

    elif chars > 2000:
        ttype = TaskType.LONG_GEN
    else:
        ttype = TaskType.SHORT_QA

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

    length_bump = min(0.25, chars / 8000)
    difficulty = min(1.0, base + length_bump)

    return TaskFeatures(
        type=ttype,
        difficulty=difficulty,
        input_chars=chars,
        has_code=has_code,
        has_numbers=has_numbers,
        has_inline_source=has_inline,
        requires_reasoning=ttype in {TaskType.MATH, TaskType.REASONING, TaskType.CODE},
    )
