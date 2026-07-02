"""Answer extractors.

Each task shape (MCQ, numeric, JSON field, free text) gets its own extractor.
The router uses these to (a) compute self-consistency agreement and (b) format
the final output. Swap in task-specific extractors on Jul 6 once formats are known.

Reasoning-model note (Jul 2026 catalog): remote models output thinking traces
before answers. All text-based extractors run through _strip_reasoning first.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..backends.base import GenerationResult

Extractor = Callable[[GenerationResult], str]


# Patterns that reasoning models emit before the actual answer.
# Order matters: strip explicit tags first, then heuristic markers.
_REASONING_TAG_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL),
]

# Markers that separate reasoning from final answer
_FINAL_ANSWER_MARKERS = [
    re.compile(r"(?i)\bfinal\s+answer\s*[:=]\s*(.+?)(?:\n\s*\n|$)", re.DOTALL),
    re.compile(r"(?i)\banswer\s*[:=]\s*(.+?)(?:\n\s*\n|$)", re.DOTALL),
    re.compile(r"(?i)\btherefore[,\s]+(.+?)(?:\n\s*\n|$)", re.DOTALL),
    re.compile(r"\\boxed\{([^}]+)\}"),
]


def _strip_reasoning(text: str) -> str:
    """Remove thinking traces and extract the final answer segment.

    1. Strip explicit reasoning tags.
    2. If a "Final Answer:" or "Answer:" marker exists, take the content after it.
    3. Otherwise return the last non-empty paragraph (heuristic: reasoning-model
       final answers appear at the end).
    """
    if not text:
        return text

    # Strip explicit tags
    cleaned = text
    for pattern in _REASONING_TAG_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = cleaned.strip()

    # Look for a final-answer marker
    for pattern in _FINAL_ANSWER_MARKERS:
        m = pattern.search(cleaned)
        if m:
            return m.group(1).strip().rstrip(".!?,;:")

    # No marker — take last non-empty paragraph (typically the conclusion)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if paragraphs and len(paragraphs) > 1:
        # If we have multiple paragraphs, the last is usually the answer
        return paragraphs[-1]

    return cleaned


def raw_text(r: GenerationResult) -> str:
    return _strip_reasoning(r.text)


def first_line(r: GenerationResult) -> str:
    stripped = _strip_reasoning(r.text)
    return stripped.splitlines()[0] if stripped else ""


def numeric_answer(r: GenerationResult) -> str:
    """Extract the first number from the response. For math tasks.

    Applies reasoning-strip first so we don't match numbers inside the trace.
    """
    stripped = _strip_reasoning(r.text)
    m = re.search(r"-?\d+(?:\.\d+)?", stripped.replace(",", ""))
    return m.group() if m else stripped


def multiple_choice_letter(r: GenerationResult) -> str:
    """Extract A/B/C/D answer letter."""
    stripped = _strip_reasoning(r.text)
    m = re.search(r"\b([A-E])\b", stripped)
    return m.group(1) if m else stripped


def yes_no(r: GenerationResult) -> str:
    stripped = _strip_reasoning(r.text).lower()
    if re.search(r"\byes\b|\btrue\b", stripped):
        return "yes"
    if re.search(r"\bno\b|\bfalse\b", stripped):
        return "no"
    return stripped


def json_field(field: str) -> Extractor:
    """Extract a named field from JSON output. Returns a closure."""

    def _extract(r: GenerationResult) -> str:
        text = _strip_reasoning(r.text)
        # Strip code fences
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?|\n?```$", "", text)
        try:
            data = json.loads(text)
            return str(data.get(field, "")).strip()
        except (json.JSONDecodeError, AttributeError):
            m = re.search(rf'"{field}"\s*:\s*"?([^",}}\n]+)"?', text)
            return m.group(1).strip() if m else text

    return _extract


def boxed_answer(r: GenerationResult) -> str:
    """LaTeX-style \\boxed{...} answer extraction."""
    m = re.search(r"\\boxed\{([^}]+)\}", r.text)
    return m.group(1).strip() if m else numeric_answer(r)


# Registry — swap by name in config
EXTRACTORS: dict[str, Extractor] = {
    "raw": raw_text,
    "first_line": first_line,
    "numeric": numeric_answer,
    "mcq": multiple_choice_letter,
    "yes_no": yes_no,
    "boxed": boxed_answer,
}


def get_extractor(name: str, **kwargs: Any) -> Extractor:
    if name == "json_field":
        return json_field(kwargs["field"])
    return EXTRACTORS[name]
