"""Answer extractors.

Each task shape (MCQ, numeric, JSON field, free text) gets its own extractor.
The router uses these to (a) compute self-consistency agreement and (b) format
the final output. Swap in task-specific extractors on Jul 6 once formats are known.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..backends.base import GenerationResult

Extractor = Callable[[GenerationResult], str]


def raw_text(r: GenerationResult) -> str:
    return r.text.strip()


def first_line(r: GenerationResult) -> str:
    return r.text.strip().splitlines()[0] if r.text.strip() else ""


def numeric_answer(r: GenerationResult) -> str:
    """Extract the first number from the response. For math tasks."""
    m = re.search(r"-?\d+(?:\.\d+)?", r.text.replace(",", ""))
    return m.group() if m else r.text.strip()


def multiple_choice_letter(r: GenerationResult) -> str:
    """Extract A/B/C/D answer letter."""
    m = re.search(r"\b([A-E])\b", r.text.strip())
    return m.group(1) if m else r.text.strip()


def yes_no(r: GenerationResult) -> str:
    text = r.text.strip().lower()
    if re.search(r"\byes\b|\btrue\b", text):
        return "yes"
    if re.search(r"\bno\b|\bfalse\b", text):
        return "no"
    return text


def json_field(field: str) -> Extractor:
    """Extract a named field from JSON output. Returns a closure."""

    def _extract(r: GenerationResult) -> str:
        text = r.text.strip()
        # Strip code fences
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?|\n?```$", "", text)
        try:
            data = json.loads(text)
            return str(data.get(field, "")).strip()
        except (json.JSONDecodeError, AttributeError):
            # Fall back to regex
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
