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
    """Extract the concise answer from model output.

    Handles two output shapes:
      - Reasoning models (remote): think first, answer last, often with a
        "Final Answer:" marker.
      - Chatty small models (local): give the answer first, then ramble
        ("Tokyo. The capital city of Japan is Tokyo. It is located...").

    Strategy:
      1. Strip explicit <think> tags.
      2. If an explicit answer marker exists, use the text after it.
      3. Else if the FIRST line is a short standalone answer (<= ~12 words),
         use it — this catches the answer-first chatty pattern.
      4. Else fall back to the full cleaned text (don't drop content).
    """
    if not text:
        return text

    cleaned = text
    for pattern in _REASONING_TAG_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Strip leaked format-placeholder literals that models sometimes echo.
    cleaned = cleaned.replace("<answer>", "").replace("<number>", "")
    cleaned = cleaned.strip()

    # 2. Explicit answer marker (reasoning models, templated prompts)
    for pattern in _FINAL_ANSWER_MARKERS:
        m = pattern.search(cleaned)
        if m:
            return m.group(1).strip().rstrip(".!?,;:")

    # 3. Answer-first pattern: short first line/sentence, then elaboration.
    # Split on the first sentence-ending punctuation or newline.
    first_seg = re.split(r"(?<=[.!?])\s|\n", cleaned, maxsplit=1)[0].strip()
    first_seg_clean = first_seg.rstrip(".!?,;:")
    # If the first segment is a compact answer AND there's more text after it
    # (i.e., the model rambled), prefer the compact first segment.
    if first_seg_clean and len(first_seg_clean.split()) <= 12 and len(cleaned) > len(first_seg) + 5:
        return first_seg_clean

    # 4. No clear structure — return the full cleaned text.
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
    # Classifiers scan the WHOLE output (minus <think> tags), not just the
    # concise first segment — the yes/no signal can appear anywhere.
    text = r.text
    for pattern in _REASONING_TAG_PATTERNS:
        text = pattern.sub("", text)
    low = text.lower()
    # Prefer an explicit "answer: yes/no" if present
    m = re.search(r"(?i)answer\s*[:=]\s*(yes|no|true|false)", low)
    if m:
        val = m.group(1)
        return "yes" if val in ("yes", "true") else "no"
    if re.search(r"\byes\b|\btrue\b", low):
        return "yes"
    if re.search(r"\bno\b|\bfalse\b", low):
        return "no"
    return _strip_reasoning(r.text).lower()


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


def summary_text(r: GenerationResult) -> str:
    """Summaries are free-form; just strip reasoning traces and clean whitespace."""
    import re as _re
    txt = _strip_reasoning(r.text)
    return _re.sub(r"\s+", " ", txt).strip()


def entities_text(r: GenerationResult) -> str:
    """NER output — strip reasoning, return the entity list as-is."""
    return _strip_reasoning(r.text).strip()


# Registry — swap by name in config
def code_answer(r: GenerationResult) -> str:
    """Extract code. Prefer the contents of a fenced ```...``` block.

    Chatty models wrap code in ```python ... ```; we want the code inside,
    not the fence. If no fence, return the reasoning-stripped text.
    """
    text = r.text
    # Strip <think> tags first
    for pattern in _REASONING_TAG_PATTERNS:
        text = pattern.sub("", text)
    # Pull the first fenced block's contents
    m = re.search(r"```(?:\w+)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code:
            return code
    # No usable fence — strip a leading bare ```lang line if present, return rest
    text = re.sub(r"^\s*```\w*\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


EXTRACTORS: dict[str, Extractor] = {
    "raw": raw_text,
    "first_line": first_line,
    "numeric": numeric_answer,
    "mcq": multiple_choice_letter,
    "yes_no": yes_no,
    "boxed": boxed_answer,
    "summary": summary_text,
    "code": code_answer,
    "entities": entities_text,
}


def get_extractor(name: str, **kwargs: Any) -> Extractor:
    if name == "json_field":
        return json_field(kwargs["field"])
    return EXTRACTORS[name]
