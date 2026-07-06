"""Prompt templates per task type.

Small local models (≤3B) follow instructions far better when the prompt is
framed correctly. Raw `What is X?` to a 1.5B model often produces meandering
output; the same prompt wrapped in "Answer with only the value, no explanation"
produces clean extractable answers.

This is the single highest-EV component for reducing escalation rate.
"""

from __future__ import annotations

from .heuristic import TaskType


# Local templates: aggressive on brevity, output format constrained.
# Remote templates: looser — bigger models can self-format.
LOCAL_TEMPLATES: dict[TaskType, str] = {
    TaskType.EXTRACTION: (
        "Extract the requested value from the text. "
        "Respond with ONLY the extracted value, no preamble, no explanation, no quotes.\n\n"
        "Task: {prompt}\n\nAnswer:"
    ),
    TaskType.CLASSIFICATION: (
        "Classify the input. Respond with ONLY the category label, lowercase, no explanation.\n\n"
        "Task: {prompt}\n\nLabel:"
    ),
    TaskType.SHORT_QA: (
        "Answer the question in as few words as possible. No preamble, no explanation.\n\n"
        "Question: {prompt}\n\nAnswer:"
    ),
    TaskType.MATH: (
        "Solve the problem. Show brief reasoning then on the LAST line write only:\n"
        "Answer: <number>\n\n"
        "Problem: {prompt}"
    ),
    TaskType.REASONING: (
        "Think step by step, then on the LAST line write only:\n"
        "Answer: <answer>\n\n"
        "Question: {prompt}"
    ),
    TaskType.CODE: (
        "Answer the code question. If asked for a value, output only the value. "
        "If asked for code, output only the code in a single ```python block.\n\n"
        "Task: {prompt}\n\nAnswer:"
    ),
    TaskType.SUMMARIZATION: (
        "Summarize the text as instructed. Output ONLY the summary, "
        "no preamble, respecting any length or format constraint given.\n\n"
        "Task: {prompt}\n\nSummary:"
    ),
    TaskType.NER: (
        "Extract the named entities as instructed. Output ONLY the entities "
        "(and their labels if requested), no explanation.\n\n"
        "Task: {prompt}\n\nEntities:"
    ),
    TaskType.LONG_GEN: "{prompt}",  # don't constrain free-form generation
    TaskType.UNKNOWN: "{prompt}",
}

# Remote: reasoning models will "think" regardless; ask for a clear final-answer
# marker so extractors can slice out the answer without the trace.
_REMOTE_ANSWER_MARKER_TEMPLATE = (
    "{prompt}\n\n"
    "Provide your answer in this exact format on the LAST line:\n"
    "Final Answer: <answer>"
)

REMOTE_TEMPLATES: dict[TaskType, str] = {
    t: _REMOTE_ANSWER_MARKER_TEMPLATE for t in TaskType
}
REMOTE_TEMPLATES[TaskType.LONG_GEN] = "{prompt}"
REMOTE_TEMPLATES[TaskType.UNKNOWN] = _REMOTE_ANSWER_MARKER_TEMPLATE


def format_for_local(prompt: str, task_type: TaskType) -> str:
    """Wrap a raw prompt in the task-appropriate local template."""
    template = LOCAL_TEMPLATES.get(task_type, "{prompt}")
    return template.format(prompt=prompt)


def format_for_remote(prompt: str, task_type: TaskType) -> str:
    """Lighter wrap for remote models."""
    template = REMOTE_TEMPLATES.get(task_type, "{prompt}")
    return template.format(prompt=prompt)


def format_verify(original_prompt: str, draft: str, task_type: TaskType) -> str:
    """Verification mode: ask remote to confirm or correct a local draft.

    Caller is responsible for the length guard (skip verify if it inflates tokens).
    """
    return (
        f"Original task: {original_prompt}\n\n"
        f"Draft answer: {draft}\n\n"
        f"If the draft is correct, output it exactly. "
        f"If incorrect, output only the corrected answer. No explanation."
    )
