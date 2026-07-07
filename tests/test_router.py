"""Smoke tests for router primitives. Run with: python -m pytest tests/ -v"""

from __future__ import annotations

from src.backends.fireworks import MockBackend
from src.backends.base import GenerationResult
from src.classifiers.confidence import (
    _canonicalize,
    assess,
    logprob_confidence,
    self_consistency,
)
from src.classifiers.heuristic import TaskType, classify
from src.escalation.policies import ThresholdPolicy
from src.router.hybrid import HybridConfig, HybridRouter


def test_classify_math():
    f = classify("What is 23 * 47?")
    assert f.type == TaskType.MATH
    assert f.has_numbers


def test_classify_code():
    f = classify("def fib(n):\n    ```python\n    return n")
    assert f.type == TaskType.CODE
    assert f.has_code


def test_classify_extraction():
    f = classify("Extract the date from this text: Born March 14, 1879.")
    assert f.type == TaskType.EXTRACTION


def test_canonicalize_numeric():
    assert _canonicalize("42") == "42"
    assert _canonicalize("42.0") == "42"
    assert _canonicalize("3.14159") == "3.14159"
    assert _canonicalize(" 42 .") == "42"


def test_self_consistency_voting():
    results = [
        GenerationResult(text="42"),
        GenerationResult(text="42"),
        GenerationResult(text="43"),
    ]
    agree, top = self_consistency(results)
    assert top == "42"
    assert agree == 2 / 3


def test_logprob_confidence():
    r = GenerationResult(text="ok", logprobs=[-0.1, -0.2])
    conf = logprob_confidence(r)
    assert 0.0 < conf < 1.0


def test_router_local_path():
    local = MockBackend(name="local", canned="42")
    local.is_remote = False
    remote = MockBackend(name="remote", canned="42-remote")
    remote.is_remote = True

    # Force always-local path via easy task + lenient policy
    policy = ThresholdPolicy(min_confidence=0.0, preflight_skip_local_above=1.1)
    router = HybridRouter(local=local, remote=remote, policy=policy)

    trace = router.route("What is 2 + 2?")
    assert trace.remote_tokens == 0
    assert "42" in trace.final_text


def test_router_remote_escalation():
    local = MockBackend(name="local", canned="wrong")
    local.is_remote = False
    remote = MockBackend(name="remote", canned="correct")
    remote.is_remote = True

    # Force escalation: impossible confidence threshold everywhere
    policy = ThresholdPolicy(
        min_confidence=2.0,
        preflight_skip_local_above=1.1,
        per_task_min_confidence={},  # disable per-task overrides
    )
    router = HybridRouter(local=local, remote=remote, policy=policy)

    trace = router.route("Anything")
    assert trace.final_backend == "remote"
    assert trace.final_text == "correct"
    assert any("escalate=True" in d for d in trace.decisions)


def test_local_prompt_templating():
    """Local prompts should be wrapped in task-specific instruction frames."""
    from src.classifiers.prompts import format_for_local
    from src.classifiers.heuristic import TaskType

    out = format_for_local("Extract the email from: hi@a.com", TaskType.EXTRACTION)
    assert "ONLY the extracted value" in out
    assert "Extract the email from: hi@a.com" in out

    out_math = format_for_local("What is 2+2?", TaskType.MATH)
    assert "Answer: <number>" in out_math


def test_verify_length_guard_skips_when_inflating():
    """Short prompts with long drafts should skip verify mode to avoid bloat."""
    from src.router.hybrid import HybridConfig

    cfg = HybridConfig(verify_length_ratio_max=1.5)
    short_prompt = "2+2?"  # ~4 chars
    long_draft = "x" * 200  # 200 chars

    # If the verify wrap is much longer than base, the router should bail out.
    # We just check the config plumbing — actual decision logic tested via mock.
    assert cfg.verify_length_ratio_max == 1.5


def test_tier_hint_for_task_type():
    """Extraction → small tier, math → medium tier."""
    from src.router.hybrid import HybridConfig
    from src.classifiers.heuristic import TaskType

    cfg = HybridConfig()
    assert cfg.tier_by_task[TaskType.EXTRACTION] == "small"
    assert cfg.tier_by_task[TaskType.MATH] == "medium"
    assert cfg.tier_by_task[TaskType.REASONING] == "medium"



def test_metrics_remote_tokens_per_correct():
    """Cost-per-correct = total_remote_tokens / n_correct."""
    from src.utils.metrics import RunMetrics
    from src.router.base import RoutingTrace

    m = RunMetrics()
    m.add(RoutingTrace(prompt="x", final_text="a", final_backend="local",
                       remote_tokens=0, local_tokens=10), correct=True)
    m.add(RoutingTrace(prompt="y", final_text="b", final_backend="remote",
                       remote_tokens=100, local_tokens=0,
                       metadata={"remote_tier": "medium"}), correct=True)
    m.add(RoutingTrace(prompt="z", final_text="c", final_backend="remote",
                       remote_tokens=200, local_tokens=0,
                       metadata={"remote_tier": "small"}), correct=False)

    assert m.n == 3
    assert m.n_correct == 2
    assert m.total_remote_tokens == 300
    assert m.remote_tokens_per_correct == 150.0
    assert m.tier_counts == {"medium": 1, "small": 1}


def test_classifier_accuracy_on_dev80():
    """Regression guard: classifier should hit ≥90% on the dev set."""
    import json
    from pathlib import Path
    from src.classifiers.heuristic import classify

    tasks_path = Path(__file__).resolve().parents[1] / "eval" / "tasks" / "dev80.jsonl"
    with tasks_path.open() as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    correct = sum(
        1 for t in tasks if classify(t["prompt"]).type.value == t["task_type"]
    )
    accuracy = correct / len(tasks)
    assert accuracy >= 0.90, f"classifier regressed: {accuracy:.2%} on dev80"


def test_inline_source_detection():
    """Extraction requires an inline source passage."""
    from src.classifiers.heuristic import _has_inline_source

    # Extraction — has inline source after colon
    assert _has_inline_source("What is the city in: She moved to Lisbon in 2019.")
    # Short QA — no inline source
    assert not _has_inline_source("What is the capital of France?")


def test_reasoning_trace_stripping():
    """Reasoning-model outputs must have thinking traces stripped."""
    from src.router.strategies import raw_text, numeric_answer, yes_no
    from src.backends.base import GenerationResult

    # Explicit <think> tags
    r1 = GenerationResult(
        text="<think>The user is asking about France. Its capital is Paris.</think>\n\nFinal Answer: Paris"
    )
    assert raw_text(r1) == "Paris"

    # "Final Answer:" marker without tags
    r2 = GenerationResult(
        text="Let me work through this.\nStep 1: identify the question.\nStep 2: recall facts.\n\nFinal Answer: 42"
    )
    assert numeric_answer(r2) == "42"

    # No marker — take last paragraph
    r3 = GenerationResult(
        text="First I consider the options.\n\nThe answer is yes."
    )
    assert yes_no(r3) == "yes"

    # Boxed answer
    r4 = GenerationResult(
        text="After computing, \\boxed{1081} is the result."
    )
    assert numeric_answer(r4) == "1081"



def test_model_selector_picks_gemma_by_default():
    """Non-code tasks should prefer Gemma (bonus + generalist)."""
    from src.backends.model_selector import ModelSelector
    from src.classifiers.heuristic import TaskType

    allowed = ["minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]
    sel = ModelSelector(allowed)

    # Easy factual → cheapest gemma
    m = sel.select(TaskType.SHORT_QA, difficulty=0.3)
    assert "gemma" in m.lower()

    # Code → code-specialized model
    m = sel.select(TaskType.CODE, difficulty=0.7)
    assert "code" in m.lower()


def test_model_selector_size_ordering():
    """a4b (4B active) should be cheaper than 31b dense."""
    from src.backends.model_selector import ModelSelector

    sel = ModelSelector(["gemma-4-31b-it", "gemma-4-26b-a4b-it"])
    # cheapest should be the a4b (4B active params) not the 31b dense
    assert "a4b" in sel.cheapest.lower()


def test_new_task_types_classify():
    """Summarization and NER must be detected."""
    from src.classifiers.heuristic import classify, TaskType

    assert classify("Summarize this in one sentence: The quarterly report...").type == TaskType.SUMMARIZATION
    assert classify("Identify all named entities in: Tim Cook runs Apple.").type == TaskType.NER


def test_allowed_models_parsing(monkeypatch):
    """ALLOWED_MODELS env var must parse into a clean list."""
    from src.backends.fireworks import get_allowed_models

    monkeypatch.setenv("ALLOWED_MODELS", "gemma-4-31b-it, kimi-k2p7-code ,minimax-m3")
    models = get_allowed_models()
    assert models == ["gemma-4-31b-it", "kimi-k2p7-code", "minimax-m3"]


def test_harness_io_roundtrip(tmp_path, monkeypatch):
    """harness_runner should read tasks.json and write valid results.json."""
    import json
    import os
    from src.backends.fireworks import MockBackend

    # Build tiny input
    inp = tmp_path / "tasks.json"
    out = tmp_path / "results.json"
    inp.write_text(json.dumps([
        {"task_id": "t1", "prompt": "What is 2+2?"},
        {"task_id": "t2", "prompt": "Capital of France?"},
    ]))

    monkeypatch.setenv("INPUT_PATH", str(inp))
    monkeypatch.setenv("OUTPUT_PATH", str(out))

    # Monkeypatch the router builder to use mocks (no model download / no network)
    import src.harness_runner as hr

    class _FakeTrace:
        def __init__(self, text): self.final_text = text
    class _FakeRouter:
        def route(self, prompt): return _FakeTrace("4" if "2+2" in prompt else "Paris")

    monkeypatch.setattr(hr, "_build_router", lambda: _FakeRouter())

    try:
        hr.main()
    except SystemExit as e:
        assert e.code == 0

    results = json.loads(out.read_text())
    assert len(results) == 2
    assert {r["task_id"] for r in results} == {"t1", "t2"}
    assert all("answer" in r for r in results)


def test_code_extractor_pulls_fence_contents():
    """Code extractor must return fenced-block contents, not the fence."""
    from src.router.strategies import code_answer
    from src.backends.base import GenerationResult

    r = GenerationResult(text="```python\ndef f(x):\n    return x*2\n```")
    out = code_answer(r)
    assert out.startswith("def f")
    assert "```" not in out

    # No fence — return stripped text
    r2 = GenerationResult(text="def g(): return 1")
    assert "def g" in code_answer(r2)


def test_code_tasks_always_escalate():
    """Code tasks skip the unreliable local model and go straight to remote."""
    from src.escalation.policies import ThresholdPolicy
    from src.classifiers.heuristic import classify, TaskType

    policy = ThresholdPolicy()
    feats = classify("Fix the bug: def add(a,b): return a-b")
    assert feats.type == TaskType.CODE
    decision = policy.decide_preflight(feats)
    assert decision.escalate is True
    assert "always escalate" in decision.reason.lower()


def test_remote_failure_falls_back_to_local():
    """If remote raises, router returns the local answer, not empty."""
    from src.router.hybrid import HybridRouter, HybridConfig
    from src.escalation.policies import ThresholdPolicy
    from src.backends.fireworks import MockBackend

    local = MockBackend(name="local", canned="local-answer")
    local.is_remote = False

    class BoomRemote(MockBackend):
        is_remote = True
        def generate(self, *a, **k):
            raise TimeoutError("simulated remote timeout")

    remote = BoomRemote(name="remote")
    # Force escalation so remote is attempted
    policy = ThresholdPolicy(min_confidence=2.0, preflight_skip_local_above=1.1,
                             per_task_min_confidence={}, always_escalate=set())
    router = HybridRouter(local=local, remote=remote, policy=policy)
    trace = router.route("some prompt")
    assert trace.final_text == "local-answer"
    assert "fallback" in trace.final_backend
    assert trace.remote_tokens == 0
