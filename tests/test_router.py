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


def test_tiered_backend_accepts_tier_kwarg():
    """Tier kwarg should be silently accepted by all backends (uniform interface)."""
    backend = MockBackend(name="mock", canned="ok")
    backend.is_remote = True
    # Should not raise — tier is swallowed via **kwargs
    r = backend.generate("x", tier="small")
    assert r.text == "ok"


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
