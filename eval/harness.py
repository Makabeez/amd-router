"""Eval harness.

Reads JSONL tasks ({"id", "prompt", "answer", "task_type"?}), routes each,
scores against gold, prints leaderboard-shaped summary.

Usage:
    python -m eval.harness --tasks eval/tasks/sample.jsonl --config configs/dev.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.router.base import RoutingTrace
from src.router.hybrid import HybridRouter
from src.router.strategies import get_extractor
from src.utils.metrics import RunMetrics

from .scorer import score_answer


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def run(router: HybridRouter, tasks: list[dict], extractor_name: str = "raw") -> RunMetrics:
    extractor = get_extractor(extractor_name)
    metrics = RunMetrics()
    t0 = time.time()
    for i, task in enumerate(tasks):
        trace = router.route(task["prompt"])
        # Score using the same extractor on the final text
        # Build a fake GenerationResult so we can reuse the extractor
        from src.backends.base import GenerationResult

        fake = GenerationResult(text=trace.final_text)
        predicted = extractor(fake)
        correct = score_answer(predicted, task["answer"], task.get("task_type"))
        metrics.add(trace, correct)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[{i+1}/{len(tasks)}] acc={metrics.accuracy:.3f} "
                f"remote_toks={metrics.total_remote_tokens} "
                f"local_rate={metrics.local_rate:.2f} "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
            )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--extractor", default="raw")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mock", action="store_true", help="Use mock backends only")
    args = parser.parse_args()

    if args.mock:
        from src.backends.fireworks import MockBackend
        from src.escalation.policies import ThresholdPolicy

        local = MockBackend(name="mock-local", canned="42")
        local.is_remote = False
        remote = MockBackend(name="mock-remote", canned="42")
        remote.is_remote = True  # pretend
        router = HybridRouter(local=local, remote=remote, policy=ThresholdPolicy())
    else:
        # Replace with real wiring on Day 1
        raise SystemExit(
            "Wire up your real backends in eval/harness.py main() before running."
        )

    tasks = load_tasks(args.tasks)
    metrics = run(router, tasks, extractor_name=args.extractor)
    summary = metrics.summary()
    print(json.dumps(summary, indent=2, default=str))

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
