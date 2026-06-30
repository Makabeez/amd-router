"""Threshold calibration.

Sweeps escalation thresholds and per-task confidence cutoffs against a held-out
eval set, plotting accuracy-vs-tokens Pareto. Use after a dry-run on real
hackathon tasks to lock optimal thresholds before final submission.

Usage:
    python scripts/calibrate_thresholds.py --tasks eval/tasks/sample.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.escalation.policies import ThresholdPolicy
from src.router.hybrid import HybridConfig, HybridRouter
from src.router.strategies import get_extractor
from src.utils.metrics import RunMetrics
from eval.harness import load_tasks, run


def calibrate(
    local_factory, remote_factory, tasks: list[dict], extractor_name: str = "raw"
) -> list[dict]:
    """Grid-search over confidence thresholds and preflight cutoffs.

    `local_factory` and `remote_factory` are zero-arg callables that return
    fresh Backend instances. We rebuild backends across runs to keep state clean.
    """
    grid = list(
        itertools.product(
            [0.55, 0.65, 0.75],  # min_confidence
            [0.75, 0.85, 0.95],  # preflight_skip_above
            [0, 3, 5],            # n_samples
        )
    )

    results = []
    for min_conf, preflight, n_samples in grid:
        policy = ThresholdPolicy(
            min_confidence=min_conf,
            preflight_skip_local_above=preflight,
        )
        cfg = HybridConfig(n_samples=n_samples)
        local = local_factory()
        remote = remote_factory()
        router = HybridRouter(local=local, remote=remote, policy=policy, config=cfg)

        metrics = run(router, tasks, extractor_name=extractor_name)
        s = metrics.summary()
        s["params"] = {
            "min_confidence": min_conf,
            "preflight_skip_local_above": preflight,
            "n_samples": n_samples,
        }
        results.append(s)
        print(
            f"min_conf={min_conf} pre={preflight} n={n_samples} "
            f"=> acc={s['accuracy']} remote_toks={s['total_remote_tokens']}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("calibration.json"))
    args = parser.parse_args()

    print("Wire up local_factory/remote_factory in this script before running.")
    print("Stub left intentionally to avoid bg model downloads in CI.")
    # Example:
    # local_factory = lambda: LocalTransformersBackend("Qwen/Qwen2.5-1.5B-Instruct")
    # remote_factory = lambda: FireworksBackend("accounts/fireworks/models/...")


if __name__ == "__main__":
    main()
