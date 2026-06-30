"""Bench candidate local models on a task suite.

Run before Jul 6 to know which small models are worth carrying into the
standardized eval env. The env will cap local model size — find which one
gives the best accuracy/latency for what we expect to see.

Usage:
    python scripts/bench_local_models.py \\
        --models Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-1B-Instruct \\
        --tasks eval/tasks/sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backends.local import LocalTransformersBackend
from eval.scorer import score_answer
from src.router.strategies import get_extractor


def bench_one(model_id: str, tasks: list[dict], extractor_name: str) -> dict:
    print(f"\n=== {model_id} ===")
    backend = LocalTransformersBackend(model_id=model_id)
    extractor = get_extractor(extractor_name)

    correct = 0
    total_out_tokens = 0
    total_time = 0.0
    for task in tasks:
        t0 = time.time()
        r = backend.generate(task["prompt"], max_tokens=256, return_logprobs=False)
        total_time += time.time() - t0
        total_out_tokens += r.local_output_tokens

        pred = extractor(r)
        if score_answer(pred, task["answer"], task.get("task_type")):
            correct += 1

    result = {
        "model": model_id,
        "n": len(tasks),
        "accuracy": round(correct / len(tasks), 4),
        "avg_output_tokens": round(total_out_tokens / len(tasks), 1),
        "avg_latency_s": round(total_time / len(tasks), 3),
        "total_time_s": round(total_time, 1),
    }
    print(json.dumps(result, indent=2))

    # Free GPU memory before next model
    import torch

    del backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--extractor", default="raw")
    parser.add_argument("--out", type=Path, default=Path("bench_results.json"))
    args = parser.parse_args()

    with args.tasks.open() as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    results = [bench_one(m, tasks, args.extractor) for m in args.models]

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.out}")

    # Pareto-print
    print("\n=== Pareto: accuracy / tokens / latency ===")
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        print(
            f"{r['model']:<50} acc={r['accuracy']}  "
            f"tok={r['avg_output_tokens']}  lat={r['avg_latency_s']}s"
        )


if __name__ == "__main__":
    main()
