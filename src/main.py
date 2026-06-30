"""Main entrypoint for the routing agent.

Submission contract (assumed — confirm Jul 6):
  - Reads JSONL tasks from stdin OR a file path arg
  - Writes JSONL predictions to stdout OR an output path
  - One line per task: {"id": <id>, "answer": <text>}

This is the script the judge runs. Keep it dumb — all logic lives in the router.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.backends.fireworks import FireworksBackend
from src.backends.local import LocalTransformersBackend
from src.router.hybrid import HybridConfig, HybridRouter
from src.router.strategies import get_extractor
from src.escalation.policies import ThresholdPolicy


def build_router() -> HybridRouter:
    """Build the router from env config. Tune defaults via .env."""
    local_model = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    remote_model = os.environ.get(
        "FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"
    )

    local = LocalTransformersBackend(model_id=local_model)
    remote = FireworksBackend(model=remote_model)

    policy = ThresholdPolicy(
        min_confidence=float(os.environ.get("MIN_CONFIDENCE", "0.65")),
        preflight_skip_local_above=float(
            os.environ.get("PREFLIGHT_SKIP_ABOVE", "0.85")
        ),
    )
    config = HybridConfig(
        n_samples=int(os.environ.get("N_SAMPLES", "0")),
        local_max_tokens=int(os.environ.get("LOCAL_MAX_TOKENS", "512")),
        remote_max_tokens=int(os.environ.get("REMOTE_MAX_TOKENS", "512")),
    )

    extractor_name = os.environ.get("ANSWER_EXTRACTOR", "raw")
    extractor = get_extractor(extractor_name)

    return HybridRouter(
        local=local, remote=remote, policy=policy, config=config,
        answer_extractor=extractor,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="JSONL input; stdin if omitted")
    parser.add_argument("--output", type=Path, default=None, help="JSONL output; stdout if omitted")
    parser.add_argument("--trace", type=Path, default=None, help="Write traces JSONL here")
    args = parser.parse_args()

    in_stream = args.input.open() if args.input else sys.stdin
    out_stream = args.output.open("w") if args.output else sys.stdout
    trace_stream = args.trace.open("w") if args.trace else None

    router = build_router()
    extractor_name = os.environ.get("ANSWER_EXTRACTOR", "raw")
    extractor = get_extractor(extractor_name)

    total_remote = 0
    n = 0
    try:
        for line in in_stream:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            trace = router.route(task["prompt"])

            from src.backends.base import GenerationResult

            answer = extractor(GenerationResult(text=trace.final_text))
            out_stream.write(
                json.dumps({"id": task.get("id"), "answer": answer}) + "\n"
            )
            out_stream.flush()

            if trace_stream:
                trace_stream.write(
                    json.dumps(
                        {
                            "id": task.get("id"),
                            "backend": trace.final_backend,
                            "remote_tokens": trace.remote_tokens,
                            "local_tokens": trace.local_tokens,
                            "decisions": trace.decisions,
                        }
                    )
                    + "\n"
                )

            total_remote += trace.remote_tokens
            n += 1

        print(
            f"# done: n={n} total_remote_tokens={total_remote} "
            f"avg_remote_tokens={total_remote/max(n,1):.1f}",
            file=sys.stderr,
        )
    finally:
        if args.input:
            in_stream.close()
        if args.output:
            out_stream.close()
        if trace_stream:
            trace_stream.close()


if __name__ == "__main__":
    main()
