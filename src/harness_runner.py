"""AMD Track 1 harness entrypoint.

Contract (Participant Guide):
  1. Read /input/tasks.json  -> [{"task_id", "prompt"}, ...]
  2. Route each task (local-first, escalate to Fireworks only when needed)
  3. Write /output/results.json -> [{"task_id", "answer"}, ...]
  4. Exit 0 on success, non-zero on failure
  5. Total runtime <= 10 min; per-request < 30s; ready < 60s

Env (harness-injected at eval; use .env locally):
  FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS

Strategy: local model answers cheaply (0 tokens scored). Escalate to a
Fireworks model from ALLOWED_MODELS only when local confidence is low.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _load_dotenv_if_present() -> None:
    """Load a local .env for DEV convenience only.

    At eval time the harness injects FIREWORKS_API_KEY / FIREWORKS_BASE_URL /
    ALLOWED_MODELS as real environment variables — those always win because we
    use setdefault (never overwrite an already-set var). If no .env exists
    (the eval image must not bundle one), this is a silent no-op.
    """
    for candidate in (Path("/app/.env"), Path(__file__).resolve().parents[1] / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            break


_load_dotenv_if_present()

INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))

# Global time budget guard (leave headroom under the 10-min cap)
TOTAL_BUDGET_S = float(os.environ.get("TOTAL_BUDGET_S", "540"))  # 9 min
PER_TASK_BUDGET_S = float(os.environ.get("PER_TASK_BUDGET_S", "28"))  # < 30s rule


def _load_tasks() -> list[dict]:
    if not INPUT_PATH.exists():
        print(f"FATAL: {INPUT_PATH} not found", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(INPUT_PATH.read_text())
    except json.JSONDecodeError as e:
        print(f"FATAL: {INPUT_PATH} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print("FATAL: tasks.json must be a JSON array", file=sys.stderr)
        sys.exit(1)
    return data


def _write_results(results: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish write: temp then rename, so a partial crash can't corrupt output
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False))
    tmp.rename(OUTPUT_PATH)


def _build_router():
    """Wire the router from env. Import here so import errors surface at runtime."""
    from src.backends.local import LocalTransformersBackend
    from src.backends.fireworks import FireworksBackend, get_allowed_models
    from src.backends.harness_remote import HarnessRemoteBackend
    from src.router.hybrid import HybridConfig, HybridRouter
    from src.escalation.policies import ThresholdPolicy

    local_model = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    local = LocalTransformersBackend(model_id=local_model)

    allowed = get_allowed_models()
    if not allowed:
        # Dev fallback so local runs don't crash without ALLOWED_MODELS set.
        allowed = [os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/gemma-4-31b-it")]
    remote = HarnessRemoteBackend(allowed_models=allowed)

    policy = ThresholdPolicy(
        min_confidence=float(os.environ.get("MIN_CONFIDENCE", "0.65")),
        preflight_skip_local_above=float(os.environ.get("PREFLIGHT_SKIP_ABOVE", "0.9")),
    )
    config = HybridConfig(
        n_samples=int(os.environ.get("N_SAMPLES", "0")),
        local_max_tokens=int(os.environ.get("LOCAL_MAX_TOKENS", "512")),
        remote_max_tokens=int(os.environ.get("REMOTE_MAX_TOKENS", "1024")),
    )
    return HybridRouter(local=local, remote=remote, policy=policy, config=config)


def main() -> None:
    t0 = time.time()
    tasks = _load_tasks()
    print(f"loaded {len(tasks)} tasks", file=sys.stderr)

    # Build router once (model load counts against the 60s readiness budget).
    try:
        router = _build_router()
    except Exception as e:
        # If the router can't build, still emit a valid (empty-answer) file so
        # we don't score zero for malformed output — every task gets a stub.
        print(f"router build failed: {type(e).__name__}: {e}", file=sys.stderr)
        results = [{"task_id": t.get("task_id"), "answer": ""} for t in tasks]
        _write_results(results)
        sys.exit(1)

    from src.router.strategies import get_extractor
    # Per-task-type extractor: code tasks need fence-aware extraction, NER/summary
    # their own, etc. A single global extractor mangles structured outputs.
    from src.classifiers.heuristic import classify, TaskType
    _EXTRACTOR_BY_TYPE = {
        TaskType.CODE: "code",
        TaskType.SUMMARIZATION: "summary",
        TaskType.NER: "entities",
        TaskType.MATH: "raw",          # numeric extraction is greedy; raw+strip is safer
        TaskType.CLASSIFICATION: "raw",
        TaskType.SHORT_QA: "raw",
        TaskType.EXTRACTION: "raw",
        TaskType.REASONING: "raw",
    }
    # Global override via env — but ONLY when explicitly set to a real
    # non-default extractor. "raw" (or unset) means "use per-task extractors",
    # otherwise a stale ANSWER_EXTRACTOR=raw silently disables code/NER/summary
    # extraction and mangles those categories.
    _forced = os.environ.get("ANSWER_EXTRACTOR", "").strip()
    if _forced in ("", "raw", "auto", "per_task"):
        _forced = ""
    default_extractor = get_extractor("raw")
    from src.backends.base import GenerationResult

    results: list[dict] = []
    for i, task in enumerate(tasks):
        tid = task.get("task_id")
        prompt = task.get("prompt", "")

        # Time-budget guard: if we're near the cap, stub remaining tasks fast.
        elapsed = time.time() - t0
        if elapsed > TOTAL_BUDGET_S:
            print(f"time budget hit at task {i}; stubbing rest", file=sys.stderr)
            results.append({"task_id": tid, "answer": ""})
            continue

        try:
            trace = router.route(prompt)
            if _forced:
                extractor = get_extractor(_forced)
            else:
                ttype = classify(prompt).type
                extractor = get_extractor(_EXTRACTOR_BY_TYPE.get(ttype, "raw"))
            answer = extractor(GenerationResult(text=trace.final_text))
        except Exception as e:
            print(f"task {tid} failed: {type(e).__name__}: {e}", file=sys.stderr)
            answer = ""

        results.append({"task_id": tid, "answer": answer})

        # Persist incrementally so a late crash still yields a full file.
        if (i + 1) % 10 == 0:
            _write_results(results)
            print(f"[{i+1}/{len(tasks)}] elapsed={time.time()-t0:.1f}s", file=sys.stderr)

    _write_results(results)
    print(f"done: {len(results)} results in {time.time()-t0:.1f}s", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
