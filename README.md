<div align="center">

<svg viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0a0a0a"/>
      <stop offset="50%" stop-color="#1a0505"/>
      <stop offset="100%" stop-color="#0a0a0a"/>
    </linearGradient>
    <linearGradient id="text-grad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#ED1C24">
        <animate attributeName="stop-color" values="#ED1C24;#ff5555;#ED1C24" dur="3s" repeatCount="indefinite"/>
      </stop>
      <stop offset="100%" stop-color="#ffffff"/>
    </linearGradient>
  </defs>
  <rect width="800" height="200" fill="url(#bg)"/>
  <g font-family="ui-monospace, SFMono-Regular, Menlo, monospace" text-anchor="middle">
    <text x="400" y="80" font-size="48" font-weight="900" fill="url(#text-grad)">AMD ROUTER</text>
    <text x="400" y="115" font-size="14" fill="#aaa">hybrid token-efficient routing agent</text>
    <text x="400" y="155" font-size="11" fill="#666">local-first · escalate on uncertainty · zero local tokens</text>
  </g>
  <g stroke="#ED1C24" stroke-width="1" opacity="0.5">
    <line x1="80" y1="180" x2="720" y2="180">
      <animate attributeName="x2" values="80;720;80" dur="6s" repeatCount="indefinite"/>
    </line>
  </g>
</svg>

**Local-first routing. Escalate only when the small model says "I don't know."**

![status](https://img.shields.io/badge/status-validated_8%2F8-success?style=for-the-badge)
![track](https://img.shields.io/badge/track-1_routing-ED1C24?style=for-the-badge)
![event](https://img.shields.io/badge/AMD-Hackathon_Act_II-000000?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge)
![image](https://img.shields.io/badge/image-ghcr.io%2Fmakabeez%2Famd--router-2088FF?style=for-the-badge&logo=github)

</div>

> Track 1 scoring is `accuracy × (1 / remote_tokens)` in disguise. Local tokens are free. So we burn local generously to gain confidence, and only call Fireworks when the local model genuinely can't commit. The router is the product — base models are interchangeable.

---

## Architecture

```
                            ┌────────────────────────┐
                            │   Heuristic Classifier │   (free, regex)
                            │   task_type + difficulty│
                            └──────────┬─────────────┘
                                       │
                            ┌──────────▼─────────────┐
                            │   Escalation Policy    │
                            │   preflight: skip local?│
                            └────┬────────────┬──────┘
                                 │ no          │ yes
                  ┌──────────────▼──┐          │
                  │ Local Backend   │          │
                  │ + logprobs      │          │
                  │ + n samples     │          │
                  └────────┬────────┘          │
                           │                   │
                  ┌────────▼────────┐          │
                  │ Confidence      │          │
                  │ logprob × vote  │          │
                  └────────┬────────┘          │
                           │                   │
                  ┌────────▼─────────┐         │
                  │ Escalation       │         │
                  │ postlocal: send? │         │
                  └────┬─────────┬───┘         │
                  no  │     yes │              │
                       │         └──────┬──────┘
                       │                ▼
                       │       ┌──────────────────┐
                       │       │ Fireworks API    │  ← only paid tokens
                       │       │ (optional verify │
                       │       │  mode: draft     │
                       │       │  attached)       │
                       │       └────────┬─────────┘
                       │                │
                       └────────┬───────┘
                                ▼
                        ┌───────────────┐
                        │ Final answer  │
                        │ + RoutingTrace│
                        └───────────────┘
```

## Why this design

Four conviction points behind the scaffold:

1. **Heuristic preflight is free.** Task-type detection runs on the raw prompt with regex. We never call a model just to decide which model to call.
2. **Local samples are free.** Self-consistency with `n=3–5` samples at temperature > 0 is the cheapest confidence signal in the game. Use it.
3. **Verify mode beats fresh remote.** When escalating, pass the local draft to the remote model as context — verification is ~30–60% cheaper than fresh generation.
4. **Thresholds are calibrated, not guessed.** `scripts/calibrate_thresholds.py` grid-searches confidence cutoffs against a held-out eval set after the real tasks drop on Jul 6.

## Repo layout

```
amd-router/
├── src/
│   ├── harness_runner.py          # container entrypoint (/input → route → /output)
│   ├── router/
│   │   ├── base.py                # Router ABC + RoutingTrace
│   │   ├── hybrid.py              # main orchestration class
│   │   └── strategies.py          # answer extractors (numeric, MCQ, JSON, ...)
│   ├── backends/
│   │   ├── base.py                # Backend ABC + GenerationResult
│   │   ├── local.py               # HF Transformers wrapper with logprobs
│   │   ├── fireworks.py           # OpenAI-compatible Fireworks client + retry + Mock
│   │   ├── harness_remote.py      # routes escalations across ALLOWED_MODELS
│   │   └── model_selector.py      # picks cheapest-capable model per task (Gemma-preferred)
│   ├── classifiers/
│   │   ├── heuristic.py           # task type + difficulty (regex)
│   │   └── confidence.py          # logprob + self-consistency assessment
│   ├── escalation/
│   │   └── policies.py            # Threshold / AlwaysLocal / AlwaysRemote
│   └── utils/
│       ├── tokens.py              # tiktoken counts (cost estimation)
│       └── metrics.py             # RunMetrics + leaderboard summary
├── eval/
│   ├── harness.py                 # JSONL in → JSONL out, prints metrics
│   ├── scorer.py                  # numeric + text + substring matching
│   └── tasks/
│       └── sample.jsonl           # 15-task dev set (math/ext/qa/cls/reasoning/code)
├── scripts/
│   ├── bench_local_models.py      # Pareto: accuracy / tokens / latency per model
│   └── calibrate_thresholds.py    # grid-search thresholds against dev set
├── tests/
│   └── test_router.py             # 8 smoke tests, all green
├── Dockerfile                     # Python 3.11 base, pre-downloads local model
├── requirements.txt
├── .env.example                   # all router knobs
└── README.md
```

## The harness contract

The container implements the Track 1 contract exactly:

- Reads `/input/tasks.json` — `[{"task_id", "prompt"}, ...]`
- Writes `/output/results.json` — `[{"task_id", "answer"}, ...]`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` from the
  environment at runtime (injected by the harness — never hardcoded, no bundled `.env`)
- All remote inference routes through `FIREWORKS_BASE_URL`; local inference is free
- Exits 0 on success; time-budgeted under the 10-min cap with crash-safe incremental writes

Model IDs are read from `ALLOWED_MODELS` at runtime, so the router is model-agnostic:
it works with whatever the harness provides. `ModelSelector` prefers Gemma for general
tasks (strong generalist + the *Best Use of Gemma* bonus) and code-specialized models
for code tasks.

## Validated across all 8 categories

End-to-end run through the container (`/input` → route → `/output`), one task per
capability category — all clean, all extractable:

| # | Category | Prompt (abbrev.) | Answer | Path |
|---|----------|------------------|--------|------|
| 1 | Factual | capital of Japan | `Tokyo` | escalated |
| 2 | Math | 15% of 240 | `36` | escalated |
| 3 | Sentiment | "exceeded expectations" | `Positive` | local |
| 4 | Summarization | earnings passage | one-sentence summary | escalated |
| 5 | NER | "Satya Nadella … Microsoft … Dublin" | `Satya Nadella, Person …` | escalated |
| 6 | Code debug | `sum(nums)/len(nums) + 1` | bug removed | remote (always-escalate) |
| 7 | Logic | race ordering | `Carl` | local |
| 8 | Code gen | `is_palindrome(s)` | working function | remote (always-escalate) |

Robustness: transient remote failures (timeout / 429 / 5xx) retry with backoff, then
fall back to the local answer rather than emitting an empty string — a mediocre answer
beats a blank at the accuracy gate.

## Quick start (local dev)

```bash
git clone https://github.com/Makabeez/amd-router && cd amd-router
pip install -r requirements.txt
cp .env.example .env          # add your dev FIREWORKS_API_KEY + ALLOWED_MODELS
pytest tests/ -q              # 23 tests, all green

# Harness-style run: reads /input, writes /output
mkdir -p _run/input _run/output
echo '[{"task_id":"t1","prompt":"What is 15% of 240?"}]' > _run/input/tasks.json
INPUT_PATH=_run/input/tasks.json OUTPUT_PATH=_run/output/results.json \
  python -m src.harness_runner
cat _run/output/results.json
```

## Containerized submission

```bash
# Build for the linux/amd64 judging VM
docker buildx build --platform linux/amd64 \
  --tag ghcr.io/makabeez/amd-router:latest --push .

# Run exactly as the harness does
docker run --rm \
  -v $PWD/_run/input:/input:ro \
  -v $PWD/_run/output:/output \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL=$FIREWORKS_BASE_URL \
  -e ALLOWED_MODELS=$ALLOWED_MODELS \
  ghcr.io/makabeez/amd-router:latest
```

Image: **`ghcr.io/makabeez/amd-router:latest`** (public, linux/amd64, 3.47 GB — well
under the 10 GB cap; CPU-only torch keeps it lean). Local model (Qwen 0.5B) is pre-baked
so cold start clears the 60 s readiness cap.

## Routing primitives reference

| Primitive | File | Cost | When to use |
|-----------|------|------|-------------|
| Heuristic task classification | `classifiers/heuristic.py` | 0 tokens | always — preflight |
| Logprob confidence | `classifiers/confidence.py` | 0 (local) | single-shot tasks |
| Self-consistency voting | `classifiers/confidence.py` | n × local | reasoning / math |
| Threshold escalation | `escalation/policies.py` | 0 (decision) | postlocal hook |
| Verify-mode remote | `router/hybrid.py` | ~½ fresh remote | when local has a plausible draft |

## Status

- [x] Harness contract implemented (`/input` → route → `/output`, env-injected config)
- [x] 8 capability categories — all validated end-to-end (table above)
- [x] Per-task answer extraction (code fences, NER lists, summaries, numeric, MCQ, yes/no)
- [x] `ALLOWED_MODELS`-driven model selection, Gemma-preferred
- [x] Retry + local-fallback for transient remote failures
- [x] 23 tests green
- [x] Docker image public on GHCR, linux/amd64, 3.47 GB
- [x] Time-budget guard under the 10-min cap with crash-safe incremental writes

## License

MIT.

---

<div align="center">

*Routing intelligence wins, not raw compute.*

</div>
