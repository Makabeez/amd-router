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

![status](https://img.shields.io/badge/status-day_0_prep-yellow?style=for-the-badge)
![track](https://img.shields.io/badge/track-1_routing-ED1C24?style=for-the-badge)
![event](https://img.shields.io/badge/AMD-Hackathon_Act_II-000000?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge)

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
│   ├── main.py                    # containerized entrypoint (stdin → stdout JSONL)
│   ├── router/
│   │   ├── base.py                # Router ABC + RoutingTrace
│   │   ├── hybrid.py              # main orchestration class
│   │   └── strategies.py          # answer extractors (numeric, MCQ, JSON, ...)
│   ├── backends/
│   │   ├── base.py                # Backend ABC + GenerationResult
│   │   ├── local.py               # HF Transformers wrapper with logprobs
│   │   └── fireworks.py           # OpenAI-compatible Fireworks client + Mock
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

## Quick start

```bash
# 1. clone
git clone https://github.com/Makabeez/amd-router && cd amd-router

# 2. deps
pip install -r requirements.txt

# 3. configure
cp .env.example .env
# fill in FIREWORKS_API_KEY, FIREWORKS_MODEL, LOCAL_MODEL

# 4. smoke tests
pytest tests/ -v

# 5. mock dry-run
python -m eval.harness --tasks eval/tasks/sample.jsonl --mock

# 6. real run (one prompt per JSONL line)
echo '{"id":"q1","prompt":"What is 15% of 240?"}' | python -m src.main
```

## Containerized submission

```bash
docker build -t amd-router .
docker run --rm -i \
    -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
    amd-router < tasks.jsonl > predictions.jsonl
```

## Routing primitives reference

| Primitive | File | Cost | When to use |
|-----------|------|------|-------------|
| Heuristic task classification | `classifiers/heuristic.py` | 0 tokens | always — preflight |
| Logprob confidence | `classifiers/confidence.py` | 0 (local) | single-shot tasks |
| Self-consistency voting | `classifiers/confidence.py` | n × local | reasoning / math |
| Threshold escalation | `escalation/policies.py` | 0 (decision) | postlocal hook |
| Verify-mode remote | `router/hybrid.py` | ~½ fresh remote | when local has a plausible draft |

## Day 0 prep checklist (Jun 30 → Jul 6)

- [x] Pluggable router architecture
- [x] Local + remote + mock backends
- [x] Heuristic classifier
- [x] Confidence assessment (logprob + self-consistency)
- [x] Escalation policy interface + threshold impl
- [x] Eval harness + scorer + sample tasks
- [x] Containerized entrypoint
- [x] 8 smoke tests passing
- [ ] Register AMD AI Developer Program by **Jul 2** (credits cutoff)
- [ ] Bench candidate small models on dev set (`scripts/bench_local_models.py`)
- [ ] Pre-cache 2–3 finalist local models in Docker image
- [ ] Confirm Fireworks model lineup + token pricing
- [ ] Read standardized eval env spec on kickoff (Jul 6 18:00 CEST)

## Kickoff day plan (Jul 6, 18:00 CEST)

| Time | Action |
|------|--------|
| 18:00 | Tasks revealed — read all instructions before touching code |
| 18:30 | Lock answer extractor (numeric / MCQ / JSON / ...) for the task format |
| 19:00 | First baseline: AlwaysRemote → upper-bound accuracy + token cost |
| 20:00 | Second baseline: AlwaysLocal → lower-bound accuracy, zero cost |
| 21:00 | First hybrid run with default thresholds |
| 22:00 | Calibration grid search on held-out subset |
| Day 2+ | Iterate: prompt engineering, extractor tuning, threshold refinement |

## License

MIT.

---

<div align="center">

*Routing intelligence wins, not raw compute.*

</div>
