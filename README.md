<div align="center">

<img src="assets/banner.svg" alt="AMD Router" width="800">

**Local-first routing. Escalate only when the small model says "I don't know."**

![accuracy](https://img.shields.io/badge/eval-15%2F15_auto-success?style=for-the-badge)
![tokens](https://img.shields.io/badge/remote_tokens-3009_(--34%25)-ED1C24?style=for-the-badge)
![track](https://img.shields.io/badge/track-1_routing-000000?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge)
![image](https://img.shields.io/badge/image-ghcr.io%2Fmakabeez%2Famd--router-2088FF?style=for-the-badge&logo=github)

</div>

> Track 1 has an **80% accuracy gate**. Below 16/19 you don't appear on the leaderboard at all — token count is the tiebreaker *among teams that already pass*. So the goal is not "minimize remote calls." It is **clear the gate, then minimize remote calls.** Local tokens are free; we burn them to earn the right not to make a network call.

**Result on a 19-task eval mirroring the harness categories: same accuracy as always-remote, 34% fewer remote tokens.**

| Policy | Accuracy (auto-graded) | Remote tokens | Escalated | Clears gate |
|---|---|---|---|---|
| Local only (`always_local`) | 8/15 | 0 | 0/19 | ❌ |
| Remote only (`always_remote`) | 15/15 | 4526 | 19/19 | ✅ |
| **Hybrid (shipped)** | **15/15** | **3009** | **13/19** | ✅ |

Six of nineteen tasks never touch the network. The four free-form tasks (2× summarization, 2× NER) are graded by eye, not by the auto-scorer.

---

## How it routes

```
                            ┌─────────────────────────┐
                            │   Heuristic Classifier  │   (free, regex)
                            │  task_type + difficulty │
                            └──────────┬──────────────┘
                                       │
                            ┌──────────▼──────────────┐
                            │    Escalation Policy    │
                            │  preflight: skip local? │
                            └────┬───────────────┬────┘
                                 │ no            │ yes
                  ┌──────────────▼──┐            │
                  │ Local Backend   │            │
                  │ Qwen2.5-0.5B    │            │
                  │ + logprobs      │            │
                  │ + stop seqs     │            │
                  └────────┬────────┘            │
                           │                     │
                  ┌────────▼────────┐            │
                  │   Confidence    │            │
                  │  exp(mean lp)   │            │
                  └────────┬────────┘            │
                           │                     │
                  ┌────────▼──────────┐          │
                  │   Escalation      │          │
                  │ postlocal: send?  │          │
                  └───┬───────────┬───┘          │
                   no │       yes │              │
                      │           └──────┬───────┘
                      │                  ▼
                      │       ┌────────────────────┐
                      │       │   Fireworks API    │  ← the only paid tokens
                      │       │  model fallback on │
                      │       │  404 / undeployed  │
                      │       └─────────┬──────────┘
                      │                 │
                      └────────┬────────┘
                               ▼
                       ┌───────────────┐
                       │ Final answer  │
                       │ + RoutingTrace│
                       └───────────────┘
```

Confidence per task type has its own threshold — math demands 0.80, classification 0.55 —
because a wrong number is worse than a clumsy label.

## What measurement changed

The scaffold shipped with three assumptions. Two were wrong, and the eval caught them.

**1. "Code is locally unreliable, always escalate."**
The original `ThresholdPolicy` force-escalated every code task on preflight — the local
model never even tried. Measured: Qwen2.5-0.5B passes 3/3 code tasks locally, at
confidence **0.90 / 0.91 / 0.77**. Two stay local; only the third escalates.
**Deleting that one assumption saved 691 remote tokens at zero accuracy cost.**

**2. "Self-consistency is the cheapest confidence signal. Use it."**
It votes on canonicalized answer strings. On free-form output — summaries, NER lists,
math with reasoning — three samples at temperature 0.7 never agree, so agreement pins at
`1/n`. Since `score = 0.7·agreement + 0.3·logprob`, enabling it *drove confidence down
everywhere* and escalated tasks that were previously correct and local. It also took
runtime from 128s to 464s on CPU. **Disabled** (`N_SAMPLES=0`). It works only where the
raw text is the answer (`qa_03`: agreement 1.0). Fixing it properly means feeding the
per-task extractors into `assess()` — future work, not hackathon work.

**3. "Verify-mode is 30-60% cheaper than fresh remote."**
In practice the guard rejects it every time: attaching a 512-token local draft inflates a
109-char prompt to 2742 chars. It never fires. The guard is correct; the premise was not.

**4. The local model was reciting its own training data.**
Qwen finishes the answer, then keeps generating a fresh conversation turn —
`"You are an AI assistant that helps people find information."` — appended straight into
the summary. `local.py` supported `stop` sequences; the router never passed any. Now it
does. Zero token cost, pure accuracy.

## The harness contract

- Reads `/input/tasks.json` — `[{"task_id", "prompt"}, ...]`
- Writes `/output/results.json` — `[{"task_id", "answer"}, ...]`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` from the environment
  at runtime — never hardcoded, no bundled `.env`
- All remote inference routes through `FIREWORKS_BASE_URL`; local inference is free
- Exits 0 on success; time-budgeted under the 10-min cap with crash-safe incremental writes

**Model IDs are read from `ALLOWED_MODELS` at runtime.** Bare IDs (`minimax-m3`) and full
paths (`accounts/fireworks/models/minimax-m3`) both work — the client normalizes.

**Undeployed models don't kill the run.** Gemma on Fireworks is on-demand: a 404 means
"not deployed," not "banned." `HarnessRemoteBackend` walks the allowed list, marks any
model returning 400/404 as dead, and retries the next one. A single undeployed model would
otherwise collapse every escalation to the 0.5B fallback and drop you below the gate.

**Reasoning models return `reasoning_content`, not `content`.** When `minimax-m3` runs out
of tokens mid-think there is no `content` field at all. The client falls back to the
reasoning trace rather than silently writing `""`.

## Reproduce the numbers

```bash
git clone https://github.com/Makabeez/amd-router && cd amd-router
export FIREWORKS_API_KEY=...      # your key
export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
export ALLOWED_MODELS=minimax-m3,kimi-k2p6,glm-5p2

docker build --platform linux/amd64 -t amd-router:eval .

for p in always_local always_remote threshold; do
  docker run --rm --platform linux/amd64 --network host \
    -e ESCALATION_POLICY=$p -e ALLOWED_MODELS -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL \
    -v $PWD/eval/in19:/input:ro -v $PWD/eval/output:/output \
    amd-router:eval 2>&1 | tee eval/run_$p.log
  python3 eval/score.py eval/output/results.json
  grep -oP 'remote_tokens=\K\d+' eval/run_$p.log | paste -sd+ | bc
done
```

Every routing decision prints to stderr:

```
[code_01] backend=local:Qwen/Qwen2.5-0.5B-Instruct remote_tokens=0
    classified: type=TaskType.CODE difficulty=0.72
    preflight: escalate=False (preflight: local first)
    local: 164 tokens, mean_logprob=-0.1007
    confidence: score=0.90
    postlocal: escalate=False (postlocal: confident (0.90 >= 0.80))
```

## Containerized submission

```bash
docker run --rm \
  -v $PWD/_run/input:/input:ro -v $PWD/_run/output:/output \
  -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
  ghcr.io/makabeez/amd-router:v0.12
```

Image: **`ghcr.io/makabeez/amd-router:v0.12`** (also `:latest`)
Digest: `sha256:5837f24730f43d1767408e2e91e83f206dd83948c78e20bab0b3e95fcce2e407`
Public, linux/amd64, 3.47 GB — under the 10 GB cap. Qwen2.5-0.5B is pre-baked into the
image so cold start clears the 60s readiness cap. Verified by deleting the local image,
pulling from GHCR, and re-running the full eval: 15/15, 3009 tokens.

## Router knobs

All optional; defaults are the shipped configuration.

| Env var | Default | Effect |
|---|---|---|
| `ESCALATION_POLICY` | `threshold` | `always_local` / `always_remote` for baselines |
| `ALWAYS_ESCALATE` | *(empty)* | comma list of task types to skip local entirely |
| `N_SAMPLES` | `0` | self-consistency samples — see "What measurement changed" |
| `MIN_CONFIDENCE` | `0.65` | global floor; per-task thresholds override |
| `LOCAL_MAX_TOKENS` | `512` | |
| `REMOTE_MAX_TOKENS` | `1024` | raise for reasoning models that think before answering |
| `PREFER_GEMMA` | *(unset)* | opt in to Gemma-first selection (requires deploying it) |

## Routing primitives

| Primitive | File | Cost | Notes |
|---|---|---|---|
| Heuristic task classification | `classifiers/heuristic.py` | 0 tokens | regex, preflight |
| Logprob confidence | `classifiers/confidence.py` | 0 (local) | `exp(mean logprob)` |
| Per-task-type thresholds | `escalation/policies.py` | 0 | the routing decision |
| Model fallback on 404 | `backends/harness_remote.py` | 0 | survives undeployed models |
| Per-task answer extraction | `router/strategies.py` | 0 | code fences, NER, numeric |
| Local stop sequences | `router/hybrid.py` | 0 | prevents training-data recitation |

## Status

- [x] Harness contract implemented (`/input` → route → `/output`, env-injected config)
- [x] 19-task eval across all 8 categories; hybrid clears the 80% gate
- [x] 34% fewer remote tokens than always-remote at identical accuracy
- [x] `ALLOWED_MODELS`-driven selection; bare and full model IDs both accepted
- [x] 404 / undeployed-model fallback; reasoning-model content extraction
- [x] Retry + local-fallback for transient remote failures
- [x] Docker image public on GHCR, pull-verified end-to-end
- [x] Time-budget guard under the 10-min cap with crash-safe incremental writes
- [ ] Self-consistency reinstated with per-task extractors (future work)

## License

MIT.

---

<div align="center">

*Routing intelligence wins, not raw compute.*

</div>
