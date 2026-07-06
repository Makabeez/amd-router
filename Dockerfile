# AMD Track 1 — General-Purpose AI Agent (harness contract).
# Judging VM runs linux/amd64. Build with:
#   docker buildx build --platform linux/amd64 -t <registry>/amd-router:latest --push .
#
# Harness contract:
#   - reads /input/tasks.json, writes /output/results.json
#   - env injected at eval: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
#   - <=10 min runtime, ready <60s, <30s/request, image <=10GB compressed

FROM --platform=linux/amd64 python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src

# Pre-bake the free local model so cold start stays under the 60s readiness cap.
# Qwen 0.5B: fast, ~1GB, keeps the image well under 10GB. Override LOCAL_MODEL
# at eval only if the env allows a bigger local model and time budget permits.
ARG LOCAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ENV LOCAL_MODEL=${LOCAL_MODEL}
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('${LOCAL_MODEL}'); \
    AutoModelForCausalLM.from_pretrained('${LOCAL_MODEL}')" || true

# Harness runner is the entrypoint: reads /input, routes, writes /output, exits.
ENTRYPOINT ["python", "-m", "src.harness_runner"]
