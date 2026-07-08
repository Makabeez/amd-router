# AMD Track 1 — General-Purpose AI Agent (harness contract).
# Judging VM runs linux/amd64. Build with:
#   docker buildx build --platform linux/amd64 -t <registry>/amd-router:latest --push .
#
# Harness contract:
#   - reads /input/tasks.json, writes /output/results.json
#   - env injected at eval: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
#   - <=10 min runtime, ready <60s, <30s/request, image <=10GB compressed

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

# Install CPU-only torch FIRST from the dedicated CPU index. Avoids ~2-3GB of
# CUDA libraries we don't use (local model runs on CPU in-container). Keeps the
# image well under the 10GB cap.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3,<3.0"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src

ARG LOCAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ENV LOCAL_MODEL=${LOCAL_MODEL}
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('${LOCAL_MODEL}'); \
    AutoModelForCausalLM.from_pretrained('${LOCAL_MODEL}')" || true

ENTRYPOINT ["python", "-m", "src.harness_runner"]
