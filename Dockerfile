# Containerized submission for AMD Track 1.
# Uses CPU base; swap to rocm/pytorch image for AMD GPU run.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY eval ./eval
COPY scripts ./scripts

# Pre-download local model into image for fast cold-start in the eval env.
# Override LOCAL_MODEL at build time: --build-arg LOCAL_MODEL=...
ARG LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
ENV LOCAL_MODEL=${LOCAL_MODEL}
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('${LOCAL_MODEL}'); \
    AutoModelForCausalLM.from_pretrained('${LOCAL_MODEL}')" || true

ENTRYPOINT ["python", "-m", "src.main"]
