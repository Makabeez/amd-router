# Containerized submission for AMD Track 1.
# CPU base for dev/testing; the AMD eval env provides the GPU at scoring time.
# For a ROCm build, swap the base image to rocm/pytorch and drop torch from
# requirements (the ROCm image ships it).

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

# ca-certificates ships in python:3.11-slim; no apt-get needed.

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY eval ./eval
COPY scripts ./scripts

# Pre-download BOTH candidate local models. Qwen 0.5B (57.5%, fast dev fallback)
# and Llama 3.2 3B (75%, scoring default on GPU eval env).
#
# Llama is gated. Pass the HF token as a BuildKit SECRET (never persisted in a
# layer) so the image contains model weights but NOT the credential:
#
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=hf_token,src=$HOME/.cache/huggingface/token \
#     -t amd-router .

RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')" || true

RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python -c "import os; from huggingface_hub import login; \
    tok=os.environ.get('HF_TOKEN'); login(token=tok) if tok else None; \
    from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct'); \
    AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')" || \
    echo "Llama 3B pre-download skipped (no secret or gated) - will fetch at runtime"

# Scoring-time default. Override at run: -e LOCAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ENV LOCAL_MODEL=meta-llama/Llama-3.2-3B-Instruct

ENTRYPOINT ["python", "-m", "src.main"]
