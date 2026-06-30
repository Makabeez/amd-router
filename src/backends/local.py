"""Local model backend via HuggingFace Transformers.

Local tokens count as ZERO toward score, so we generate generously.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .base import Backend, GenerationResult


class LocalTransformersBackend(Backend):
    """Transformers-based local backend with logprob support.

    Optimized for small models (≤3B) on the standardized eval env.
    For inference at scale, swap in vLLM/llama.cpp via the same Backend interface.
    """

    is_remote = False

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_model_len: int = 4096,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"local:{model_id}"
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_model_len = max_model_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        return_logprobs: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=return_logprobs,
        )

        out = self.model.generate(**inputs, **gen_kwargs)
        seq = out.sequences[0]
        gen_ids = seq[input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        if stop:
            for s in stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
                    break

        logprobs: list[float] | None = None
        if return_logprobs and out.scores:
            logprobs = []
            for step, score in enumerate(out.scores):
                if step >= len(gen_ids):
                    break
                log_probs = torch.log_softmax(score[0], dim=-1)
                token_id = gen_ids[step].item()
                logprobs.append(log_probs[token_id].item())

        return GenerationResult(
            text=text,
            local_input_tokens=input_len,
            local_output_tokens=len(gen_ids),
            logprobs=logprobs,
            finish_reason="stop" if stop and any(s in text for s in stop) else "length",
        )

    @torch.inference_mode()
    def generate_n(
        self,
        prompt: str,
        n: int,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> list[GenerationResult]:
        """Batched n-sample generation for self-consistency.

        Much faster than n sequential calls when n > 1.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=max(temperature, 1e-5),
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        results: list[GenerationResult] = []
        for seq in out:
            gen_ids = seq[input_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(
                GenerationResult(
                    text=text,
                    local_input_tokens=input_len,
                    local_output_tokens=len(gen_ids),
                )
            )
        return results

    def perplexity(self, prompt: str, completion: str) -> float:
        """Score how 'natural' a completion is under this model. Useful for verification."""
        with torch.inference_mode():
            full = self.tokenizer(prompt + completion, return_tensors="pt").to(self.device)
            prompt_len = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            labels = full.input_ids.clone()
            labels[:, :prompt_len] = -100
            out = self.model(**full, labels=labels)
            return math.exp(out.loss.item())
