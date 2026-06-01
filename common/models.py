"""Unified model loader supporting multiple backends.

Backends:
  - "transformers" (default): HuggingFace transformers on MPS / CUDA / CPU.
  - "ollama"                : Local Ollama HTTP server (great for verifier model).

Usage:
    from common.models import load_model, generate

    model = load_model("Qwen/Qwen2.5-1.5B-Instruct", backend="transformers")
    out = generate(model, "What is 2+2?", max_new_tokens=64)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import torch


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelHandle:
    name: str
    backend: str           # "transformers" | "ollama"
    obj: Any               # underlying handle (HF model, or just str for ollama)
    tokenizer: Any = None  # only for transformers backend
    device: str = "mps"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_model(
    name: str,
    backend: str = "transformers",
    device: str = "mps",
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
) -> ModelHandle:
    """Load a model with a unified interface."""
    if backend == "transformers":
        return _load_transformers(name, device, torch_dtype, trust_remote_code)
    if backend == "ollama":
        return _load_ollama(name)
    raise ValueError(f"Unknown backend: {backend}")


def _load_transformers(name, device, torch_dtype, trust_remote_code) -> ModelHandle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(torch_dtype, torch.float32)

    # Resolve effective device: fall back gracefully.
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"[Model] Loading {name} on {device} ({torch_dtype})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, trust_remote_code=trust_remote_code
    )
    model = model.to(device).eval()
    elapsed = time.time() - t0
    print(f"[Model] Loaded in {elapsed:.1f}s")

    return ModelHandle(name=name, backend="transformers", obj=model, tokenizer=tokenizer, device=device)


def _load_ollama(name: str) -> ModelHandle:
    """Ollama is a local HTTP server; we just record the model name."""
    # TODO: optionally ping http://localhost:11434/ to verify the server is up.
    print(f"[Model] Using Ollama backend with model={name} (HTTP @ localhost:11434)")
    return ModelHandle(name=name, backend="ollama", obj=name, tokenizer=None, device="local")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    handle: ModelHandle,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
    n: int = 1,
) -> List[str]:
    """Generate `n` completions for a single prompt. Always returns a list of strings."""
    if handle.backend == "transformers":
        return _generate_transformers(handle, prompt, max_new_tokens, temperature, top_p, do_sample, n)
    if handle.backend == "ollama":
        return _generate_ollama(handle, prompt, max_new_tokens, temperature, top_p, n)
    raise ValueError(f"Unknown backend: {handle.backend}")


def _generate_transformers(handle, prompt, max_new_tokens, temperature, top_p, do_sample, n):
    tokenizer = handle.tokenizer
    model = handle.obj

    # Use chat template if model has one (Qwen / Llama-Instruct etc.)
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(handle.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # Strip the prompt tokens off each completion.
    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    for i in range(n):
        gen_ids = out_ids[i][prompt_len:]
        completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return completions


def _generate_ollama(handle, prompt, max_new_tokens, temperature, top_p, n):
    import requests
    url = "http://localhost:11434/api/generate"
    completions = []
    for _ in range(n):
        resp = requests.post(
            url,
            json={
                "model": handle.obj,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            },
            timeout=300,
        )
        resp.raise_for_status()
        completions.append(resp.json().get("response", ""))
    return completions
