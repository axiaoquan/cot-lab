"""Smoke test: load model + run one math problem in both Direct and CoT modes.

Run:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from common.models import load_model, generate
from common.prompts import DIRECT_TEMPLATE, ZERO_SHOT_COT_TEMPLATE, build_prompt
from common.extract_answer import extract_answer


QUESTION = "If 3 pens cost 12 dollars and 2 notebooks cost 14 dollars, how much do 5 pens and 3 notebooks cost?"


def main():
    cfg = yaml.safe_load((Path(__file__).parent.parent / "configs/default.yaml").read_text())
    print("[Smoke] Loading model...")
    model = load_model(
        cfg["model"]["name"],
        backend="transformers",
        device=cfg["model"]["device"],
        torch_dtype=cfg["model"]["torch_dtype"],
    )
    print(f"[Smoke] Model on {model.device}")
    print(f"[Smoke] Question: {QUESTION}\n")

    print("--- Direct ---")
    out = generate(model, build_prompt(DIRECT_TEMPLATE, QUESTION),
                   max_new_tokens=128, temperature=0.0, do_sample=False)[0]
    print(out.strip())
    print(f"[Extracted] {extract_answer(out)}")

    print("\n--- Zero-shot CoT ---")
    out = generate(model, build_prompt(ZERO_SHOT_COT_TEMPLATE, QUESTION),
                   max_new_tokens=512, temperature=0.0, do_sample=False)[0]
    print(out.strip())
    print(f"[Extracted] {extract_answer(out)}")

    print("\n✅ Smoke test passed.")


if __name__ == "__main__":
    main()
