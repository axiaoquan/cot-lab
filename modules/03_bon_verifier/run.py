"""Module 03 · Best-of-N + Verifier.

Two stages:
  1. Generate N candidates with the small policy model.
  2. Score each with a verifier (Ollama DeepSeek-R1-Distill-7B by default).
  3. Pick the highest-scoring candidate.

Run:
    ollama pull deepseek-r1:7b
    python -m modules.03_bon_verifier.run --n 100 --bon 16
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from common.models import load_model, generate
from common.datasets import load_gsm8k
from common.prompts import ZERO_SHOT_COT_TEMPLATE, build_prompt
from common.extract_answer import extract_answer, answers_equal
from common.eval import RunReport, ExampleRecord, save_report


VERIFIER_PROMPT = (
    "You are a careful math reasoning judge. Read the candidate solution to "
    "the problem and rate its correctness on a scale of 1 to 10.\n\n"
    "Problem: {question}\n\n"
    "Candidate solution:\n{candidate}\n\n"
    "Rate the solution. Output only a single integer between 1 and 10 on the "
    "last line, prefixed with `Score:`."
)


def parse_score(text: str) -> float:
    m = re.search(r"score\s*[:：]\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    return float(nums[-1]) if nums else 5.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--bon", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--verifier-backend", default="ollama", choices=["ollama", "transformers"])
    p.add_argument("--out", default="results/03_bon_verifier")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    # Policy model (the small one we're trying to improve).
    print("[Setup] Loading policy model...")
    policy = load_model(
        cfg["model"]["name"],
        backend="transformers",
        device=cfg["model"]["device"],
        torch_dtype=cfg["model"]["torch_dtype"],
    )

    # Verifier model.
    print(f"[Setup] Loading verifier ({args.verifier_backend})...")
    if args.verifier_backend == "ollama":
        verifier = load_model(cfg["verifier"]["ollama_model"], backend="ollama")
    else:
        verifier = load_model(
            cfg["verifier"]["fallback_hf_model"],
            backend="transformers",
            device=cfg["model"]["device"],
            torch_dtype=cfg["model"]["torch_dtype"],
        )

    examples = list(load_gsm8k(split="test", n=args.n,
                               cache_dir=cfg["dataset"]["cache_dir"]))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    correct = 0

    for i, ex in enumerate(examples):
        prompt = build_prompt(ZERO_SHOT_COT_TEMPLATE, ex["question"])

        candidates = generate(
            policy, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            n=args.bon,
        )

        # Score each candidate.
        scored = []
        for c in candidates:
            verifier_prompt = VERIFIER_PROMPT.format(question=ex["question"], candidate=c)
            v_out = generate(verifier, verifier_prompt, max_new_tokens=64,
                             temperature=0.0, do_sample=False, n=1)[0]
            scored.append((parse_score(v_out), c))

        scored.sort(key=lambda x: -x[0])
        best_score, best_completion = scored[0]
        pred = extract_answer(best_completion)
        ok = answers_equal(pred, ex["answer"])
        correct += int(ok)

        records.append(ExampleRecord(
            question=ex["question"], gold=ex["answer"], pred=pred,
            completion=best_completion, correct=ok, latency_s=0.0,
        ))
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(examples)}  acc-so-far: {correct/(i+1):.3f}")

    acc = correct / len(examples)
    report = RunReport(
        method=f"bon_verifier_N={args.bon}",
        model=policy.name, dataset="gsm8k",
        n=len(examples), accuracy=acc, avg_latency_s=0.0,
        records=records,
    )
    save_report(report, out_dir / f"bon_{args.bon}.json")

    summary = (
        f"# Module 03 · Best-of-N + Verifier\n\n"
        f"Policy: `{policy.name}` · Verifier: `{verifier.name}`\n"
        f"BoN={args.bon} · Dataset: GSM8K (n={args.n})\n\n"
        f"| Method | Accuracy |\n|---|---|\n"
        f"| BoN={args.bon} + Verifier | {acc:.3f} |\n"
    )
    (out_dir / "summary.md").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
