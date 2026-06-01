"""Evaluate a trained GRPO LoRA checkpoint and analyze reflection emergence.

Run:
    python -m modules.05_grpo_lora.eval --ckpt checkpoints/grpo --n 200
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.datasets import load_gsm8k
from common.extract_answer import extract_answer, answers_equal
from common.prompts import R1_SYSTEM_PROMPT


REFLECTION_PATTERN = re.compile(
    r"\b(wait|let me reconsider|actually|hmm|on second thought|reconsider|"
    r"重新|等等|让我再想)\b",
    re.IGNORECASE,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", required=True, help="Path to LoRA checkpoint")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default="results/05_grpo_lora")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg = cfg["model"]

    print(f"[Eval] Loading base + LoRA from {args.ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(model_cfg["device"]).eval()
    model = PeftModel.from_pretrained(base, args.ckpt).eval()

    examples = list(load_gsm8k(split="test", n=args.n,
                               cache_dir=cfg["dataset"]["cache_dir"]))

    correct = 0
    reflections = 0
    response_lengths = []
    records = []

    for i, ex in enumerate(examples):
        # Build R1-style prompt
        msg = (
            f"{R1_SYSTEM_PROMPT}\n\n"
            f"User: {ex['question']}\n\nAssistant:"
        )
        ids = tokenizer(msg, return_tensors="pt").to(model_cfg["device"])
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=600, do_sample=False, temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

        pred = extract_answer(completion)
        ok = answers_equal(pred, ex["answer"])
        if ok:
            correct += 1

        n_reflect = len(REFLECTION_PATTERN.findall(completion))
        if n_reflect > 0:
            reflections += 1
        response_lengths.append(len(completion.split()))

        records.append({
            "question": ex["question"],
            "gold": ex["answer"],
            "pred": pred,
            "correct": ok,
            "reflection_count": n_reflect,
            "n_tokens": len(completion.split()),
            "completion": completion,
        })
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(examples)}  acc={correct/(i+1):.3f}  refl={reflections/(i+1):.3f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "ckpt": args.ckpt,
        "n_examples": len(examples),
        "accuracy": correct / len(examples),
        "reflection_rate": reflections / len(examples),
        "avg_length_tokens": sum(response_lengths) / len(response_lengths),
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "eval_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))

    md = (
        f"# Module 05 · GRPO LoRA Eval\n\n"
        f"Checkpoint: `{args.ckpt}`\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Accuracy | {summary['accuracy']:.3f} |\n"
        f"| Reflection rate | {summary['reflection_rate']:.3f} |\n"
        f"| Avg response length (tokens) | {summary['avg_length_tokens']:.1f} |\n"
    )
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
