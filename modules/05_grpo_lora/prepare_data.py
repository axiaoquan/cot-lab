"""Build a HuggingFace dataset for GRPO training.

Each row has:
    prompt:  R1-style chat prompt with <think>/<answer> instructions
    answer:  ground-truth gold (string)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset

from common.datasets import load_gsm8k
from common.prompts import R1_SYSTEM_PROMPT, R1_USER_TEMPLATE


def build_chat_prompt(question: str) -> str:
    """Lightweight chat-template-agnostic prompt; trl will re-template if needed."""
    return (
        f"{R1_SYSTEM_PROMPT}\n\n"
        f"User: {R1_USER_TEMPLATE.format(question=question)}\n\n"
        f"Assistant:"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", default="data/grpo_gsm8k")
    return p.parse_args()


def main():
    args = parse_args()
    examples = list(load_gsm8k(split="train", n=args.n))

    rows = []
    for ex in examples:
        rows.append({
            "prompt": build_chat_prompt(ex["question"]),
            "answer": ex["answer"],
        })

    ds = Dataset.from_list(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"[Data] Wrote {len(ds)} rows to {out}")


if __name__ == "__main__":
    main()
