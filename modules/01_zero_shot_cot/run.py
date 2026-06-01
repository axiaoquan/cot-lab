"""Module 01 · Zero-shot CoT vs Direct.

Run:
    python -m modules.01_zero_shot_cot.run --n 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from common.models import load_model, generate
from common.datasets import load_gsm8k
from common.prompts import DIRECT_TEMPLATE, ZERO_SHOT_COT_TEMPLATE, build_prompt
from common.eval import evaluate, save_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=200, help="Number of examples")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 for greedy decoding (recommended for fair comparison)")
    p.add_argument("--out", default="results/01_zero_shot_cot")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    model = load_model(
        cfg["model"]["name"],
        backend="transformers",
        device=cfg["model"]["device"],
        torch_dtype=cfg["model"]["torch_dtype"],
    )

    examples = list(load_gsm8k(split="test", n=args.n,
                               cache_dir=cfg["dataset"]["cache_dir"]))

    def make_solver(template: str):
        def solver(question: str) -> str:
            prompt = build_prompt(template, question)
            outs = generate(
                model, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=(args.temperature > 0),
                n=1,
            )
            return outs[0]
        return solver

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Running Direct (no CoT)...")
    rep_direct = evaluate("direct", model.name, "gsm8k", examples,
                          solver=make_solver(DIRECT_TEMPLATE))
    save_report(rep_direct, out_dir / "direct.json")

    print("\n[2/2] Running Zero-shot CoT...")
    rep_cot = evaluate("zero_shot_cot", model.name, "gsm8k", examples,
                       solver=make_solver(ZERO_SHOT_COT_TEMPLATE))
    save_report(rep_cot, out_dir / "cot.json")

    # Summary
    summary = (
        f"# Module 01 · Zero-shot CoT vs Direct\n\n"
        f"Model: `{model.name}` · Dataset: GSM8K (n={args.n})\n\n"
        f"| Method | Accuracy | Avg Latency (s) |\n"
        f"|---|---|---|\n"
        f"| Direct | {rep_direct.accuracy:.3f} | {rep_direct.avg_latency_s:.2f} |\n"
        f"| Zero-shot CoT | {rep_cot.accuracy:.3f} | {rep_cot.avg_latency_s:.2f} |\n"
    )
    (out_dir / "summary.md").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
