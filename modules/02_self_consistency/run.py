"""Module 02 · Self-Consistency.

Run:
    python -m modules.02_self_consistency.run --n 100 --samples 1,4,16,32
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from common.models import load_model, generate
from common.datasets import load_gsm8k
from common.prompts import ZERO_SHOT_COT_TEMPLATE, build_prompt
from common.extract_answer import extract_answer, answers_equal
from common.eval import RunReport, ExampleRecord, save_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--samples", default="1,4,16,32",
                   help="Comma-separated list of N values to sweep")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", default="results/02_self_consistency")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plotting (e.g. on a headless machine)")
    return p.parse_args()


def majority_vote(answers):
    """Return the (most common answer, vote count). Filters out None."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0
    counter = Counter(valid)
    ans, cnt = counter.most_common(1)[0]
    return ans, cnt


def plot_scaling(table, out_path):
    """Plot accuracy vs N with log-x scale."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not installed; skipping plot.")
        return

    Ns = [N for N, _ in table]
    accs = [acc for _, acc in table]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ns, accs, marker="o", linewidth=2, markersize=8, color="#3b82f6")
    ax.set_xscale("log")
    ax.set_xlabel("Samples (N)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Self-Consistency · Accuracy vs N", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    # Annotate each point
    for n, acc in zip(Ns, accs):
        ax.annotate(f"{acc:.3f}", (n, acc), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


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
    sample_sizes = sorted(int(s) for s in args.samples.split(","))
    max_n = max(sample_sizes)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: for each example, generate max_n completions ONCE.
    print(f"[1/2] Generating {max_n} samples per example...")
    all_completions = []  # List[ List[str] ] of length len(examples)
    for i, ex in enumerate(examples):
        prompt = build_prompt(ZERO_SHOT_COT_TEMPLATE, ex["question"])
        comps = generate(
            model, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            n=max_n,
        )
        all_completions.append(comps)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(examples)}")

    # Step 2: for each N in sample_sizes, compute accuracy by majority voting on first N.
    print(f"\n[2/2] Computing accuracy for N in {sample_sizes}...")
    table = []
    for N in sample_sizes:
        correct = 0
        records = []
        for ex, comps in zip(examples, all_completions):
            sub = comps[:N]
            answers = [extract_answer(c) for c in sub]
            voted, _ = majority_vote(answers)
            ok = answers_equal(voted, ex["answer"])
            correct += int(ok)
            records.append(ExampleRecord(
                question=ex["question"], gold=ex["answer"], pred=voted,
                completion="\n---\n".join(sub),
                correct=ok, latency_s=0.0,
            ))
        acc = correct / len(examples)
        table.append((N, acc))
        report = RunReport(
            method=f"self_consistency_N={N}",
            model=model.name, dataset="gsm8k",
            n=len(examples), accuracy=acc, avg_latency_s=0.0,
            records=records,
        )
        save_report(report, out_dir / f"samples_{N}.json")

    # Plot
    if not args.no_plot:
        plot_scaling(table, out_dir / "scaling.png")

    # Summary
    rows = "\n".join(f"| N={N} | {acc:.3f} |" for N, acc in table)
    summary = (
        f"# Module 02 · Self-Consistency\n\n"
        f"Model: `{model.name}` · Dataset: GSM8K (n={args.n}) · T={args.temperature}\n\n"
        f"| Samples | Accuracy |\n|---|---|\n{rows}\n\n"
        f"![scaling curve](scaling.png)\n"
    )
    (out_dir / "summary.md").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
