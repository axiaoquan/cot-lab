"""Aggregate per-module summary.md + JSON into a single master comparison table.

Run:
    python scripts/plot_master_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS = Path(__file__).parent.parent / "results"


def collect():
    rows = []
    for module_dir in sorted(RESULTS.iterdir()):
        if not module_dir.is_dir():
            continue
        for json_file in module_dir.glob("*.json"):
            try:
                d = json.loads(json_file.read_text())
            except Exception:
                continue
            summary = d.get("summary") or d
            if "accuracy" in summary:
                rows.append({
                    "module": module_dir.name,
                    "method": summary.get("method", json_file.stem),
                    "accuracy": summary["accuracy"],
                    "avg_latency_s": summary.get("avg_latency_s", 0.0),
                    "n": summary.get("n", 0),
                })
    return pd.DataFrame(rows)


def main():
    df = collect()
    if df.empty:
        print("No results found yet. Run modules first.")
        return
    print(df.to_string(index=False))

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    df_sorted = df.sort_values("accuracy", ascending=False)
    ax.barh(df_sorted["method"], df_sorted["accuracy"])
    ax.set_xlabel("Accuracy")
    ax.set_title("CoT-Lab · Master Comparison")
    plt.tight_layout()
    out = RESULTS / "master_table.png"
    plt.savefig(out, dpi=150)
    print(f"[Plot] Saved {out}")


if __name__ == "__main__":
    main()
