"""Module 04 · Tree of Thoughts on the 24-game.

Skeleton implementation; the LLM-evaluator + beam-search loop is left as a TODO
that should be filled in once Module 1-3 are working.

Run:
    python -m modules.04_tree_of_thoughts.run --n 5 --beam 5
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import List, Tuple

import yaml

from common.models import load_model, generate
from common.datasets import load_24_game


# ---------------------------------------------------------------------------
# Game logic (deterministic part — no LLM needed here)
# ---------------------------------------------------------------------------


def expand_states(state: List[float]) -> List[Tuple[List[float], str]]:
    """Given a list of numbers, return all (new_state, op_description) pairs
    achievable by combining two of them with +,-,*,/.

    Returns:
        list of (new_remaining_numbers, "a op b = c")
    """
    if len(state) < 2:
        return []
    out = []
    for i, j in itertools.combinations(range(len(state)), 2):
        a, b = state[i], state[j]
        rest = [state[k] for k in range(len(state)) if k != i and k != j]
        # Each unordered pair gives multiple ops; for non-commutative we try both orders.
        for x, y in [(a, b), (b, a)]:
            candidates = [
                (x + y, f"{x}+{y}={x+y}"),
                (x - y, f"{x}-{y}={x-y}"),
                (x * y, f"{x}*{y}={x*y}"),
            ]
            if y != 0:
                candidates.append((x / y, f"{x}/{y}={x/y}"))
            for new_val, descr in candidates:
                out.append((rest + [new_val], descr))
    return out


def is_solved(state: List[float], target: float = 24.0, tol: float = 1e-6) -> bool:
    return len(state) == 1 and abs(state[0] - target) < tol


# ---------------------------------------------------------------------------
# LLM evaluator (TODO: fill in)
# ---------------------------------------------------------------------------


EVAL_PROMPT = (
    "We are trying to make 24 from a list of numbers using +,-,*,/.\n"
    "Current numbers remaining: {state}\n"
    "Last operation taken: {op}\n"
    "Rate the prospect of reaching 24 from this state on a 3-point scale:\n"
    "  sure / likely / impossible\n"
    "Output only one of those words."
)


def llm_score(model, state, op_descr) -> float:
    """Map LLM's verdict to a numeric score. Higher is better."""
    prompt = EVAL_PROMPT.format(state=state, op=op_descr)
    text = generate(model, prompt, max_new_tokens=8, temperature=0.0, do_sample=False)[0]
    text = text.strip().lower()
    if "sure" in text:
        return 1.0
    if "likely" in text:
        return 0.5
    if "impossible" in text:
        return 0.0
    return 0.3  # uncertain default


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def tot_solve(model, numbers: List[int], beam: int = 5, depth: int = 3) -> Tuple[bool, List[str]]:
    """Returns (solved, trace_of_ops)."""
    # state = (remaining_numbers, list_of_ops_taken)
    frontier = [(list(map(float, numbers)), [])]
    for _step in range(depth):
        candidates = []
        for state, ops in frontier:
            for new_state, descr in expand_states(state):
                score = llm_score(model, new_state, descr)
                candidates.append((score, new_state, ops + [descr]))
        # Keep top-beam by score
        candidates.sort(key=lambda x: -x[0])
        frontier = [(s, o) for _, s, o in candidates[:beam]]
        # Early exit
        for s, o in frontier:
            if is_solved(s):
                return True, o
    return False, []


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--out", default="results/04_tree_of_thoughts")
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

    examples = list(load_24_game(n=args.n))
    solved = 0
    rows = []

    for i, ex in enumerate(examples):
        nums = ex["raw"]["numbers"]
        ok, trace = tot_solve(model, nums, beam=args.beam, depth=args.depth)
        solved += int(ok)
        rows.append((nums, ok, trace))
        print(f"[{i+1}/{len(examples)}] nums={nums} solved={ok}")
        if trace:
            for t in trace:
                print(f"    -> {t}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"# Module 04 · Tree of Thoughts (24-game)",
        "",
        f"Model: `{model.name}` · beam={args.beam} · depth={args.depth} · n={len(examples)}",
        "",
        f"**Solved: {solved}/{len(examples)} ({solved/len(examples)*100:.1f}%)**",
        "",
        "| nums | solved | trace |",
        "|---|---|---|",
    ]
    for nums, ok, tr in rows:
        summary_lines.append(f"| {nums} | {'✅' if ok else '❌'} | {' / '.join(tr) if tr else '—'} |")
    (out_dir / "summary.md").write_text("\n".join(summary_lines))
    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()
