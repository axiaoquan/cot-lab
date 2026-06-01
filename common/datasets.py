"""Dataset loaders. All return iterables of dicts with at least:
    {"question": str, "answer": str, "raw": dict}
where `answer` is a normalized string (final numeric answer for GSM8K).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def load_gsm8k(split: str = "test", n: int | None = None, cache_dir: str | None = None):
    """Yields normalized GSM8K examples.

    GSM8K answer field looks like:
        "Janet sells ... #### 18"
    We extract the integer/float after `####`.
    """
    # Lazy import so unit tests that don't need the network can still import this module.
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    for ex in ds:
        ans_raw = ex["answer"]
        m = re.search(r"####\s*([\-\d\.,]+)", ans_raw)
        gold = m.group(1).replace(",", "").strip() if m else ans_raw.strip()
        yield {
            "question": ex["question"],
            "answer": gold,
            "raw": ex,
        }


# ---------------------------------------------------------------------------
# 24-game (used by Module 04 / Tree of Thoughts)
# ---------------------------------------------------------------------------


def load_24_game(n: int | None = None) -> Iterator[Dict[str, Any]]:
    """Returns puzzles where four numbers must reach 24 with +,-,*,/, and parens.

    For now we ship a small handcrafted set; a full version can pull from
    https://github.com/princeton-nlp/tree-of-thought-llm
    """
    examples = [
        {"numbers": [1, 1, 4, 6], "answer": "(1 + 1 * 4) * 6"},
        {"numbers": [2, 3, 8, 12], "answer": "(12 - 8) * (3 + 3)"},  # placeholder
        {"numbers": [4, 4, 6, 8], "answer": "(8 - 4) * 6"},
        {"numbers": [3, 5, 7, 8], "answer": "(8 - 5 + 3) * 7"},  # placeholder
        {"numbers": [1, 5, 5, 5], "answer": "5 * 5 - 1"},  # placeholder
    ]
    if n is not None:
        examples = examples[:n]
    for ex in examples:
        q = f"Make 24 using each of the numbers exactly once with +,-,*,/ and parentheses: {ex['numbers']}"
        yield {"question": q, "answer": ex["answer"], "raw": ex}


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def get_loader(name: str):
    name = name.lower()
    if name == "gsm8k":
        return load_gsm8k
    if name in ("24", "24game", "24-game"):
        return load_24_game
    raise ValueError(f"Unknown dataset: {name}")
