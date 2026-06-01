# Module 04 · Tree of Thoughts (24-game)

> **Question**: For tasks that require **search** (not just step-by-step calculation), can a tree-structured exploration beat linear CoT?

## Paper

- Yao et al., _Tree of Thoughts: Deliberate Problem Solving with Large Language Models_, NeurIPS 2023.

## Why 24-game

- Single-CoT struggles (small models < 5% solve rate)
- The state space is small enough to search exhaustively if needed
- Easy to verify (run `eval()` on the expression)

## How to run

```bash
python -m modules.04_tree_of_thoughts.run --n 5 --beam 5 --depth 4
```

## Algorithm sketch

```
state := list of remaining numbers (start with 4 numbers)
expand(state):
    for each pair (a, b) in state:
        for op in {+, -, *, /}:
            propose new_state = state - {a, b} + {a op b}
            ask LLM: "is this useful?" (sure / likely / impossible)
            if not impossible: add to candidates
beam-search the proposal tree to depth 3 (4 nums -> 3 ops to reach 1 num)
return any leaf where remaining number == 24
```

## Expected pattern

For Qwen2.5-1.5B-Instruct on a small handcrafted 24-game set:
- Single CoT: ~3-10% solve rate
- ToT (beam=5, depth=3): ~30-50%

The win comes from **explicit branching + LLM-evaluated pruning**, not from raw compute.
