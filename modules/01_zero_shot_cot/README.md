# Module 01 · Zero-shot CoT

> **Question**: Does the magic phrase _"Let's think step by step"_ actually help small models?
>
> **Method**: Run the same N problems with two prompts (`Direct` vs `CoT`), compare accuracy and avg latency.

## Papers

- Wei et al., _Chain-of-Thought Prompting Elicits Reasoning in Large Language Models_, NeurIPS 2022.
- Kojima et al., _Large Language Models are Zero-Shot Reasoners_, NeurIPS 2022.

## How to run

```bash
# Quick smoke test (10 examples, ~30s)
python -m modules.01_zero_shot_cot.run --n 10

# Full run (200 examples)
python -m modules.01_zero_shot_cot.run --n 200
```

Outputs land in `results/01_zero_shot_cot/`:
- `direct.json` — per-example records for direct prompting
- `cot.json` — per-example records for CoT prompting
- `summary.md` — auto-generated comparison table

## Expected pattern

For Qwen2.5-1.5B-Instruct on GSM8K:

| Method | Accuracy | Avg Latency |
|---|---|---|
| Direct | ~25-35% | 1× |
| Zero-shot CoT | ~50-65% | 3-4× |

## Key observation to watch for

- CoT helps on multi-step problems but **may hurt on trivial 1-step problems** (the "rambling penalty").
- Length of CoT correlates with task hardness more than with correctness.
