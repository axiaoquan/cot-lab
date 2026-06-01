# Module 02 · Self-Consistency

> **Question**: If we sample N reasoning paths and majority-vote on the final answer, how much accuracy do we gain?

## Paper

- Wang et al., _Self-Consistency Improves Chain of Thought Reasoning in Language Models_, ICLR 2023.

## How to run

```bash
# Sweep N=1..32 to plot the scaling curve
python -m modules.02_self_consistency.run --n 100 --samples 1,2,4,8,16,32
```

Outputs:
- `samples_<N>.json` — per-example records at each N
- `scaling.png` — accuracy vs N (log-x)
- `summary.md`

## Expected pattern

Accuracy increases roughly logarithmically with N, then saturates around N=32-64.

## Key observation

- **Temperature matters**: too low → all samples identical → no benefit. Recommended T=0.7-0.9.
- **The "right answer" wins because errors are diverse** while correct paths converge.
