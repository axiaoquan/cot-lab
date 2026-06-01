# Paper Notes

Short, opinionated notes on each paper underpinning a module. Update as you read.

---

## CoT Prompting (Wei et al., NeurIPS 2022)

> **TL;DR**: Few-shot prompting with reasoning steps unlocks reasoning that single-step prompts can't elicit.

- The effect is _emergent_: only large models (>~60B at the time) clearly benefit.
- Manual exemplar quality > exemplar quantity.
- Confirmed CoT works across arithmetic, commonsense, symbolic.

**Why it matters for us**: Module 01 baseline.

---

## Zero-shot CoT (Kojima et al., NeurIPS 2022)

> **TL;DR**: A single phrase ("Let's think step by step") matches few-shot CoT on many tasks.

- Implies that CoT capability is _learned during pretraining_, not by exemplars.
- Decoding is two-stage: 1) generate reasoning, 2) extract answer with another prompt.

**Why it matters for us**: This is the actual prompt used in Module 01 / 02 / 03.

---

## Self-Consistency (Wang et al., ICLR 2023)

> **TL;DR**: Sample N reasoning paths, majority-vote the final answer.

- Works because there are usually many valid paths to the right answer, but
  many distinct paths to wrong ones — voting amplifies correct signals.
- Saturates around N=40-64.

**Why it matters for us**: Module 02. Strong baseline before introducing learned verifiers.

---

## Process Reward Model (Lightman et al., 2024)

> **TL;DR**: Score every step of CoT, not just the final answer. PRM > ORM.

- 800K human-annotated step labels (PRM800K).
- Model size 175B base, but the methodology transfers down.

**Why it matters for us**: Module 03's verifier. We use a strong off-the-shelf
reasoner as a stand-in for a "trained PRM".

---

## Tree of Thoughts (Yao et al., NeurIPS 2023)

> **TL;DR**: Frame reasoning as a tree; expand, evaluate, prune; BFS or DFS.

- 24-game: GPT-4 solves 4% with CoT, 74% with ToT.
- Requires a self-evaluation prompt that the model is confident with.

**Why it matters for us**: Module 04. Demonstrates **search > linear CoT** for
problems that require lookahead.

---

## DeepSeekMath / GRPO (Shao et al., 2024)

> **TL;DR**: Replace value model with group-relative advantage. Save ~50% memory.

- For each prompt q, sample G outputs, normalize within the group.
- Math reward is rule-based (compare against gold). No reward model trained.

**Why it matters for us**: Module 05's training loss. The reason small-Mac training is even feasible.

---

## DeepSeek-R1 (DeepSeek-AI, 2025) ⭐

> **TL;DR**: Pure RL with rule-based rewards on a base model produces emergent
> long-chain reasoning + self-reflection ("aha moments").

- R1-Zero: no SFT, just RL → already strong on math/code.
- R1: 4-stage pipeline: cold-start SFT → R1-Zero RL → rejection sampling → final RL.
- Distill: R1 outputs are SFT data for smaller (1.5B-70B) open-source models.

**Why it matters for us**: This is the **direct inspiration** for Module 05's
training recipe. Our setup is a small-scale, Mac-friendly version of R1-Zero.
