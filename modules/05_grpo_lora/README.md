# Module 05 · GRPO LoRA Training (R1-Zero mini)

> **Question**: Can we use **rule-based rewards + GRPO** to teach a small model to actually _think_ — i.e. generate `<think>...</think>` reasoning and self-reflect?

This is the **training** module. It's the most interesting and the most resource-intensive.

## Papers

- Shao et al., _DeepSeekMath_, 2024 (introduces GRPO).
- DeepSeek-AI, _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL_, 2025.

## How it works

```
For each prompt q:
    Sample G completions {o_1, ..., o_G} from the policy
    Compute reward r_i for each completion (rule-based)
    Compute advantage A_i = (r_i - mean(r)) / std(r)     ← group-relative, no value model!
    Update policy with PPO-clip-style loss + KL penalty toward reference
```

## Reward function

We use 3 rule-based components — no learned reward model:

```python
r = 1.0 * (final_answer == ground_truth)          # correctness
  + 0.1 * has_think_tag(c) * has_answer_tag(c)    # format
  + 0.05 * (50 <= n_tokens <= 400)                # length sanity
```

## How to run

```bash
# Step 1: build SFT-style cold-start data (a few hundred examples)
python -m modules.05_grpo_lora.prepare_data --n 200

# Step 2: train (about 8-12 hours on M5 Pro 48GB for ~500 steps)
python -m modules.05_grpo_lora.train --steps 500

# Step 3: evaluate the trained LoRA
python -m modules.05_grpo_lora.eval --ckpt checkpoints/grpo-step-500
```

## What to look for in training

- **Reward goes up** (slowly, with high variance — that's normal in RL)
- **Response length grows** from ~150 tokens → ~500+ tokens
- **Reflection words** ("wait", "let me reconsider", "actually") start appearing
  more frequently after ~200-500 steps. This is the **"aha moment"**.

The `train.py` script logs all three metrics; results land in `results/05_grpo_lora/`.

## Resource budget

| Component | Memory | Notes |
|---|---|---|
| Policy (Qwen 1.5B, bf16) | ~3 GB | LoRA-trainable |
| Reference (frozen) | ~3 GB | Required by GRPO |
| KV cache for G=4 rollouts | ~5-10 GB | Depends on max length |
| Optimizer states (LoRA only) | ~0.5 GB | LoRA is the lifesaver |
| **Total** | **~15 GB** | Comfortable on 48GB Mac |

## Caveats

- MPS doesn't support `flash-attn` or `bitsandbytes` 4-bit training, so we stick to bf16.
- No `vLLM` rollouts → generation is slower than CUDA equivalents (~3-5×).
- For larger models (3B+), consider renting an RTX 4090 cloud instance.
