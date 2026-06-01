"""Module 05 · GRPO LoRA training.

Uses trl's GRPOTrainer with:
  - Qwen2.5-1.5B-Instruct as the policy
  - LoRA adapters (q/k/v/o)
  - Rule-based reward (correctness + format + length + reflection)
  - Group size = 4 (Mac-friendly)

Run:
    python -m modules.05_grpo_lora.prepare_data --n 500
    python -m modules.05_grpo_lora.train --steps 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# trl >= 0.12 exposes GRPOTrainer / GRPOConfig
try:
    from trl import GRPOTrainer, GRPOConfig
except ImportError as e:
    raise ImportError(
        "Could not import trl.GRPOTrainer. Make sure you have trl>=0.12.\n"
        "    pip install --upgrade 'trl>=0.12'"
    ) from e

from .rewards import reward_fn
from .callbacks import GRPOMetricsCallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data", default="data/grpo_gsm8k")
    p.add_argument("--out", default="checkpoints/grpo")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--log-dir", default="results/05_grpo_lora",
                   help="Where to dump training_log.jsonl + training_curves.png")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    grpo_cfg = cfg["grpo"]
    model_cfg = cfg["model"]

    print(f"[Train] Loading model {model_cfg['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if model_cfg["torch_dtype"] == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], torch_dtype=dtype, trust_remote_code=True,
    )

    # Wrap with LoRA.
    lora_cfg = LoraConfig(
        r=grpo_cfg["lora_r"],
        lora_alpha=grpo_cfg["lora_alpha"],
        target_modules=grpo_cfg["lora_target_modules"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"[Train] Loading dataset from {args.data}...")
    train_ds = load_from_disk(args.data)

    training_args = GRPOConfig(
        output_dir=args.out,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=grpo_cfg["per_device_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        learning_rate=grpo_cfg["learning_rate"],
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=grpo_cfg["max_completion_length"],
        beta=grpo_cfg["beta"],
        bf16=(model_cfg["torch_dtype"] == "bfloat16"),
        logging_steps=10,
        save_steps=100,
        report_to=[],  # disable wandb by default
        # Mac/MPS safety:
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_ds,
        callbacks=[GRPOMetricsCallback(log_dir=args.log_dir)],
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"[Train] Done. Checkpoint at {args.out}")
    print(f"[Train] Curves at {args.log_dir}/training_curves.png")


if __name__ == "__main__":
    main()
