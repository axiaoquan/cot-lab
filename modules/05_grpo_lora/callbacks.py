"""Custom callbacks for tracking GRPO training dynamics.

Specifically tuned to surface "R1-Zero emergence" signals:
  - reward (per-step mean and std)
  - response length (token count of completions)
  - reflection-word frequency (how often "wait", "let me reconsider", etc. appear)

Outputs:
  results/05_grpo_lora/training_log.jsonl   (one JSON per step)
  results/05_grpo_lora/training_curves.png  (plotted at end of training)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from transformers import TrainerCallback


REFLECTION_PATTERN = re.compile(
    r"\b(wait|let me reconsider|let me re-?check|actually|hmm|on second thought|"
    r"reconsider|wait,|hold on|重新|等等|让我再想)\b",
    re.IGNORECASE,
)


def count_reflection(text: str) -> int:
    return len(REFLECTION_PATTERN.findall(text or ""))


class GRPOMetricsCallback(TrainerCallback):
    """Streams per-step metrics to JSONL + plots curves at the end.

    trl's GRPOTrainer logs `reward` / `reward_std` / etc. via `on_log`.
    This callback augments that with response-length + reflection stats by
    monkeying state every step.

    NOTE: trl's API surface here is a moving target. We try to be defensive.
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "training_log.jsonl"
        self.records: List[Dict[str, Any]] = []
        # Truncate at start of run so reruns don't pile up.
        self.jsonl_path.write_text("")

    # ------------------------------------------------------------------
    # Hook: called whenever the trainer pushes a log dict.
    # ------------------------------------------------------------------
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, **logs}
        # Best-effort: pull most recent completion lengths + reflection rate
        # from anywhere we can find them on the trainer state.
        completions = self._maybe_get_completions(kwargs)
        if completions:
            lens = [len((c or "").split()) for c in completions]
            record["completion_len_mean"] = sum(lens) / max(len(lens), 1)
            record["completion_len_max"] = max(lens) if lens else 0
            refl = [count_reflection(c) for c in completions]
            record["reflection_rate"] = (
                sum(1 for r in refl if r > 0) / max(len(refl), 1)
            )
            record["reflection_count_mean"] = sum(refl) / max(len(refl), 1)

        self.records.append(record)
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _maybe_get_completions(kwargs):
        """Best-effort completion fishing across trl versions."""
        for key in ("completions", "_completions", "outputs"):
            v = kwargs.get(key)
            if isinstance(v, list) and v and isinstance(v[0], str):
                return v
        return None

    # ------------------------------------------------------------------
    # Hook: called at end of training — plot.
    # ------------------------------------------------------------------
    def on_train_end(self, args, state, control, **kwargs):
        self.plot()

    def plot(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Plot] matplotlib not installed; skipping curves.")
            return

        if not self.records:
            return

        steps = [r.get("step", i) for i, r in enumerate(self.records)]
        rewards = [r.get("reward") for r in self.records]
        lens = [r.get("completion_len_mean") for r in self.records]
        refl = [r.get("reflection_rate") for r in self.records]

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        if any(r is not None for r in rewards):
            axes[0].plot(steps, rewards, color="#3b82f6", linewidth=1.5)
            axes[0].set_ylabel("Reward")
            axes[0].set_title("Mean reward per step")
            axes[0].grid(alpha=0.3)

        if any(r is not None for r in lens):
            axes[1].plot(steps, lens, color="#10b981", linewidth=1.5)
            axes[1].set_ylabel("Avg completion length\n(tokens, whitespace)")
            axes[1].set_title("Response length growth")
            axes[1].grid(alpha=0.3)

        if any(r is not None for r in refl):
            axes[2].plot(steps, refl, color="#ef4444", linewidth=1.5)
            axes[2].set_ylabel("Reflection rate")
            axes[2].set_xlabel("Training step")
            axes[2].set_title('Frequency of "wait / let me reconsider / ..." emergence')
            axes[2].grid(alpha=0.3)

        plt.tight_layout()
        out = self.log_dir / "training_curves.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[Plot] Saved {out}")
