"""Reward functions for Module 05 GRPO training.

All reward functions take a list of completions (strings) plus per-example
ground truths, and return a list of floats — one reward per completion.
"""
from __future__ import annotations

import re
from typing import List

from common.extract_answer import extract_answer, answers_equal


# ---------------------------------------------------------------------------
# Component rewards
# ---------------------------------------------------------------------------


def correctness_reward(completion: str, ground_truth: str) -> float:
    """+1 if the extracted answer equals ground truth, else 0."""
    pred = extract_answer(completion)
    return 1.0 if answers_equal(pred, ground_truth) else 0.0


def format_reward(completion: str) -> float:
    """+0.1 if both <think>...</think> and <answer>...</answer> are present."""
    has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
    return 0.1 if (has_think and has_answer) else 0.0


def length_reward(completion: str, lo: int = 50, hi: int = 800) -> float:
    """Small bonus for completions in a reasonable length range (token count proxy)."""
    n = len(completion.split())
    return 0.05 if lo <= n <= hi else 0.0


def reflection_bonus(completion: str) -> float:
    """Tiny bonus when the completion contains reflective phrases.
    Keep small — we want emergence, not Goodhart."""
    pattern = r"\b(wait|let me reconsider|actually|hmm|on second thought|重新|等等)\b"
    hits = len(re.findall(pattern, completion, re.IGNORECASE))
    return min(0.02 * hits, 0.05)


# ---------------------------------------------------------------------------
# Composite reward (the one trl GRPOTrainer will call)
# ---------------------------------------------------------------------------


def reward_fn(prompts, completions, ground_truths=None, **kwargs) -> List[float]:
    """trl-compatible reward function.

    Args:
        prompts:        list[str]  — original prompts (unused but trl passes them)
        completions:    list[str]  — model outputs
        ground_truths:  list[str]  — gold answers (must be passed in via dataset column)

    Returns:
        list[float] of rewards.
    """
    if ground_truths is None:
        # If trl passes the dataset row dict directly, look for "answer" key.
        ground_truths = kwargs.get("answer", [None] * len(completions))

    rewards = []
    for c, gt in zip(completions, ground_truths):
        # If the chat template is used, completions may include the role token.
        # Strip everything before <think> if present, otherwise leave as-is.
        r = correctness_reward(c, gt) + format_reward(c) + length_reward(c) + reflection_bonus(c)
        rewards.append(r)
    return rewards
