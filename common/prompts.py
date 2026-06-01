"""Prompt templates used across modules."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Direct (no CoT)
# ---------------------------------------------------------------------------

DIRECT_TEMPLATE = (
    "Solve the following problem. Output only the final answer, nothing else.\n\n"
    "Problem: {question}\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Zero-shot CoT — Kojima et al. 2022 ("Let's think step by step")
# ---------------------------------------------------------------------------

ZERO_SHOT_COT_TEMPLATE = (
    "Solve the following problem. Think step by step, then put the final answer "
    "after the marker `Final answer:`.\n\n"
    "Problem: {question}\n"
    "Let's think step by step."
)


# ---------------------------------------------------------------------------
# Few-shot CoT (smaller, for quick experiments)
# ---------------------------------------------------------------------------

FEW_SHOT_COT_TEMPLATE = (
    "Solve math problems step by step.\n\n"
    "Problem: Janet has 3 apples and buys 5 more. How many does she have?\n"
    "Solution: She starts with 3, buys 5 more, so 3 + 5 = 8.\n"
    "Final answer: 8\n\n"
    "Problem: A pen costs $2 and a notebook costs $5. Total cost of 3 pens and 2 notebooks?\n"
    "Solution: 3 pens = 3 * 2 = 6. 2 notebooks = 2 * 5 = 10. Total = 6 + 10 = 16.\n"
    "Final answer: 16\n\n"
    "Problem: {question}\n"
    "Solution:"
)


# ---------------------------------------------------------------------------
# R1-style (used by Module 05 GRPO training)
# ---------------------------------------------------------------------------

R1_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The Assistant first thinks about the reasoning "
    "process inside <think> </think> tags, then provides the final answer "
    "inside <answer> </answer> tags."
)

R1_USER_TEMPLATE = "{question}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_prompt(template: str, question: str) -> str:
    return template.format(question=question)
