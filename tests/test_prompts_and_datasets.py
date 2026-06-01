"""Smoke tests for prompt builders and dataset loaders.

These don't hit the network — for GSM8K we test parsing, not download.
"""
from common.prompts import (
    DIRECT_TEMPLATE,
    ZERO_SHOT_COT_TEMPLATE,
    R1_SYSTEM_PROMPT,
    build_prompt,
)
from common.datasets import load_24_game


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_build_prompt_inserts_question():
    p = build_prompt(DIRECT_TEMPLATE, "What is 2+2?")
    assert "What is 2+2?" in p
    assert "Answer:" in p


def test_zero_shot_cot_includes_step_phrase():
    p = build_prompt(ZERO_SHOT_COT_TEMPLATE, "Q?")
    assert "step by step" in p.lower()


def test_r1_system_prompt_has_tags():
    assert "<think>" in R1_SYSTEM_PROMPT
    assert "<answer>" in R1_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


def test_load_24_game_basic():
    examples = list(load_24_game(n=3))
    assert len(examples) == 3
    for ex in examples:
        assert "question" in ex
        assert "answer" in ex
        assert "raw" in ex
        assert "numbers" in ex["raw"]
        assert len(ex["raw"]["numbers"]) == 4
