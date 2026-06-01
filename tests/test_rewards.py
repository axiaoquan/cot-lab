"""Tests for the rule-based reward functions used in Module 05 GRPO."""
import pytest

from modules import __init__  # noqa: F401  (ensure modules pkg importable)

# Import via direct file path because the module folder name has a leading digit.
import importlib.util
import pathlib

REPO = pathlib.Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location(
    "rewards", REPO / "modules" / "05_grpo_lora" / "rewards.py",
)
rewards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewards)


def test_correctness_reward_correct():
    completion = "<answer>42</answer>"
    assert rewards.correctness_reward(completion, "42") == 1.0


def test_correctness_reward_wrong():
    completion = "<answer>17</answer>"
    assert rewards.correctness_reward(completion, "42") == 0.0


def test_correctness_reward_no_answer():
    """Empty / no-answer completion gets 0."""
    assert rewards.correctness_reward("", "42") == 0.0


def test_format_reward_full():
    text = "<think>step 1</think><answer>42</answer>"
    assert rewards.format_reward(text) == 0.1


def test_format_reward_only_think():
    text = "<think>step 1</think>final: 42"
    assert rewards.format_reward(text) == 0.0


def test_format_reward_only_answer():
    text = "stuff <answer>42</answer>"
    assert rewards.format_reward(text) == 0.0


def test_length_reward_in_range():
    completion = " ".join(["word"] * 100)
    assert rewards.length_reward(completion) == 0.05


def test_length_reward_too_short():
    assert rewards.length_reward("hi") == 0.0


def test_length_reward_too_long():
    completion = " ".join(["word"] * 2000)
    assert rewards.length_reward(completion) == 0.0


def test_reflection_bonus_zero():
    assert rewards.reflection_bonus("just a plain answer") == 0.0


def test_reflection_bonus_nonzero():
    assert rewards.reflection_bonus("Wait, let me reconsider.") > 0.0


def test_reflection_bonus_capped():
    # More reflection words should not blow up the bonus.
    text = "wait wait wait actually wait wait hmm hmm 重新 等等"
    assert rewards.reflection_bonus(text) <= 0.05


def test_composite_reward_correct_full_format():
    completion = "<think>3+5=8 then 8+2=10</think><answer>10</answer> wait, let me recheck."
    rewards_list = rewards.reward_fn(
        prompts=["dummy"], completions=[completion], ground_truths=["10"],
    )
    assert len(rewards_list) == 1
    # 1.0 (correct) + 0.1 (format) + 0.05 (length) + small reflection bonus
    # bonus is min(0.02 * hits, 0.05); we tolerate the actual computed value.
    assert rewards_list[0] >= 1.10
    assert rewards_list[0] <= 1.25
