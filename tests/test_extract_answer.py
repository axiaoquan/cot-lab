"""Tests for common.extract_answer.

These functions are the single most fragile thing in the project — every module
relies on them to compare model outputs against gold answers.
"""
import pytest

from common.extract_answer import (
    extract_answer,
    extract_thinking,
    answers_equal,
)


# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    def test_returns_none_on_empty(self):
        assert extract_answer("") is None
        assert extract_answer(None) is None

    # --- Tier 1: <answer> tag ---
    def test_answer_tag_simple(self):
        assert extract_answer("<answer>42</answer>") == "42"

    def test_answer_tag_with_text(self):
        text = "Some thinking. <answer>The answer is 18</answer> More text."
        assert extract_answer(text) == "18"

    def test_answer_tag_multiline(self):
        text = "<answer>\n  17.5\n</answer>"
        assert extract_answer(text) == "17.5"

    # --- Tier 2: \boxed{} ---
    def test_boxed(self):
        assert extract_answer("Therefore $\\boxed{42}$.") == "42"

    def test_boxed_with_unit(self):
        assert extract_answer("\\boxed{18 dollars}") == "18"

    # --- Tier 3: Final answer / The answer is ---
    def test_final_answer_marker(self):
        assert extract_answer("Some reasoning.\nFinal answer: 42") == "42"

    def test_the_answer_is(self):
        assert extract_answer("Therefore, the answer is 42.") == "42"

    def test_chinese_marker(self):
        assert extract_answer("推理过程...\n答案是 42") == "42"

    def test_gsm8k_marker(self):
        assert extract_answer("blah blah #### 42") == "42"

    # --- Tier 4: last number fallback ---
    def test_last_number_fallback(self):
        text = "We have 3 apples and 5 oranges, total 8."
        assert extract_answer(text) == "8"

    def test_negative_numbers(self):
        assert extract_answer("Final answer: -7") == "-7"

    def test_decimals_preserve(self):
        assert extract_answer("Final answer: 3.14") == "3.14"

    def test_decimals_normalize_integer_floats(self):
        # 3.00 should normalize to 3
        assert extract_answer("Final answer: 3.00") == "3"

    def test_commas_in_numbers(self):
        # 1,000 should be normalized to 1000
        assert extract_answer("Final answer: 1,000") == "1000"

    def test_priority_answer_tag_wins(self):
        """<answer> should win over \\boxed{} and last number."""
        text = "Lots of numbers 1, 2, 3. \\boxed{99}. <answer>42</answer>"
        assert extract_answer(text) == "42"


# ---------------------------------------------------------------------------
# extract_thinking
# ---------------------------------------------------------------------------


class TestExtractThinking:
    def test_simple(self):
        text = "<think>step 1, step 2</think><answer>42</answer>"
        assert extract_thinking(text) == "step 1, step 2"

    def test_no_think_tag(self):
        assert extract_thinking("just an answer") is None

    def test_multiline(self):
        text = "<think>\nLet me work this out.\n3 + 5 = 8\n</think>"
        assert "3 + 5 = 8" in extract_thinking(text)


# ---------------------------------------------------------------------------
# answers_equal
# ---------------------------------------------------------------------------


class TestAnswersEqual:
    def test_string_match(self):
        assert answers_equal("42", "42")

    def test_none_pred(self):
        assert not answers_equal(None, "42")

    def test_int_vs_float(self):
        assert answers_equal("42", "42.0")
        assert answers_equal("42.0", "42")

    def test_close_floats(self):
        assert answers_equal("3.14159", "3.1416")  # within 1e-4? No → False
        assert answers_equal("3.14160", "3.14160")
        assert not answers_equal("3.14", "3.15")

    def test_with_commas(self):
        assert answers_equal("1,000", "1000")

    def test_obvious_mismatch(self):
        assert not answers_equal("42", "43")

    def test_string_non_numeric(self):
        # If the prediction is a non-numeric string, it should still match exactly.
        assert answers_equal("yes", "yes")
        assert not answers_equal("yes", "no")
