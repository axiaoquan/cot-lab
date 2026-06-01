"""Robust answer extraction from free-form model output.

Supports the following formats (in priority order):
  1. <answer>...</answer> tag
  2. \\boxed{...}
  3. "Final answer: ..." marker
  4. Last number in the string
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

_NUM_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize_number(s: str) -> str:
    s = s.strip().rstrip(".").replace(",", "")
    # Strip trailing zeros after decimal (e.g. "8.00" -> "8")
    if "." in s:
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
            return str(f)
        except ValueError:
            return s
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    """Try multiple patterns to extract a final numeric/string answer."""
    if not text:
        return None

    # 1. <answer>...</answer>
    m = re.search(r"<answer>\s*(.+?)\s*</answer>", text, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        # Strip commas in numbers like "1,000" before extraction.
        nums = _NUM_PATTERN.findall(inner.replace(",", ""))
        return _normalize_number(nums[-1]) if nums else inner

    # 2. \boxed{...}
    m = re.search(r"\\boxed\{([^{}]+)\}", text)
    if m:
        inner = m.group(1).strip()
        nums = _NUM_PATTERN.findall(inner.replace(",", ""))
        return _normalize_number(nums[-1]) if nums else inner

    # 3. Final answer: <X>  /  The answer is <X>  /  答案是 <X>
    for marker in [
        r"Final answer\s*[:：]\s*(.+)",
        r"The answer is\s*(.+)",
        r"答案是\s*(.+)",
        r"####\s*(.+)",
    ]:
        m = re.search(marker, text, re.IGNORECASE)
        if m:
            tail = m.group(1).strip().split("\n")[0]
            nums = _NUM_PATTERN.findall(tail.replace(",", ""))
            return _normalize_number(nums[-1]) if nums else tail.strip()

    # 4. Last number anywhere in the string
    nums = _NUM_PATTERN.findall(text.replace(",", ""))
    if nums:
        return _normalize_number(nums[-1])

    return None


def extract_thinking(text: str) -> Optional[str]:
    """Extract the content of <think>...</think> if present."""
    m = re.search(r"<think>\s*(.+?)\s*</think>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Equality check
# ---------------------------------------------------------------------------


def answers_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p = _normalize_number(str(pred))
    g = _normalize_number(str(gold))
    if p == g:
        return True
    # Try float compare
    try:
        return abs(float(p) - float(g)) < 1e-4
    except (ValueError, TypeError):
        return False
