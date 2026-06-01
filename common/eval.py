"""Shared evaluation: accuracy + latency + per-example records."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, List, Dict, Any

from .extract_answer import extract_answer, answers_equal


@dataclass
class ExampleRecord:
    question: str
    gold: str
    pred: str | None
    completion: str
    correct: bool
    latency_s: float


@dataclass
class RunReport:
    method: str
    model: str
    dataset: str
    n: int
    accuracy: float
    avg_latency_s: float
    records: List[ExampleRecord] = field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "model": self.model,
            "dataset": self.dataset,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "avg_latency_s": round(self.avg_latency_s, 4),
        }


def evaluate(
    method_name: str,
    model_name: str,
    dataset_name: str,
    examples: Iterable[Dict[str, Any]],
    solver: Callable[[str], str],
) -> RunReport:
    """Run `solver(question)` over examples; collect accuracy + latency.

    `solver` should return the *full completion string*; this fn extracts the answer.
    """
    records: List[ExampleRecord] = []
    correct = 0
    total_latency = 0.0
    n = 0

    for ex in examples:
        q, gold = ex["question"], ex["answer"]
        t0 = time.time()
        completion = solver(q)
        dt = time.time() - t0

        pred = extract_answer(completion)
        is_correct = answers_equal(pred, gold)

        records.append(
            ExampleRecord(
                question=q, gold=gold, pred=pred,
                completion=completion, correct=is_correct, latency_s=dt,
            )
        )
        if is_correct:
            correct += 1
        total_latency += dt
        n += 1

    return RunReport(
        method=method_name,
        model=model_name,
        dataset=dataset_name,
        n=n,
        accuracy=(correct / n) if n else 0.0,
        avg_latency_s=(total_latency / n) if n else 0.0,
        records=records,
    )


def save_report(report: RunReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": report.to_summary(), "records": [asdict(r) for r in report.records]},
            f, ensure_ascii=False, indent=2,
        )
    print(f"[Eval] Wrote {path}")
