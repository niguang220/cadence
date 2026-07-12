"""Classification metrics for the gate surface.

Pure data + maths (no LLM, no I/O). ``binary_metrics`` reports the positive class we
care about (a refusal), and ``support`` is the number of ACTUAL positives so a 0.0
that means "no positives in this set" is distinguishable from "predicted all wrong".
Zero-denominator always yields 0.0 -- never a flattering 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Fraction of positions where prediction equals truth. Empty -> 1.0."""
    if not y_true:
        return 1.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


@dataclass
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int          # number of ACTUAL positives (tp + fn)


def binary_metrics(y_true: list[str], y_pred: list[str], *, positive: str) -> ClassMetrics:
    """Precision/recall/f1/support for the ``positive`` class. Zero denom -> 0.0."""
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ClassMetrics(positive, precision, recall, f1, support=tp + fn)


def confusion(y_true: list[str], y_pred: list[str],
              labels: list[str]) -> dict[tuple[str, str], int]:
    """Counts keyed by (true_label, pred_label) over the given label set."""
    counts = {(t, p): 0 for t in labels for p in labels}
    for t, p in zip(y_true, y_pred):
        if (t, p) in counts:
            counts[(t, p)] += 1
    return counts
