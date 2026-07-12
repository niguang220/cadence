"""Tests for the classification metrics used by the gate surface.

Pins accuracy, the binary positive-class precision/recall/f1, support (# actual
positives), the zero-denominator convention (0.0 + support, never a flattering 1.0),
and the confusion matrix shape.
"""
import pytest

from evalharness.classification_metrics import (
    ClassMetrics, accuracy, binary_metrics, confusion,
)


def test_accuracy_counts_exact_matches():
    assert accuracy(["a", "b", "a"], ["a", "b", "b"]) == pytest.approx(2 / 3)


def test_accuracy_empty_is_1():
    assert accuracy([], []) == 1.0


def test_binary_metrics_perfect():
    m = binary_metrics(["refuse", "ok", "refuse"], ["refuse", "ok", "refuse"], positive="refuse")
    assert (m.precision, m.recall, m.f1, m.support) == (1.0, 1.0, 1.0, 2)


def test_binary_metrics_false_positive_drops_precision():
    # predicts refuse on an ok case -> precision 1/2, recall 1/1
    m = binary_metrics(["refuse", "ok"], ["refuse", "refuse"], positive="refuse")
    assert m.precision == pytest.approx(0.5) and m.recall == 1.0 and m.support == 1


def test_no_predicted_positives_is_zero_not_one():
    # never predicts the positive class -> precision denom 0 -> 0.0 (not a silent 1.0)
    m = binary_metrics(["refuse", "ok"], ["ok", "ok"], positive="refuse")
    assert m.precision == 0.0 and m.recall == 0.0 and m.support == 1


def test_no_actual_positives_support_zero():
    # support distinguishes "0.0 because none in the set" from "predicted all wrong"
    m = binary_metrics(["ok", "ok"], ["ok", "ok"], positive="refuse")
    assert m.support == 0 and m.recall == 0.0


def test_confusion_matrix_counts():
    c = confusion(["a", "a", "b"], ["a", "b", "b"], labels=["a", "b"])
    assert c[("a", "a")] == 1 and c[("a", "b")] == 1 and c[("b", "b")] == 1 and c[("b", "a")] == 0
