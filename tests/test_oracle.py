"""Tests for the result oracle (evalharness/oracle.py).

These pin the canonicalization rules: row order, float precision, NULL, and
int/float equivalence. No database and no LLM — pure comparison logic.
"""
from evalharness.oracle import canonicalize_result, execution_match


def test_row_order_ignored_by_default():
    assert execution_match([(1,), (2,), (3,)], [(3,), (1,), (2,)])


def test_row_order_enforced_when_ordered():
    # top-N / "first" questions: order is part of the answer
    assert not execution_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert execution_match([(1,), (2,)], [(1,), (2,)], ordered=True)


def test_float_tolerance_rounds():
    assert execution_match([(1.001,)], [(1.0,)])          # within 0.01 -> equal
    assert not execution_match([(1.05,)], [(1.0,)])       # beyond 0.01 -> not equal


def test_null_compares_equal():
    assert execution_match([(None,)], [(None,)])
    assert not execution_match([(None,)], [(1,)])


def test_int_and_float_are_equivalent():
    # COUNT(*) -> 306 (int) vs a gold that yields 306.0 (float) should match
    assert execution_match([(306,)], [(306.0,)])


def test_detects_genuine_mismatch():
    assert not execution_match([("Rock", 38)], [("Rock", 39)])
    assert not execution_match([(1,)], [(1,), (2,)])      # extra row


def test_canonicalize_is_deterministic_for_mixed_rows():
    rows = [("b", None), ("a", 2), ("a", 1)]
    assert canonicalize_result(rows) == canonicalize_result(list(reversed(rows)))
