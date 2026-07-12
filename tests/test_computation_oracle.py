"""Tests for the sandbox computation comparator.

execution_match is row-oriented; sandbox output is arbitrary JSON. This locks:
numeric tolerance, bool != int, dict key-set equality, ordered lists, nesting, and
that charts are explicitly unsupported (a "chart" key raises, never silently passes).
"""
import pytest

from evalharness.computation_oracle import computation_match


def test_scalar_within_tolerance():
    assert computation_match(105.00001, 105.0, tolerance=1e-3)
    assert not computation_match(105.2, 105.0, tolerance=1e-3)


def test_bool_is_not_int():
    assert not computation_match(True, 1)
    assert computation_match(True, True)


def test_dict_key_set_must_match():
    assert computation_match({"a": 1, "b": 2.0}, {"a": 1.0, "b": 2})
    assert not computation_match({"a": 1}, {"a": 1, "b": 2})


def test_list_is_ordered():
    assert computation_match([1, 2, 3], [1, 2, 3])
    assert not computation_match([1, 3, 2], [1, 2, 3])


def test_nested_structure():
    p = [{"region": "eu", "avg": 150.0}, {"region": "us", "avg": 300.0}]
    e = [{"region": "eu", "avg": 150.0}, {"region": "us", "avg": 300.0}]
    assert computation_match(p, e)


def test_type_mismatch_is_false():
    assert not computation_match({"a": 1}, [1])


def test_chart_key_is_unsupported():
    with pytest.raises(ValueError):
        computation_match({"chart": "iVBOR..."}, {"chart": "iVBOR..."})
