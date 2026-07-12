"""Tests for the strict golden loaders.

Every loader rejects an empty dataset, duplicate ids, and unknown fields (naming the
offending key). Enum fields are validated. A gate question="" is LEGAL (empty input is
a valid adversarial out_of_scope case). Sandbox rejects a truncated input and a chart
expected_output.
"""
import json

import pytest

from evalharness.golden import (
    GateCase, load_consistency, load_gate, load_sandbox,
)


def _write(tmp_path, data):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_gate_loads_and_allows_empty_question(tmp_path):
    p = _write(tmp_path, [{"id": "e", "question": "", "expected_route": "out_of_scope"}])
    cases = load_gate(p)
    assert cases == [GateCase(id="e", question="", expected_route="out_of_scope")]


def test_reject_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_gate(_write(tmp_path, []))


def test_reject_duplicate_id(tmp_path):
    data = [{"id": "x", "question": "a", "expected_route": "proceed", "recalled_tables": ["t"]},
            {"id": "x", "question": "b", "expected_route": "proceed", "recalled_tables": ["t"]}]
    with pytest.raises(ValueError, match="duplicate"):
        load_gate(_write(tmp_path, data))


def test_reject_unknown_field_names_the_key(tmp_path):
    data = [{"id": "x", "question": "a", "expected_route": "proceed", "bogus": 1}]
    with pytest.raises(ValueError, match="bogus"):
        load_gate(_write(tmp_path, data))


def test_reject_bad_route_enum(tmp_path):
    with pytest.raises(ValueError, match="expected_route"):
        load_gate(_write(tmp_path, [{"id": "x", "question": "a", "expected_route": "maybe"}]))


def test_consistency_clean_requires_empty_category(tmp_path):
    ok = [{"id": "c", "question": "q", "candidate_sql": "S", "gold_sql": "S",
           "category": "", "expected_caught": False}]
    assert load_consistency(_write(tmp_path, ok))[0].expected_caught is False
    bad = [{"id": "c", "question": "q", "candidate_sql": "S", "gold_sql": "S",
            "category": "measure", "expected_caught": False}]
    with pytest.raises(ValueError, match="category"):
        load_consistency(_write(tmp_path, bad))


def test_sandbox_rejects_truncated_input(tmp_path):
    data = [{"id": "s", "instruction": "i",
             "input": {"columns": ["a"], "rows": [[1]], "truncated": True},
             "expected_output": {"v": 1}}]
    with pytest.raises(ValueError, match="truncated"):
        load_sandbox(_write(tmp_path, data))


def test_sandbox_rejects_chart_expected(tmp_path):
    data = [{"id": "s", "instruction": "i", "input": {"columns": ["a"], "rows": [[1]]},
             "expected_output": {"chart": "iVBOR"}}]
    with pytest.raises(ValueError, match="chart"):
        load_sandbox(_write(tmp_path, data))
