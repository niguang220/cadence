"""Pure unit tests for three_layer_recall: |gold ∩ stage| / |gold| over unique table sets."""
import pytest

from agent.retrieval.contracts import (RelationPlan, RetrievalResult, SelectionDecision, TableCandidate)
from evalharness.e2e_eval import three_layer_recall


def _rr(cands, anchors, context):
    return RetrievalResult(config_name="x", signals=[],
        candidates=[TableCandidate(t, {}, None, i + 1) for i, t in enumerate(cands)],
        metric_matches=[],
        selection=SelectionDecision(anchors, [], "topk", {}),
        relation_plan=RelationPlan("shortest_path", anchors, [], context, [], [], []),
        stage_events=[])


def test_full_recall():
    r = three_layer_recall(_rr(["a", "b"], ["a", "b"], ["a", "b"]), ["a", "b"], case_id="c")
    assert r == {"candidate_recall": 1.0, "selection_recall": 1.0, "context_recall": 1.0}


def test_partial_and_selector_drop_and_closure_recovery():
    # gold {a,b}: candidates have both; selector drops b (anchors=[a]); closure recovers b in context.
    r = three_layer_recall(_rr(["a", "b", "c"], ["a"], ["a", "b"]), ["a", "b"], case_id="c")
    assert r["candidate_recall"] == 1.0      # both gold among candidates
    assert r["selection_recall"] == 0.5      # only a survived selection
    assert r["context_recall"] == 1.0        # b recovered by closure


def test_duplicate_table_names_do_not_change_denominator():
    r = three_layer_recall(_rr(["a", "a", "a"], ["a", "a"], ["a", "a"]), ["a", "b"], case_id="c")
    assert r["candidate_recall"] == 0.5      # |{a}∩{a,b}|/|{a,b}| = 1/2, dups collapse
    assert r["selection_recall"] == 0.5 and r["context_recall"] == 0.5


def test_missing_or_empty_gold_fails_fast():
    with pytest.raises(ValueError, match="mycase"):
        three_layer_recall(_rr(["a"], ["a"], ["a"]), [], case_id="mycase")
    with pytest.raises(ValueError, match="mycase"):
        three_layer_recall(_rr(["a"], ["a"], ["a"]), None, case_id="mycase")
