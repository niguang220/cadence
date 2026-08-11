"""Rank-sensitive retrieval metrics: ordered candidates, Fusion@5 (distinct from saturated
candidate recall), precision, selection/context — pure derivation from a typed RetrievalResult."""
from agent.retrieval.contracts import (RelationPlan, RetrievalResult, SelectionDecision,
                                        TableCandidate)
from evalharness.value_metrics import rank_sensitive_metrics


def _rr(cand_tables):
    cands = [TableCandidate(t, {}, 1.0 / (i + 1), i + 1) for i, t in enumerate(cand_tables)]
    return RetrievalResult(
        config_name="x", signals=[], candidates=cands, metric_matches=[],
        selection=SelectionDecision(cand_tables[:2], [], "topk", {}),
        relation_plan=RelationPlan("shortest_path", cand_tables[:2], [], cand_tables[:3], [], [], []),
        stage_events=[])


def test_metrics_full():
    m = rank_sensitive_metrics(_rr(["company", "ticket", "x", "y", "z", "w"]), ["company", "ticket"])
    assert m["candidate_tables_ordered"] == ["company", "ticket", "x", "y", "z", "w"]
    assert m["candidate_count"] == 6
    assert m["candidate_recall"] == 1.0
    assert m["fusion_at_5_recall"] == 1.0
    assert m["candidate_precision"] == 2 / 6
    assert m["selection_tables"] == ["company", "ticket"]
    assert m["context_tables"] == ["company", "ticket", "x"]


def test_fusion_at_5_is_not_saturated_candidate_recall():
    # gold 'w' is candidate rank 6 -> in the candidate set but NOT in the fusion top-5
    m = rank_sensitive_metrics(_rr(["a", "b", "c", "d", "e", "w"]), ["w"])
    assert m["candidate_recall"] == 1.0
    assert m["fusion_at_5_recall"] == 0.0


def test_goldless_negative_recall_is_none():
    m = rank_sensitive_metrics(_rr([]), [])
    assert m["candidate_recall"] is None and m["fusion_at_5_recall"] is None
    assert m["candidate_precision"] is None and m["candidate_count"] == 0


def test_selection_and_context_recall():
    # selection = first 2 (company, ticket); context = first 3 (company, ticket, x)
    m = rank_sensitive_metrics(_rr(["company", "ticket", "x", "y", "z", "w"]), ["company", "x"])
    assert m["selection_recall"] == 0.5      # 'company' selected, 'x' only a candidate
    assert m["context_recall"] == 1.0        # both in the one-hop context set


def test_goldless_selection_context_recall_is_none():
    m = rank_sensitive_metrics(_rr(["a", "b", "c"]), [])
    assert m["selection_recall"] is None and m["context_recall"] is None
