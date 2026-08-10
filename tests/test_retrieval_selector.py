from agent.retrieval.contracts import MetricMatch, TableCandidate
from agent.retrieval.selector import (NoOpSelector, TopKSelector, fallback_topk,
                                      protected_anchors, validate_structured_selection)


def _cand(table, rank):
    return TableCandidate(table=table, channel_results={}, fusion_score=1.0 / rank, fusion_rank=rank)


def _mm(tables):
    return MetricMatch(metric="mrr", match_type="alias", score=1.0, required_tables=tables,
                       required_columns=[], required_filters=[])


def test_topk_takes_fusion_top_k_plus_protected():
    cands = [_cand(t, i + 1) for i, t in enumerate("abcdef")]
    dec = TopKSelector().select(cands, protected=["z"], context_anchor_k=5)   # 'z' not in candidates
    assert dec.selector == "topk"
    assert {"a", "b", "c", "d", "e"}.issubset(set(dec.anchor_tables))
    assert "z" in dec.anchor_tables and "z" not in dec.dropped_tables         # protected, undroppable
    assert "f" in dec.dropped_tables


def test_noop_keeps_all_candidates():
    cands = [_cand("a", 1), _cand("b", 2)]
    dec = NoOpSelector().select(cands, protected=[], context_anchor_k=5)
    assert dec.selector == "noop" and set(dec.anchor_tables) == {"a", "b"}


def test_protected_anchors_is_sorted_union():
    assert protected_anchors([_mm(["subscription", "account"]), _mm(["account"])]) == \
        ["account", "subscription"]


def test_validate_rejects_empty_and_unknown():
    cands = [_cand("a", 1), _cand("b", 2)]
    assert validate_structured_selection([], cands) is None
    assert validate_structured_selection(["zzz"], cands) is None
    assert validate_structured_selection("a", cands) is None
    assert validate_structured_selection(["a", "b"], cands) == ["a", "b"]


def test_fallback_topk_emits_stage_event_and_keeps_protected():
    cands = [_cand(t, i + 1) for i, t in enumerate("abcdef")]
    dec, ev = fallback_topk(cands, protected=["z"], context_anchor_k=5)
    assert dec.selector == "topk" and "z" in dec.anchor_tables
    assert ev.stage == "selection" and ev.event == "selector_fallback"
