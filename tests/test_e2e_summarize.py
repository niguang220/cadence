"""Aggregation of EvalRecords into the E2E capability report (pure logic)."""
from evalharness.e2e_eval import EvalRecord, summarize

_CFG = {"name": "lexical_baseline"}


def _rec(cat, sl, match, *, first=True, final=True, repairs=0, lat=100.0, retrieval_k=5,
        repeat_index=0):
    return EvalRecord(id=f"{cat}-{sl}-{match}", category=cat, question="q",
                      semantic_layer=sl, retrieval_config=_CFG, retrieval_k=retrieval_k,
                      repeat_index=repeat_index,
                      predicted_sql="s", gold_sql="g", exec_match=match,
                      sql_valid_first_try=first, sql_valid_final=final,
                      repair_attempts=repairs, latency_ms=lat,
                      prompt_tokens=10, completion_tokens=2)


def _mk_control(*, id, semantic_layer, exec_match, retrieval_config, retrieval_k, repeat_index,
                first=True, final=True, repairs=0, lat=1.0):
    return EvalRecord(id=id, category="control", question="q",
                      semantic_layer=semantic_layer, retrieval_config=retrieval_config,
                      retrieval_k=retrieval_k, repeat_index=repeat_index,
                      predicted_sql="s", gold_sql="g", exec_match=exec_match,
                      sql_valid_first_try=first, sql_valid_final=final,
                      repair_attempts=repairs, latency_ms=lat,
                      prompt_tokens=1, completion_tokens=1)


def test_traps_split_on_off():
    recs = [_rec("mrr", True, True), _rec("mrr", True, False),
            _rec("mrr", False, False), _rec("mrr", False, False)]
    s = summarize(recs)
    assert s["traps"]["on"] == {"exec_match_rate": 0.5, "matched": 1, "n": 2}
    assert s["traps"]["off"] == {"exec_match_rate": 0.0, "matched": 0, "n": 2}


def test_controls_diverging_ids_flag_asymmetry():
    on = _mk_control(id="c1", semantic_layer=True, exec_match=True, retrieval_config=_CFG,
                     retrieval_k=5, repeat_index=0)
    off = _mk_control(id="c1", semantic_layer=False, exec_match=False, retrieval_config=_CFG,
                      retrieval_k=5, repeat_index=0)
    s = summarize([on, off])
    assert s["controls"]["diverging_ids"] == ["c1"]


def test_repeats_disagree_but_paired_on_off_agree_is_not_divergent():
    cfg = {"name": "baseline"}

    def rec(sl, match, rep):
        return _mk_control(id="c1", semantic_layer=sl, exec_match=match, retrieval_config=cfg,
                           retrieval_k=5, repeat_index=rep)

    recs = [rec(False, True, 0), rec(True, True, 0),     # repeat 0: ON==OFF (agree)
            rec(False, False, 1), rec(True, False, 1)]   # repeat 1: ON==OFF (agree) but differs from rep 0
    assert summarize(recs)["controls"]["diverging_ids"] == []    # model randomness, NOT divergence


def test_repair_lift_and_percentiles():
    recs = [_rec("mrr", True, True, first=False, final=True, repairs=1, lat=100.0),
            _rec("mrr", True, True, first=True, final=True, repairs=0, lat=300.0)]
    s = summarize(recs)
    assert s["sql_valid_first_try_rate"] == 0.5
    assert s["sql_valid_after_repair_rate"] == 1.0
    assert s["avg_repair_attempts"] == 0.5
    assert s["latency_ms"]["p50"] == 100.0 and s["latency_ms"]["p95"] == 300.0


def test_empty_records_is_safe():
    s = summarize([])
    assert s["n_records"] == 0 and s["traps"]["on"]["n"] == 0
