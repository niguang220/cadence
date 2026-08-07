"""Aggregation of EvalRecords into the roadmap Step-1 report (pure logic)."""
from evalharness.e2e_eval import EvalRecord, summarize


def _rec(cat, sl, match, *, first=True, final=True, repairs=0, lat=100.0):
    return EvalRecord(id=f"{cat}-{sl}-{match}", category=cat, question="q",
                      semantic_layer=sl, retrieval_config="lexical",
                      predicted_sql="s", gold_sql="g", exec_match=match,
                      sql_valid_first_try=first, sql_valid_final=final,
                      repair_attempts=repairs, latency_ms=lat,
                      prompt_tokens=10, completion_tokens=2)


def test_traps_split_on_off():
    recs = [_rec("mrr", True, True), _rec("mrr", True, False),
            _rec("mrr", False, False), _rec("mrr", False, False)]
    s = summarize(recs)
    assert s["traps"]["on"] == {"exec_match_rate": 0.5, "matched": 1, "n": 2}
    assert s["traps"]["off"] == {"exec_match_rate": 0.0, "matched": 0, "n": 2}


def test_controls_diverging_ids_flag_asymmetry():
    on = EvalRecord(id="c1", category="control", question="q", semantic_layer=True,
                    retrieval_config="lexical", predicted_sql="s", gold_sql="g",
                    exec_match=True, sql_valid_first_try=True, sql_valid_final=True,
                    repair_attempts=0, latency_ms=1.0, prompt_tokens=1, completion_tokens=1)
    off = EvalRecord(id="c1", category="control", question="q", semantic_layer=False,
                     retrieval_config="lexical", predicted_sql="s", gold_sql="g",
                     exec_match=False, sql_valid_first_try=True, sql_valid_final=True,
                     repair_attempts=0, latency_ms=1.0, prompt_tokens=1, completion_tokens=1)
    s = summarize([on, off])
    assert s["controls"]["diverging_ids"] == ["c1"]


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
