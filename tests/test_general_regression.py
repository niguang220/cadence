"""Stage 3C general-mix regression driver + summary (service-free).

Summary logic is exercised on hand-built EvalRecords (exact arithmetic, no agent); the driver mechanics
run through the real graph with fakes (PlanningFakeModel + FakeValueBackend + a zeroed dense index)."""
import agent.hybrid_retriever as hr
from evalharness.e2e_eval import EvalRecord
from evalharness.golden import load_saas_metrics
from evals.general_regression import (build_report, config_provenance, run_general_regression,
                                      summarize)


def _rec(cid, category, config, sem, rep, match, *, failure_stage=None, latency=100.0,
         pt=10, ct=5, value_tier=None, cand=1.0, sel=1.0, ctx=1.0, sql_valid_final=None,
         events=()):
    return EvalRecord(
        id=cid, category=category, question="q", semantic_layer=sem,
        retrieval_config={"name": config}, retrieval_k=5, repeat_index=rep,
        predicted_sql="SELECT 1", gold_sql="SELECT 1", exec_match=match,
        sql_valid_first_try=match, sql_valid_final=match if sql_valid_final is None else sql_valid_final,
        repair_attempts=0, latency_ms=latency, prompt_tokens=pt, completion_tokens=ct,
        retrieved_tables=[], gold_tables=["t"],
        failure_stage=failure_stage if failure_stage is not None else (None if match else "answer_mismatch"),
        candidate_recall=cand, selection_recall=sel, context_recall=ctx,
        retrieval_stage_events=list(events), value_hit=value_tier is not None, value_hit_tier=value_tier)


def _cell(cid, category, config, sem, matched, n=5, **kw):
    return [_rec(cid, category, config, sem, i, i < matched, **kw) for i in range(n)]


def _synthetic_off():
    recs = []
    # traps: a dv win, a clean loss (dv<=ch-3), a soft loss, a tie
    recs += _cell("trap_win", "mrr", "current_hybrid", False, 2) + _cell("trap_win", "mrr", "dense_value", False, 5)
    recs += _cell("trap_clean", "arr", "current_hybrid", False, 5) + _cell("trap_clean", "arr", "dense_value", False, 1)
    recs += _cell("trap_soft", "arpu", "current_hybrid", False, 5) + _cell("trap_soft", "arpu", "dense_value", False, 4)
    recs += _cell("trap_tie", "mrr", "current_hybrid", False, 3) + _cell("trap_tie", "mrr", "dense_value", False, 3)
    # controls: one stable, one where dense_value regresses
    recs += _cell("ctrl_ok", "control", "current_hybrid", False, 5) + _cell("ctrl_ok", "control", "dense_value", False, 5)
    recs += _cell("ctrl_reg", "control", "current_hybrid", False, 5) + _cell("ctrl_reg", "control", "dense_value", False, 3)
    return recs


def test_paired_wins_losses_ties_and_clean_loss():
    s = summarize(_synthetic_off())
    t = s["by_mode"]["off"]["traps"]["paired"]
    assert (t["wins"], t["losses"], t["ties"]) == (1, 2, 1)
    assert t["clean_loss_ids"] == ["trap_clean"]                 # dv 1/5 <= ch 5/5 - 3
    assert t["regression_ids"] == ["trap_clean", "trap_soft"]
    assert s["by_mode"]["off"]["traps"]["current_hybrid"]["matched"] == 15    # 2+5+5+3
    assert s["by_mode"]["off"]["traps"]["dense_value"]["matched"] == 13       # 5+1+4+3


def test_control_diverging_ids_are_dense_value_regressions():
    s = summarize(_synthetic_off())
    assert s["by_mode"]["off"]["controls"]["diverging_ids"] == ["ctrl_reg"]
    assert s["by_mode"]["off"]["controls"]["paired"]["clean_loss_ids"] == []   # 3/5 vs 5/5 is not clean


def test_latency_tokens_three_denominators():
    recs = (
        _cell("a", "mrr", "dense_value", False, 5, latency=200.0, pt=100, ct=20) +               # answered
        _cell("b", "mrr", "dense_value", False, 0, failure_stage="no_sql", latency=10.0,          # refused
              pt=5, ct=0, sql_valid_final=False) +
        _cell("c", "mrr", "dense_value", False, 0, failure_stage="execution_error", latency=150.0,  # gen-reached, not answered
              pt=80, ct=15, sql_valid_final=False))
    lt = summarize(recs)["latency_tokens"]["dense_value"]
    assert lt["all"]["n"] == 15
    assert lt["generation_reached"]["n"] == 10                   # a (answered) + c (exec_error); b refused
    assert lt["answered"]["n"] == 5                              # only a has sql_valid_final
    assert lt["answered"]["avg_prompt_tokens"] == 100
    assert lt["all"]["latency_p50"] <= lt["generation_reached"]["latency_p50"]  # refusals drag all-p50 down


def test_value_hits_and_events_reported():
    recs = (_cell("a", "mrr", "dense_value", False, 5, value_tier="exact_phrase") +
            _cell("b", "mrr", "dense_value", False, 5, sel=0.5, cand=1.0) +          # selector dropped a gold table
            _cell("c", "mrr", "dense_value", False, 0, failure_stage="no_sql",
                  events=[{"stage": "channel", "event": "value_degraded"}]))
    s = summarize(recs)
    assert s["value_hits"]["records_with_value_hit"] == 5 and s["value_hits"]["by_tier"] == {"exact_phrase": 5}
    ev = s["events"]["dense_value"]
    assert ev["value_degraded"] == 5 and ev["selector_dropped_gold"] == 5


def test_driver_runs_full_matrix_with_provenance(saas_db, monkeypatch):
    from agent.db.introspect import introspect
    from agent.retrieval.value_backend import FakeValueBackend
    from conftest import PlanningFakeModel

    class _ZeroIndex:
        def __init__(self, tables):
            pass

        def table_scores(self, q):
            return {}

    monkeypatch.setattr(hr, "SemanticIndex", _ZeroIndex)          # deterministic, no fastembed
    hr._INDEX_CACHE.clear()
    tables = introspect(saas_db)
    cases = load_saas_metrics()[:2]                              # keep the mechanics test small
    rep = build_report(saas_db, tables, cases, PlanningFakeModel("SELECT 1"), FakeValueBackend(),
                       model_name="fake", repeats=1, concurrency=1)
    # 2 cases x 2 configs x 2 semantic modes x 1 repeat = 8
    assert rep["n_records"] == 8
    assert rep["configs"] == ["current_hybrid", "dense_value"]
    assert [c["name"] for c in rep["config_provenance"]] == ["current_hybrid", "dense_value"]
    assert len(rep["golden_sha256"]) == 64 and len(rep["frozen_config_sha256"]) == 64
    assert {r["config"] for r in rep["records"]} == {"current_hybrid", "dense_value"}
    assert {r["semantic_layer"] for r in rep["records"]} == {True, False}
    hr._INDEX_CACHE.clear()


def test_compact_records_carry_no_sql_or_question(saas_db, monkeypatch):
    import json

    from agent.db.introspect import introspect
    from agent.retrieval.value_backend import FakeValueBackend
    from conftest import PlanningFakeModel

    class _ZeroIndex:
        def __init__(self, tables):
            pass

        def table_scores(self, q):
            return {}

    monkeypatch.setattr(hr, "SemanticIndex", _ZeroIndex)
    hr._INDEX_CACHE.clear()
    tables = introspect(saas_db)
    rep = build_report(saas_db, tables, load_saas_metrics()[:1], PlanningFakeModel("SELECT 1"),
                       FakeValueBackend(), model_name="fake", repeats=1, concurrency=1)
    for r in rep["records"]:
        assert "predicted_sql" not in r and "gold_sql" not in r and "question" not in r
    # value inert by construction on saas (no searchable columns)
    assert rep["summary"]["value_hits"]["records_with_value_hit"] == 0
    hr._INDEX_CACHE.clear()


def test_config_provenance_is_the_two_configs():
    assert [c["name"] for c in config_provenance()] == ["current_hybrid", "dense_value"]
