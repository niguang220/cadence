"""RRF lexical-weight sweep — summary flip logic on synthetic records, then a deterministic real run
proving that lowering the lexical weight restores a dense-hit gold table that equal-weight RRF demoted
out of Top15 (the ARPU `plan` case). Value channel is excluded by construction (sweep_case never takes
a value backend); semantic-ON protected anchors come from the real registry."""
from evals.rrf_weight_sweep import _WEIGHTS, summarize


def _cell(r15, sel, fused):
    return {"recall": {5: 0.0, 10: 0.0, 15: r15}, "precision": {5: 0.0, 10: 0.0, 15: 0.0},
            "selection_recall": sel, "context_recall": sel, "gold_fused_ranks": fused}


def _rec(cid, cat, sem, gold, dense_ranks, cells):
    default = _cell(0.0, 0.0, {g: None for g in gold})
    by_weight = {w: cells.get(w, default) for w in _WEIGHTS}
    return {"id": cid, "category": cat, "semantic_layer": sem, "required_tables": gold,
            "lex_ranks": {}, "dense_ranks": dense_ranks, "by_weight": by_weight}


def test_summarize_flags_dense_restored_and_selection_gain():
    # account: dense rank 12; equal-weight (1.0) fused=None -> out of Top15; at w=0.25 fused=8 -> restored
    rec = _rec("cA", "mrr", False, ["account"], {"account": 12},
               {"1.0": _cell(0.0, 0.0, {"account": None}), "0.25": _cell(1.0, 1.0, {"account": 8})})
    v = summarize([rec])["off"]["0.25"]["vs_equal_weight"]
    assert v["dense_restored"] == ["cA"]
    assert v["selection_gained"] == ["cA"] and v["candidate_top15_gained"] == ["cA"]
    # the baseline compared against itself flips nothing
    assert summarize([rec])["off"]["1.0"]["vs_equal_weight"]["dense_restored"] == []


def test_summarize_flags_control_regression_at_low_weight():
    # a control fully selected at equal weight but lost when lexical weight -> 0
    rec = _rec("ctrl1", "control", False, ["account"], {"account": 2},
               {"1.0": _cell(1.0, 1.0, {"account": 1}), "0": _cell(0.0, 0.0, {"account": None})})
    v = summarize([rec])["off"]["0"]["vs_equal_weight"]
    assert v["controls_selection_lost"] == ["ctrl1"] and v["controls_selection_gained"] == []


def test_real_sweep_is_deterministic_and_restores_arpu_plan():
    import tempfile

    from agent.db.build_saas_db import build
    from agent.db.introspect import introspect
    from agent.retrieval.contracts import RetrievalConfig
    from agent.semantic_layer import MetricRegistry
    from evalharness.golden import load_saas_metrics
    from evals.rrf_weight_sweep import sweep_case
    d = tempfile.mkdtemp()
    tables = introspect(build(f"{d}/e.db", confounders=True))
    reg = MetricRegistry.load()
    cfg = RetrievalConfig.dense_value()
    arpu = next(c for c in load_saas_metrics() if c.id == "arpu_asof")
    a = sweep_case(tables, arpu, reg, cfg, semantic_layer=False)
    b = sweep_case(tables, arpu, reg, cfg, semantic_layer=False)
    assert a["by_weight"] == b["by_weight"]                    # deterministic
    assert a["dense_ranks"].get("plan") is not None            # dense DID retrieve plan
    plan_at = {w: a["by_weight"][w]["gold_fused_ranks"]["plan"] for w in _WEIGHTS}
    # equal weight demotes plan out of Top15; a lower lexical weight brings it back into candidates
    assert plan_at["1.0"] is None or plan_at["1.0"] > 15
    assert any(plan_at[w] is not None and plan_at[w] <= 15 for w in ("0", "0.1", "0.25"))
