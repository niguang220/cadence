"""Deterministic value-linking discriminative eval: the expanded set runs 4 configs, value linking
improves recall on multiple independent categories, and every negative/safety case is leak-free."""
from agent.db.build_value_db import build
from agent.db.introspect import introspect
from evalharness.golden import load_value_linking
from evals.value_discriminative import run_value_discriminative, summarize


def _setup(tmp_path):
    db = build(tmp_path / "v.db")
    return introspect(db), db


def test_runs_four_configs_over_all_linking_cases(tmp_path):
    tables, db = _setup(tmp_path)
    cases = load_value_linking()
    res = run_value_discriminative(tables, db, cases)
    linking = [c for c in cases if c.role in ("primary", "diagnostic")]
    assert len(res["records"]) == 4 * len(linking)
    assert {r["config"] for r in res["records"]} == {"lexical_baseline", "value_ablation",
                                                     "rrf_hybrid", "dense_value"}


def test_value_improves_multiple_categories_meets_gate(tmp_path):
    tables, db = _setup(tmp_path)
    s = summarize(run_value_discriminative(tables, db, load_value_linking()))
    # Stage-3A gate: value improves >= 6 primary cases across >= 3 categories
    assert s["value_improves"]["n_cases"] >= 6, s["value_improves"]
    assert s["value_improves"]["n_categories"] >= 3, s["value_improves"]


def test_value_never_regresses_below_lexical_on_primaries(tmp_path):
    tables, db = _setup(tmp_path)
    s = summarize(run_value_discriminative(tables, db, load_value_linking()))
    assert all(d >= 0 for d in s["paired_value_vs_lexical"].values()), s["paired_value_vs_lexical"]


def test_all_negatives_and_safety_are_leak_free(tmp_path):
    tables, db = _setup(tmp_path)
    s = summarize(run_value_discriminative(tables, db, load_value_linking()))["safety"]
    assert s["all_negatives_safe"] and s["all_safety_safe"]
    assert not s["any_pii_leak"] and not s["any_value_degraded"]


def test_report_has_no_raw_values_or_pii(tmp_path):
    import json
    tables, db = _setup(tmp_path)
    blob = json.dumps(run_value_discriminative(tables, db, load_value_linking()), ensure_ascii=False)
    for raw in ("Globex Corporation", "北京数据科技有限公司", "上海云图信息技术", "WGT-100",
                "@globex.com", "李伟", "DROP"):
        assert raw not in blob, raw
