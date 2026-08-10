"""Full-agent value E2E driver MECHANICS (fakes only — NOT official numbers).

Proves the driver runs the 4-config x positive x repeat matrix, scores exec_match against gold_sql,
records the full-agent + retrieval latency / recall / token / event fields, and runs the negatives
through the deterministic safety acceptance. Official numbers require a REAL ES backend + real model
(see value_e2e.main); this test only exercises the plumbing with a fake backend and fake model."""
from agent.db.build_value_db import build
from agent.db.introspect import introspect
from agent.retrieval.value_backend import FakeValueBackend
from evalharness.golden import load_value_linking
from evals.value_e2e import negative_safety, run_value_e2e
from conftest import PlanningFakeModel


def _setup(tmp_path):
    db = build(tmp_path / "v.db")
    return db, introspect(db)


def test_e2e_mechanics_runs_matrix_and_scores(tmp_path):
    db, tables = _setup(tmp_path)
    positives = [c for c in load_value_linking() if c.id == "en_globex_tickets"]
    negatives = [c for c in load_value_linking() if not c.expect_value_hit]
    model = PlanningFakeModel(positives[0].gold_sql)   # returns the gold SQL -> exec_match True
    out = run_value_e2e(db, tables, positives, negatives, model, FakeValueBackend(),
                        repeats=1, concurrency=1)
    assert out["n_positive_runs"] == 4                 # 4 configs x 1 case x 1 repeat
    assert {r["config"] for r in out["positive_records"]} == {
        "lexical_baseline", "value_ablation", "rrf_hybrid", "dense_value"}
    rec = out["positive_records"][0]
    for field in ("exec_match", "no_sql", "answer_mismatch", "agent_latency_ms",
                  "retrieval_latency_ms", "candidate_recall", "fusion_at_5_recall",
                  "prompt_tokens", "value_degraded", "admission_rejected"):
        assert field in rec
    assert all(r["exec_match"] for r in out["positive_records"])   # fake returns gold on every config


def test_e2e_negatives_are_safe(tmp_path):
    db, tables = _setup(tmp_path)
    negatives = [c for c in load_value_linking() if not c.expect_value_hit]
    safety = negative_safety(tables, negatives, FakeValueBackend())
    assert {s["case"] for s in safety} == {c.id for c in negatives}
    assert all(s["safe"] and not s["admitting_value_hit"] and not s["pii_touched"] for s in safety)


def test_e2e_records_do_not_leak_raw_values(tmp_path):
    import json
    db, tables = _setup(tmp_path)
    positives = [c for c in load_value_linking() if c.id == "en_globex_tickets"]
    model = PlanningFakeModel(positives[0].gold_sql)
    out = run_value_e2e(db, tables, positives, [], model, FakeValueBackend(),
                        repeats=1, concurrency=1)
    blob = json.dumps(out, ensure_ascii=False)
    assert "Globex Corporation" not in blob             # records store tables/tiers/ranks, not values
