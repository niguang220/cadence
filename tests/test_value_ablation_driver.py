"""Value-linking ablation driver: runs the golden through 4 configs (lexical / value / dense /
dense+value) and records rank-sensitive metrics. Deterministic (FakeValueBackend, no LLM/ES)."""
from agent.db.build_value_db import build
from agent.db.introspect import introspect
from evalharness.golden import load_value_linking
from evals.value_ablation import run_value_ablation


def _setup(tmp_path):
    db = build(tmp_path / "v.db")
    return introspect(db), db


def test_runs_four_configs_over_all_cases(tmp_path):
    tables, db = _setup(tmp_path)
    cases = load_value_linking()
    recs = run_value_ablation(tables, db, cases)
    assert {r["config"] for r in recs} == {"lexical_baseline", "value_ablation", "rrf_hybrid",
                                            "dense_value"}
    assert len(recs) == 4 * len(cases)


def test_value_lifts_candidate_recall_over_lexical(tmp_path):
    tables, db = _setup(tmp_path)
    cases = [c for c in load_value_linking() if c.id == "en_globex_tickets"]
    by = {r["config"]: r for r in run_value_ablation(tables, db, cases)}
    assert by["value_ablation"]["value_hit"] is True        # value links the company
    assert by["lexical_baseline"]["value_hit"] is False      # lexical cannot
    assert by["value_ablation"]["candidate_recall"] >= by["lexical_baseline"]["candidate_recall"]
    assert "company" in by["value_ablation"]["candidate_tables_ordered"]
    assert "company" not in by["lexical_baseline"]["candidate_tables_ordered"]


def test_negatives_never_produce_a_value_hit(tmp_path):
    tables, db = _setup(tmp_path)
    cases = [c for c in load_value_linking() if not c.expect_value_hit]
    recs = run_value_ablation(tables, db, cases)
    for r in recs:
        if r["config"] in ("value_ablation", "dense_value"):
            assert r["value_hit"] is False, f"{r['config']}/{r['case']} unexpected value hit"
