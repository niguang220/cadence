"""The consistency surface: scorer units (fake outcomes) + deterministic CI teeth.

The scorer is pure -- catch-rate over adversarial, false-positive-rate over clean, with
support. The teeth run on the real SaaS DB with NO LLM: they prove every case executes
successfully and is governance-clean, that each adversarial candidate genuinely diverges
from its gold (a failed candidate faking divergence is caught), and that each clean
candidate matches gold. This is what stops a fixture from having no teeth; the measured
catch-rate itself is produced only by the manual driver.
"""
from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from evalharness.consistency_eval import (
    ConsistencyOutcome, diverges, execute_case, governance_clean, score_consistency,
)
from evalharness.golden import load_consistency


def test_score_consistency_reports_both_rates():
    outcomes = [
        ConsistencyOutcome("a1", expected_caught=True, caught=True),
        ConsistencyOutcome("a2", expected_caught=True, caught=False),
        ConsistencyOutcome("c1", expected_caught=False, caught=False),
        ConsistencyOutcome("c2", expected_caught=False, caught=True),
    ]
    s = score_consistency(outcomes)
    assert s["catch_rate"] == 0.5 and s["adversarial_support"] == 2
    assert s["fp_rate"] == 0.5 and s["clean_support"] == 2


def test_every_fixture_has_teeth_and_executes_clean(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    tables = introspect(db)
    for case in load_consistency():
        cand, gold = execute_case(case, db, tables)
        assert cand.ok and gold.ok, f"{case.id}: a SQL failed to execute"
        assert governance_clean(cand, tables) and governance_clean(gold, tables), f"{case.id}: governance-blocked"
        if case.expected_caught:
            assert diverges(cand, gold), f"{case.id}: adversarial candidate does NOT diverge (no teeth)"
        else:
            assert not diverges(cand, gold), f"{case.id}: clean candidate does NOT match gold (mislabeled)"
