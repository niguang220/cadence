"""run_case wires gold execution + a full agent run into one EvalRecord (service-free)."""
import pytest
from conftest import PlanningFakeModel

from agent.db.introspect import introspect
from evalharness.e2e_eval import run_case
from evalharness.golden import SaasMetricsCase


def _tables(saas_db):
    return introspect(saas_db)


def test_run_case_matches_gold_when_model_returns_gold_sql(saas_db):
    # A subscription count that the gold repeats -> exec_match must be True.
    gold = "SELECT COUNT(*) FROM subscription"
    case = SaasMetricsCase(id="cnt", category="control", metric="count",
                           question="how many subscriptions?", gold_sql=gold,
                           required_tables=["subscription"])
    model = PlanningFakeModel(gold)
    rec = run_case(saas_db, _tables(saas_db), case, model, semantic_layer=False)
    assert rec.exec_match is True and rec.sql_valid_final is True
    assert rec.id == "cnt" and rec.semantic_layer is False


def test_run_case_mismatch_when_model_returns_wrong_sql(saas_db):
    case = SaasMetricsCase(id="cnt", category="control", metric="count",
                           question="how many subscriptions?",
                           gold_sql="SELECT COUNT(*) FROM subscription",
                           required_tables=["subscription"])
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")   # different number
    rec = run_case(saas_db, _tables(saas_db), case, model, semantic_layer=False)
    assert rec.exec_match is False and rec.failure_stage == "answer_mismatch"


def test_run_case_raises_on_broken_gold_sql(saas_db):
    case = SaasMetricsCase(id="broken", category="mrr", metric="mrr", question="q?",
                           gold_sql="SELECT * FROM no_such_table",
                           required_tables=["subscription"])
    model = PlanningFakeModel("SELECT 1")
    with pytest.raises(RuntimeError, match="gold_sql"):
        run_case(saas_db, _tables(saas_db), case, model, semantic_layer=False)
