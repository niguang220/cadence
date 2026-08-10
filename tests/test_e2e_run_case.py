"""run_case wires gold execution + a full agent run into one EvalRecord (service-free)."""
import pytest
from conftest import PlanningFakeModel

from agent.db.introspect import introspect
from agent.execution import ExecutionResult
from agent.generation import AnswerResult
from agent.retrieval.contracts import (RelationPlan, RetrievalConfig, RetrievalResult,
                                        SelectionDecision, TableCandidate)
from evalharness.e2e_eval import run_case
from evalharness.golden import SaasMetricsCase


def _tables(saas_db):
    return introspect(saas_db)


def _fake_answer_with_rr(config_name, *, sql="SELECT 42", rows=((42,),), ok=True):
    tables = ["subscription"]
    rr = RetrievalResult(config_name=config_name, signals=[],
        candidates=[TableCandidate(t, {}, None, i + 1) for i, t in enumerate(tables)],
        metric_matches=[], selection=SelectionDecision(tables, [], "topk", {}),
        relation_plan=RelationPlan("legacy_one_hop", tables, [], tables, [], [], []),
        stage_events=[])
    return AnswerResult(
        question="q?", retrieved_tables=tables, sql=sql,
        execution=ExecutionResult(ok, columns=["c"], rows=list(rows)) if ok
                  else ExecutionResult(False, error="boom"),
        answer="a", trace=[], usage={}, retrieval_result=rr)


def test_run_case_matches_gold_when_model_returns_gold_sql(saas_db):
    # A subscription count that the gold repeats -> exec_match must be True.
    gold = "SELECT COUNT(*) FROM subscription"
    case = SaasMetricsCase(id="cnt", category="control", metric="count",
                           question="how many subscriptions?", gold_sql=gold,
                           required_tables=["subscription"])
    model = PlanningFakeModel(gold)
    rec = run_case(saas_db, _tables(saas_db), case, model, semantic_layer=False,
                   config=RetrievalConfig.current_hybrid(), k=5, repeat_index=0)
    assert rec.exec_match is True and rec.sql_valid_final is True
    assert rec.id == "cnt" and rec.semantic_layer is False


def test_run_case_mismatch_when_model_returns_wrong_sql(saas_db):
    case = SaasMetricsCase(id="cnt", category="control", metric="count",
                           question="how many subscriptions?",
                           gold_sql="SELECT COUNT(*) FROM subscription",
                           required_tables=["subscription"])
    model = PlanningFakeModel("SELECT COUNT(*) FROM account")   # different number
    rec = run_case(saas_db, _tables(saas_db), case, model, semantic_layer=False,
                   config=RetrievalConfig.current_hybrid(), k=5, repeat_index=0)
    assert rec.exec_match is False and rec.failure_stage == "answer_mismatch"


def test_run_case_raises_on_broken_gold_sql(saas_db):
    case = SaasMetricsCase(id="broken", category="mrr", metric="mrr", question="q?",
                           gold_sql="SELECT * FROM no_such_table",
                           required_tables=["subscription"])
    model = PlanningFakeModel("SELECT 1")
    with pytest.raises(RuntimeError, match="gold_sql"):
        run_case(saas_db, _tables(saas_db), case, model, semantic_layer=False,
                config=RetrievalConfig.current_hybrid(), k=5, repeat_index=0)


def test_run_case_forwards_config_and_k(saas_db, monkeypatch):
    import evalharness.e2e_eval as ee
    seen = {}
    cfg = RetrievalConfig.current_hybrid()

    def spy(db, q, *, model, tables, semantic_layer, k, retrieval_config):
        seen.update(k=k, retrieval_config=retrieval_config)
        return _fake_answer_with_rr(cfg.name)

    monkeypatch.setattr(ee, "run_agent", spy)
    tables = introspect(saas_db)
    case = SaasMetricsCase(id="cnt", category="control", metric="count",
                           question="how many subscriptions?",
                           gold_sql="SELECT COUNT(*) FROM subscription", required_tables=["subscription"])
    ee.run_case(saas_db, tables, case, object(), semantic_layer=False, config=cfg, k=7, repeat_index=2)
    assert seen["k"] == 7 and seen["retrieval_config"] is cfg
