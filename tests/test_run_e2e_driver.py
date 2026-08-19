"""The e2e driver assembles records + summary + provenance (service-free)."""
from conftest import PlanningFakeModel

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.serde import serialize_config
from evals.run_e2e import build_report, run_e2e
from evalharness.golden import SaasMetricsCase


def _one_control_case():
    return SaasMetricsCase(id="cnt", category="control", metric="count",
                           question="how many subscriptions?",
                           gold_sql="SELECT COUNT(*) FROM subscription",
                           required_tables=["subscription"])


def test_run_e2e_runs_every_config_and_repeat(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    tables = introspect(db)
    gold = "SELECT COUNT(*) FROM subscription"
    cases = [_one_control_case()]
    config = RetrievalConfig.lexical_baseline()
    out = run_e2e(db, tables, cases, PlanningFakeModel(gold), config=config, k=5, repeats=2,
                  semantic_layers=(False, True))
    # 1 case x 2 semantic-layer settings x 2 repeats = 4 records
    assert len(out["records"]) == 4
    assert out["repeats"] == 2 and out["semantic_layer_configs"] == [False, True]
    assert out["summary"]["n_records"] == 4
    assert out["retrieval_config"] == serialize_config(config)
    assert out["retrieval_k"] == 5
    assert {r["repeat_index"] for r in out["records"]} == {0, 1}


def test_run_e2e_loop_order_is_repeat_then_semantic_layer_then_case(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    tables = introspect(db)
    gold = "SELECT COUNT(*) FROM subscription"
    cases = [_one_control_case()]
    config = RetrievalConfig.lexical_baseline()
    out = run_e2e(db, tables, cases, PlanningFakeModel(gold), config=config, k=5, repeats=3,
                  semantic_layers=(False, True))
    assert len(out["records"]) == 3 * 2 * len(cases)
    seen = [(r["repeat_index"], r["semantic_layer"]) for r in out["records"]]
    assert seen == [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True)]
    assert all(0 <= r["repeat_index"] < 3 for r in out["records"])


def test_build_report_stamps_provenance(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    tables = introspect(db)
    gold = "SELECT COUNT(*) FROM subscription"
    cases = [_one_control_case()]
    config = RetrievalConfig.default()
    report = build_report(db, tables, cases, PlanningFakeModel(gold), model_name="fake",
                          config=config, k=5, repeats=1)
    assert report["measured"] is True
    assert report["model"] == "fake"
    assert len(report["golden_sha256"]) == 64      # hex sha-256 of saas_metrics.json
    assert "timestamp" in report and report["summary"]["n_records"] == 2
    assert report["retrieval_config"] == serialize_config(config)
    assert report["retrieval_k"] == 5
    assert report["semantic_layer_configs"] == [False, True]
