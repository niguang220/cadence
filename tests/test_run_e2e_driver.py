"""The e2e driver assembles records + summary + provenance (service-free)."""
from conftest import PlanningFakeModel

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
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
    out = run_e2e(db, tables, cases, PlanningFakeModel(gold), repeats=2, configs=(False, True))
    # 1 case x 2 configs x 2 repeats = 4 records
    assert len(out["records"]) == 4
    assert out["repeats"] == 2 and out["configs"] == [False, True]
    assert out["summary"]["n_records"] == 4


def test_build_report_stamps_provenance(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    tables = introspect(db)
    gold = "SELECT COUNT(*) FROM subscription"
    cases = [_one_control_case()]
    report = build_report(db, tables, cases, PlanningFakeModel(gold), model_name="fake",
                          repeats=1)
    assert report["measured"] is True
    assert report["model"] == "fake"
    assert len(report["golden_sha256"]) == 64      # hex sha-256 of saas_metrics.json
    assert "timestamp" in report and report["summary"]["n_records"] == 2
