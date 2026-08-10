"""Concurrency in the e2e driver: parallel workers, but output stays deterministically ordered
(repeat -> semantic-layer -> case), repeat_index/ON-OFF pairing never misaligns, and a worker
exception propagates (never a silently dropped record). Service-free (PlanningFakeModel)."""
import pytest
from conftest import PlanningFakeModel

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.retrieval.contracts import RetrievalConfig
from evals.run_e2e import parse_args, run_e2e
from evalharness.golden import SaasMetricsCase


def _cases():
    return [
        SaasMetricsCase(id="cnt", category="control", metric="count",
                        question="how many subscriptions?",
                        gold_sql="SELECT COUNT(*) FROM subscription",
                        required_tables=["subscription"]),
        SaasMetricsCase(id="cnt2", category="control", metric="count",
                        question="how many accounts?",
                        gold_sql="SELECT COUNT(*) FROM account",
                        required_tables=["account"]),
    ]


def _setup(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    return db, introspect(db)


def _keys(out):
    return [(r["repeat_index"], r["semantic_layer"], r["id"]) for r in out["records"]]


def test_concurrency_matches_serial_order_and_pairing(tmp_path):
    db, tables = _setup(tmp_path)
    cfg = RetrievalConfig.lexical_baseline()
    m = PlanningFakeModel("SELECT COUNT(*) FROM subscription")
    serial = run_e2e(db, tables, _cases(), m, config=cfg, k=5, repeats=3,
                     semantic_layers=(False, True), concurrency=1)
    parallel = run_e2e(db, tables, _cases(), m, config=cfg, k=5, repeats=3,
                       semantic_layers=(False, True), concurrency=4)
    expected = [(ri, sl, cid) for ri in range(3) for sl in (False, True)
                for cid in ("cnt", "cnt2")]
    assert _keys(serial) == expected            # serial baseline is repeat -> semantic -> case
    assert _keys(parallel) == expected          # concurrency preserves that exact order + pairing
    assert len(parallel["records"]) == 12


def test_high_concurrency_is_clamped_but_correct(tmp_path):
    db, tables = _setup(tmp_path)
    cfg = RetrievalConfig.lexical_baseline()
    m = PlanningFakeModel("SELECT COUNT(*) FROM subscription")
    serial = run_e2e(db, tables, _cases(), m, config=cfg, repeats=2, concurrency=1)
    huge = run_e2e(db, tables, _cases(), m, config=cfg, repeats=2, concurrency=999)
    assert _keys(huge) == _keys(serial)


def test_worker_exception_propagates_no_dropped_record(tmp_path):
    db, tables = _setup(tmp_path)
    bad = [SaasMetricsCase(id="bad", category="control", metric="x", question="q",
                           gold_sql="SELECT * FROM no_such_table",
                           required_tables=["subscription"])]
    # run_case raises (gold_sql fails the fixture self-check); the pool must re-raise, not swallow.
    with pytest.raises(Exception):
        run_e2e(db, tables, bad, PlanningFakeModel("SELECT 1"),
                config=RetrievalConfig.lexical_baseline(), repeats=2,
                semantic_layers=(False,), concurrency=4)


def test_concurrency_below_one_rejected(tmp_path):
    db, tables = _setup(tmp_path)
    with pytest.raises(ValueError):
        run_e2e(db, tables, _cases(), PlanningFakeModel("SELECT 1"),
                config=RetrievalConfig.lexical_baseline(), concurrency=0)


def test_parse_args_concurrency_default_and_override():
    assert parse_args([]).concurrency == 1
    assert parse_args(["--concurrency", "4"]).concurrency == 4
