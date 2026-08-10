"""Full-agent value E2E driver MECHANICS (fakes only — NOT official numbers).

Proves the driver runs the 4-config primary + control matrix, scores exec_match against gold_sql for
primaries and refusal/no-leak for controls, records the full-agent + retrieval latency / recall /
token / event fields, and freezes the case SHA. Official numbers require a REAL ES backend + real
model (see value_e2e.main); this test only exercises the plumbing with fakes."""
import json

from agent.db.build_value_db import build
from agent.db.introspect import introspect
from agent.retrieval.value_backend import FakeValueBackend
from evalharness.golden import load_value_linking
from evals.value_e2e import (_frozen_sha, _SELECTED_CONTROL_IDS, _SELECTED_PRIMARY_IDS,
                             build_report, negative_safety, run_value_e2e)
from conftest import PlanningFakeModel


def _setup(tmp_path):
    db = build(tmp_path / "v.db")
    return db, introspect(db)


def test_selection_is_ten_primaries_and_four_controls():
    by = {c.id: c for c in load_value_linking()}
    assert len(_SELECTED_PRIMARY_IDS) == 10 and len(_SELECTED_CONTROL_IDS) == 4
    assert all(by[i].role == "primary" for i in _SELECTED_PRIMARY_IDS)
    assert {by[i].category for i in _SELECTED_PRIMARY_IDS} == {"en", "zh", "code", "homonym"}
    assert all(by[i].role == "negative" for i in _SELECTED_CONTROL_IDS)


def test_e2e_mechanics_runs_primary_and_control_matrix(tmp_path):
    db, tables = _setup(tmp_path)
    by = {c.id: c for c in load_value_linking()}
    primaries = [by["en_globex_tickets"]]
    controls = [by["off_topic_weather"]]
    model = PlanningFakeModel(primaries[0].gold_sql)   # returns gold -> exec_match True
    out = run_value_e2e(db, tables, primaries, controls, model, FakeValueBackend(),
                        repeats=1, concurrency=1)
    assert out["n_records"] == 8 and out["n_primary"] == 4 and out["n_control"] == 4
    prim = next(r for r in out["records"] if r["kind"] == "primary")
    for f in ("exec_match", "no_sql", "clarified", "answer_mismatch", "agent_latency_ms",
              "retrieval_latency_ms", "candidate_recall", "prompt_tokens", "value_degraded"):
        assert f in prim
    ctrl = next(r for r in out["records"] if r["kind"] == "control")
    for f in ("refused_or_clarified", "pii_leaked", "agent_latency_ms"):
        assert f in ctrl
    assert all(r["exec_match"] for r in out["records"] if r["kind"] == "primary")


def test_build_report_uses_frozen_selection_and_sha(tmp_path):
    db, tables = _setup(tmp_path)
    rep = build_report(db, tables, load_value_linking(), PlanningFakeModel("SELECT 1"),
                       FakeValueBackend(), model_name="fake", repeats=1, concurrency=1)
    # repeats=1: 10 primary x 4 configs = 40, 4 control x 4 configs = 16 -> 56 records
    assert rep["n_primary"] == 40 and rep["n_control"] == 16 and rep["n_records"] == 56
    assert len({r["case"] for r in rep["records"] if r["kind"] == "primary"}) == 10
    assert len({r["case"] for r in rep["records"] if r["kind"] == "control"}) == 4
    assert len(rep["frozen_case_sha256"]) == 64
    assert rep["selected_primaries"] == list(_SELECTED_PRIMARY_IDS)


def test_control_deterministic_safety_helper(tmp_path):
    db, tables = _setup(tmp_path)
    controls = [c for c in load_value_linking() if c.id in _SELECTED_CONTROL_IDS]
    safety = negative_safety(tables, controls, FakeValueBackend())
    assert all(s["safe"] and not s["pii_touched"] for s in safety)


def test_records_do_not_leak_raw_values(tmp_path):
    db, tables = _setup(tmp_path)
    by = {c.id: c for c in load_value_linking()}
    out = run_value_e2e(db, tables, [by["en_globex_tickets"]], [], PlanningFakeModel("SELECT 1"),
                        FakeValueBackend(), repeats=1, concurrency=1)
    blob = json.dumps(out, ensure_ascii=False)
    assert "Globex Corporation" not in blob and "@globex.com" not in blob


def test_frozen_sha_is_stable_and_content_sensitive():
    cases = load_value_linking()[:3]
    assert _frozen_sha(cases) == _frozen_sha(cases)
    assert _frozen_sha(cases) != _frozen_sha(load_value_linking()[:4])
