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
from evals.value_e2e import (_CONFIGS, _frozen_sha, _SELECTED_CONTROL_IDS, _SELECTED_PRIMARY_IDS,
                             build_report, config_provenance, negative_safety, run_value_e2e,
                             summarize)
from conftest import PlanningFakeModel


def _setup(tmp_path):
    db = build(tmp_path / "v.db")
    return db, introspect(db)


def test_config_matrix_is_the_four_stage3b_configs():
    assert [f().name for f in _CONFIGS.values()] == [
        "current_hybrid", "rrf_hybrid", "value_ablation", "dense_value"]


def test_config_provenance_is_canonical_and_named():
    prov = config_provenance()
    assert [c["name"] for c in prov] == ["current_hybrid", "rrf_hybrid", "value_ablation", "dense_value"]
    # current_hybrid is the byte-identical legacy control; dense_value is RRF + shortest_path + value
    ch = next(c for c in prov if c["name"] == "current_hybrid")
    dv = next(c for c in prov if c["name"] == "dense_value")
    assert ch["fusion"] == "legacy_minmax" and ch["relation_strategy"] == "legacy_one_hop"
    assert dv["fusion"] == "rrf" and dv["value_backend"] == "es"


def test_summarize_pairs_dense_value_against_the_other_three():
    records = [
        # dense_value beats current_hybrid on case A (2/2 vs 0/2), ties on B (1/2 vs 1/2)
        {"kind": "primary", "config": "dense_value", "case": "A", "exec_match": True},
        {"kind": "primary", "config": "dense_value", "case": "A", "exec_match": True},
        {"kind": "primary", "config": "current_hybrid", "case": "A", "exec_match": False},
        {"kind": "primary", "config": "current_hybrid", "case": "A", "exec_match": False},
        {"kind": "primary", "config": "dense_value", "case": "B", "exec_match": True},
        {"kind": "primary", "config": "dense_value", "case": "B", "exec_match": False},
        {"kind": "primary", "config": "current_hybrid", "case": "B", "exec_match": True},
        {"kind": "primary", "config": "current_hybrid", "case": "B", "exec_match": False},
    ]
    s = summarize({"records": records})
    p = s["dense_value_vs_current_hybrid"]
    assert p == {"a": "dense_value", "b": "current_hybrid", "metric": "exec_match",
                 "wins": 1, "losses": 0, "ties": 1,
                 "per_case": {"A": {"dense_value": "2/2", "current_hybrid": "0/2"},
                              "B": {"dense_value": "1/2", "current_hybrid": "1/2"}}}
    assert set(s) == {"dense_value_vs_current_hybrid", "dense_value_vs_rrf_hybrid",
                      "dense_value_vs_value_ablation"}


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
    assert len(rep["frozen_config_sha256"]) == 64                     # configs frozen too
    assert [c["name"] for c in rep["config_provenance"]] == [
        "current_hybrid", "rrf_hybrid", "value_ablation", "dense_value"]
    assert rep["configs"] == ["current_hybrid", "rrf_hybrid", "value_ablation", "dense_value"]
    assert set(rep["summary"]) == {"dense_value_vs_current_hybrid", "dense_value_vs_rrf_hybrid",
                                   "dense_value_vs_value_ablation"}
    assert rep["selected_primaries"] == list(_SELECTED_PRIMARY_IDS)


def test_build_report_honors_case_id_subset(tmp_path):
    db, tables = _setup(tmp_path)
    rep = build_report(db, tables, load_value_linking(), PlanningFakeModel("SELECT 1"),
                       FakeValueBackend(), model_name="fake", repeats=1, concurrency=1,
                       primary_ids=("zh_shyuntu_tickets", "zh_tianhe_contracts"), control_ids=())
    # 2 primaries x 4 configs x 1 repeat = 8; no controls
    assert rep["n_primary"] == 8 and rep["n_control"] == 0
    assert rep["selected_primaries"] == ["zh_shyuntu_tickets", "zh_tianhe_contracts"]
    assert rep["selected_controls"] == []
    assert {r["case"] for r in rep["records"]} == {"zh_shyuntu_tickets", "zh_tianhe_contracts"}


def test_build_report_rejects_unknown_case_id(tmp_path):
    import pytest
    db, tables = _setup(tmp_path)
    with pytest.raises(KeyError):
        build_report(db, tables, load_value_linking(), PlanningFakeModel("SELECT 1"),
                     FakeValueBackend(), model_name="fake", repeats=1, concurrency=1,
                     primary_ids=("does_not_exist",), control_ids=())


def test_current_hybrid_legacy_path_has_defined_ordered_candidates_and_fusion_at_5(tmp_path,
                                                                                    monkeypatch):
    """The legacy_minmax config (current_hybrid) still yields an ordered candidate list and a defined
    Fusion@5 — the metric ranks on the config's OWN fusion_rank (min-max hybrid position), so it is
    comparable to the RRF configs. Deterministic (dense zeroed)."""
    import agent.hybrid_retriever as hr
    from agent.retrieval.contracts import RetrievalConfig
    from agent.retrieval.pipeline import run_retrieval
    from evalharness.value_metrics import rank_sensitive_metrics

    class _ZeroIndex:
        def __init__(self, tables):
            pass

        def table_scores(self, q):
            return {}

    monkeypatch.setattr(hr, "SemanticIndex", _ZeroIndex)
    hr._INDEX_CACHE.clear()
    _, tables = _setup(tmp_path)
    rr = run_retrieval("list every ticket and its company", tables,
                       RetrievalConfig.current_hybrid(), k=5)
    m = rank_sensitive_metrics(rr, ["ticket"])
    ordered = m["candidate_tables_ordered"]
    assert ordered and "ticket" in ordered                     # legacy hybrid produces & recalls
    ranks = [c.fusion_rank for c in rr.candidates]
    assert sorted(ranks) == list(range(1, len(ranks) + 1))     # fusion_rank = 1-based hybrid position
    assert m["candidate_recall"] is not None                   # defined, not crashing on the legacy path
    assert m["fusion_at_5_recall"] == (1.0 if "ticket" in ordered[:5] else 0.0)
    hr._INDEX_CACHE.clear()


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
