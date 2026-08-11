import re
import pytest

import agent.graph as graph
import agent.hybrid_retriever as hr
from agent.db.build_demo_db import build
from agent.db.build_saas_db import build as build_saas
from agent.db.introspect import introspect, render_schema, expand_with_fk_neighbors
from agent.hybrid_retriever import retrieve
from agent.schema_relations import join_paths
from agent.graph import _schema_recall, _table_relation
from agent.semantic_layer import MetricRegistry, MetricRetrievalHit
from agent.execution import ExecutionResult
from agent.generation import AnswerResult
from agent.retrieval.contracts import (RelationPlan, RetrievalConfig, RetrievalResult,
                                        SelectionDecision, TableCandidate)
from agent.retrieval.pipeline import run_retrieval
from agent.retrieval.relation import plan_relations
from agent.retrieval.selector import fallback_topk, validate_structured_selection
from agent.retrieval.serde import deserialize_result, serialize_config
from evalharness.e2e_eval import record_run
from evalharness.golden import SaasMetricsCase


class _FakeDense:
    def __init__(self, rows): self._rows = rows
    def column_scores(self, q, tables): return self._rows


class _ZeroIndex:
    def __init__(self, tables): pass
    def table_scores(self, q): return {}


@pytest.fixture
def det_index(monkeypatch):
    monkeypatch.setattr(hr, "SemanticIndex", _ZeroIndex)   # dense contributes nothing -> deterministic lexical
    hr._INDEX_CACHE.clear()
    yield
    hr._INDEX_CACHE.clear()


def _demo(tmp_path): return introspect(build(tmp_path / "t.db"))
def _saas(tmp_path): return introspect(build_saas(tmp_path / "s.db"))


def test_accept_1_current_hybrid_parity_vs_legacy_oracle(det_index, tmp_path):
    tables = _demo(tmp_path)
    q = "total sales by billing country"
    top_k = retrieve(q, tables, k=5)                                  # legacy oracle
    expected = render_schema(tables, only=sorted(expand_with_fk_neighbors(tables, top_k)))
    paths = join_paths(tables, top_k)
    if paths:
        hint = "\n".join(f"{p['from']}.{p['on']} = {p['to']}.{p['ref_on']}" for p in paths)
        expected = f"{expected}\n\nJoin paths:\n{hint}"
    state = {"question": q, "tables": tables, "k": 5,
             "retrieval_config_serialized": serialize_config(RetrievalConfig.current_hybrid())}
    s1 = _schema_recall(state)
    assert s1["retrieved_tables"] == top_k                           # candidate order + retrieved_tables
    s2 = _table_relation({**state, **s1})
    assert s2.get("schema", s1["schema"]) == expected                # complete schema + join-hint string
    # non-default runtime k matches the legacy oracle
    top3 = retrieve(q, tables, k=3)
    s_k3 = _schema_recall({**state, "k": 3})
    assert s_k3["retrieved_tables"] == top3
    # off-topic -> refusal (empty retrieved_tables + empty schema)
    s_off = _schema_recall({"question": "what is the weather today", "tables": tables, "k": 5,
                            "retrieval_config_serialized": serialize_config(RetrievalConfig.current_hybrid())})
    assert s_off["retrieved_tables"] == [] and s_off["schema"] == ""


def test_accept_2_determinism_channel_fusion_ranks_and_scores(tmp_path):
    tables = _demo(tmp_path)
    fake = _FakeDense([("track", "name", 0.9), ("genre", "name", 0.3), ("track", "title", 0.5)])

    def fingerprint():
        res = run_retrieval("how many tracks are in each genre", tables,
                            RetrievalConfig.rrf_hybrid(), k=5, dense_backend=fake)
        return [(c.table, c.fusion_rank, c.fusion_score,
                 tuple(sorted((ch, ctr.channel_rank) for ch, ctr in c.channel_results.items())))
                for c in res.candidates]

    a, b = fingerprint(), fingerprint()
    assert a == b            # identical table order, fusion_rank, fusion_score, and per-channel channel_rank


def test_accept_3_illegal_selection_falls_back_with_event():
    cands = [TableCandidate(t, {}, 1.0 / (i + 1), i + 1) for i, t in enumerate("abcdef")]
    assert validate_structured_selection([], cands) is None
    assert validate_structured_selection(["zzz"], cands) is None
    assert validate_structured_selection("not-json", cands) is None    # non-list input -> fallback
    dec, ev = fallback_topk(cands, protected=["z"], context_anchor_k=5)
    assert dec.selector == "topk" and dec.anchor_tables[:5] == ["a", "b", "c", "d", "e"]
    assert "z" in dec.anchor_tables
    assert ev.stage == "selection" and ev.event == "selector_fallback"


def test_accept_4_protected_anchor_survives_topk_and_legacy_telemetry(det_index, tmp_path):
    saas = _saas(tmp_path)
    mrr = next(m for m in MetricRegistry.load().metrics if m.name == "mrr")
    hit = MetricRetrievalHit(mrr, "alias", 1.0)
    # 7 dense candidates, account scored LAST -> fusion_rank > 5 (TopKSelector would drop it).
    dense = _FakeDense([("subscription", "c", 0.9), ("invoice", "c", 0.8), ("plan", "c", 0.7),
                        ("mrr_movement", "c", 0.6), ("activity_event", "c", 0.5), ("user", "c", 0.4),
                        ("account", "c", 0.1)])
    # 'zzz nonsense' has no lexical footing; the alias metric hit admits it and protects account+subscription.
    rrf = run_retrieval("zzz nonsense", saas, RetrievalConfig.rrf_hybrid(), k=5,
                        metric_hits=[hit], dense_backend=dense)
    assert rrf.selection.selector == "topk"                        # >5 candidates -> TopK, not NoOp
    acc = next(c for c in rrf.candidates if c.table == "account")
    assert acc.fusion_rank > 5                                     # outside Fusion top-5
    assert "account" in rrf.selection.anchor_tables                # protected -> kept
    assert "account" not in rrf.selection.dropped_tables
    assert "user" in rrf.selection.dropped_tables                  # unprotected rank-6 -> pruned by TopK
    # legacy control: metric is telemetry only; account is NOT force-added.
    legacy = run_retrieval("how many active subscriptions", saas, RetrievalConfig.current_hybrid(),
                           k=3, metric_hits=[hit])
    assert legacy.metric_matches and legacy.metric_matches[0].metric == "mrr"
    assert "account" not in legacy.selection.anchor_tables


def test_accept_5_bridges_only_from_planner_and_renderer_pure(tmp_path):
    tables = _saas(tmp_path)
    plan = plan_relations(tables, ["account", "revenue_recognition"], max_hops=3)
    assert plan.bridges == ["invoice"]
    assert plan.context_tables == ["account", "invoice", "revenue_recognition"]
    assert plan.unconnected_anchors == []
    edges = {(e.from_table, e.from_column, e.to_table, e.to_column) for e in plan.edges}
    assert ("invoice", "account_id", "account", "account_id") in edges
    assert ("revenue_recognition", "invoice_id", "invoice", "invoice_id") in edges
    # renderer is pure: given only=[revenue_recognition] it must NOT render invoice as a table
    rendered = render_schema(tables, only=["revenue_recognition"])
    assert "TABLE revenue_recognition (" in rendered and "TABLE invoice (" not in rendered


def _rendered_tables(schema):
    return set(re.findall(r"TABLE (\w+)", schema))


def test_accept_6_graph_renders_context_tables_both_strategies(det_index, tmp_path, monkeypatch):
    tables = _demo(tmp_path)
    q = "total sales by billing country"
    # legacy_one_hop via the real graph node (det_index -> deterministic retrieve)
    state_l = {"question": q, "tables": tables, "k": 5,
               "retrieval_config_serialized": serialize_config(RetrievalConfig.current_hybrid())}
    s_l = _schema_recall(state_l)
    plan_l = deserialize_result(s_l["retrieval_result_serialized"]).relation_plan
    assert plan_l.strategy == "legacy_one_hop"
    assert _rendered_tables(s_l["schema"]) == set(plan_l.context_tables)
    # shortest_path via the REAL pipeline with a deterministic dense backend (no fastembed)
    real = graph.run_retrieval
    monkeypatch.setattr(graph, "run_retrieval",
        lambda qq, tt, cc, *, k, metric_hits=None, value_backend=None, value_query=None: real(
            qq, tt, cc, k=k, metric_hits=metric_hits, value_query=value_query,
            dense_backend=_FakeDense([("track", "name", 0.9), ("genre", "name", 0.4)])))
    state_r = {"question": q, "tables": tables, "k": 5,
               "retrieval_config_serialized": serialize_config(RetrievalConfig.rrf_hybrid())}
    s_r = _schema_recall(state_r)
    plan_r = deserialize_result(s_r["retrieval_result_serialized"]).relation_plan
    assert plan_r.strategy == "shortest_path"
    assert _rendered_tables(s_r["schema"]) == set(plan_r.context_tables)


def test_accept_7_lexical_baseline_provenance_in_record():
    cfg = RetrievalConfig.lexical_baseline()
    tables = ("subscription",)
    rr = RetrievalResult(config_name=cfg.name, signals=[],
        candidates=[TableCandidate("subscription", {}, None, 1)], metric_matches=[],
        selection=SelectionDecision(["subscription"], [], "topk", {}),
        relation_plan=RelationPlan("shortest_path", ["subscription"], [], ["subscription"], [], [], []),
        stage_events=[])
    result = AnswerResult(question="q", retrieved_tables=["subscription"], sql="SELECT 1",
        execution=ExecutionResult(True, columns=["c"], rows=[(1,)]), answer="a",
        trace=[], usage={}, retrieval_result=rr)
    case = SaasMetricsCase(id="c", category="control", metric="m", question="q",
                           gold_sql="SELECT 1", required_tables=["subscription"])
    rec = record_run(case, result, [(1,)], semantic_layer=False, config=cfg, k=5, repeat_index=0)
    prov = rec.retrieval_config                                       # the STORED provenance, not the preset
    assert prov["name"] == "lexical_baseline"
    assert prov["dense_backend"] is None and prov["value_backend"] is None and prov["selector"] is None
    assert prov["lexical"] is True
