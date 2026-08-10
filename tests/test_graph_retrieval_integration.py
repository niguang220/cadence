import agent.hybrid_retriever as hr
import agent.graph as graph
from conftest import PlanningFakeModel
from agent.db.build_demo_db import build
from agent.db.introspect import introspect, render_schema, expand_with_fk_neighbors
from agent.hybrid_retriever import retrieve
from agent.schema_relations import join_paths
from agent.graph import _schema_recall, _table_relation, run_agent
from agent.pipeline import start_question_session, resume_question_session
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.serde import serialize_config


class _ZeroIndex:
    def __init__(self, tables): pass
    def table_scores(self, q): return {}


import pytest


@pytest.fixture
def det_index(monkeypatch):
    monkeypatch.setattr(hr, "SemanticIndex", _ZeroIndex)
    hr._INDEX_CACHE.clear()
    yield
    hr._INDEX_CACHE.clear()


def _tables(tmp_path):
    return introspect(build(tmp_path / "t.db"))


def _legacy_state(tables, question, k=5):
    return {"question": question, "tables": tables, "k": k,
            "retrieval_config_serialized": serialize_config(RetrievalConfig.current_hybrid())}


def test_current_hybrid_schema_and_join_hint_byte_identical(det_index, tmp_path):
    tables = _tables(tmp_path)
    q = "total sales by billing country"
    # expected legacy output computed directly with the SAME deterministic index
    top_k = retrieve(q, tables, k=5)
    expected_schema = render_schema(tables, only=sorted(expand_with_fk_neighbors(tables, top_k)))
    exp_paths = join_paths(tables, top_k)
    expected = expected_schema
    if exp_paths:
        hint = "\n".join(f"{p['from']}.{p['on']} = {p['to']}.{p['ref_on']}" for p in exp_paths)
        expected = f"{expected_schema}\n\nJoin paths:\n{hint}"
    # graph path
    s1 = _schema_recall(_legacy_state(tables, q))
    assert s1["retrieved_tables"] == top_k                        # candidate order + retrieved_tables
    merged = {**_legacy_state(tables, q), **s1}
    s2 = _table_relation(merged)
    got = s2.get("schema", s1["schema"])
    assert got == expected                                        # COMPLETE schema + join-hint string


def test_current_hybrid_refusal_on_offtopic(det_index, tmp_path):
    s = _schema_recall(_legacy_state(_tables(tmp_path), "what is the weather today"))
    assert s["retrieved_tables"] == [] and s["schema"] == ""


def test_runtime_k_honored(det_index, tmp_path):
    s = _schema_recall(_legacy_state(_tables(tmp_path), "tracks and invoices and customers", k=2))
    assert len(s["retrieved_tables"]) <= 2


def test_run_agent_current_hybrid_populates_answerresult(det_index, tmp_path):
    res = run_agent(str(build(tmp_path / "t.db")), "how many tracks are in each genre",
                    model=PlanningFakeModel("SELECT COUNT(*) FROM track"))
    assert res.retrieved_tables and res.answer and res.sql              # pre-existing fields intact
    assert res.retrieval_result is not None                             # reconstructed
    assert res.retrieval_result.relation_plan.strategy == "legacy_one_hop"


# --- Fix 1: _table_relation routes on relation_plan.strategy, not config.fusion ---

from agent.retrieval.contracts import (RelationEdge, RelationPlan, RetrievalResult, SelectionDecision)
from agent.retrieval.serde import serialize_result


def _state_with_result(tables, retrieved, result):
    return {"tables": tables, "retrieved_tables": retrieved, "schema": "SCHEMA",
            "retrieval_result_serialized": serialize_result(result)}


def test_table_relation_shortest_path_uses_edges_not_join_paths(monkeypatch, tmp_path):
    tables = _tables(tmp_path)
    result = RetrievalResult(config_name="legacy_sp", signals=[], candidates=[], metric_matches=[],
        selection=SelectionDecision(["track", "genre"], [], "noop", {}),
        relation_plan=RelationPlan("shortest_path", ["track", "genre"], ["playlist_track"],
            ["genre", "playlist_track", "track"],
            [RelationEdge("playlist_track", "track_id", "track", "track_id", "physical_fk")], [], []),
        stage_events=[])
    monkeypatch.setattr(graph, "join_paths",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("join_paths called on shortest_path")))
    out = _table_relation(_state_with_result(tables, ["track", "genre"], result))
    assert "playlist_track.track_id = track.track_id" in out["schema"]      # hints from edges


def test_table_relation_legacy_one_hop_uses_join_paths(tmp_path):
    tables = _tables(tmp_path)
    result = RetrievalResult(config_name="rrf_legacyhop", signals=[], candidates=[], metric_matches=[],
        selection=SelectionDecision(["invoice", "customer"], [], "topk", {}),
        relation_plan=RelationPlan("legacy_one_hop", ["invoice", "customer"], [],
                                   ["customer", "invoice"], [], [], []),
        stage_events=[])
    # legacy_one_hop -> join_paths(tables, retrieved). invoice->customer FK exists in the demo DB.
    out = _table_relation(_state_with_result(tables, ["invoice", "customer"], result))
    assert "Join paths:" in out.get("schema", "")


# --- Fix 2: real RRF graph path (stub run_retrieval; join_paths must NOT be called) ---

from agent.retrieval.contracts import TableCandidate


def test_rrf_graph_path_uses_anchors_bridges_and_edges(monkeypatch, tmp_path):
    tables = _tables(tmp_path)
    fixed = RetrievalResult(
        config_name="rrf_hybrid", signals=[],
        candidates=[TableCandidate("track", {}, 0.5, 1), TableCandidate("genre", {}, 0.3, 2)],
        metric_matches=[],
        selection=SelectionDecision(["genre", "track"], [], "topk", {}),
        relation_plan=RelationPlan("shortest_path", ["genre", "track"], ["playlist_track"],
            ["genre", "playlist_track", "track"],
            [RelationEdge("playlist_track", "track_id", "track", "track_id", "physical_fk")], [], []),
        stage_events=[])
    monkeypatch.setattr(graph, "run_retrieval", lambda *a, **k: fixed)
    monkeypatch.setattr(graph, "join_paths",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("join_paths called on RRF path")))
    state = {"question": "q", "tables": tables, "k": 5,
             "retrieval_config_serialized": serialize_config(RetrievalConfig.rrf_hybrid())}
    s1 = _schema_recall(state)
    assert s1["retrieved_tables"] == ["genre", "track"]                  # == selection.anchor_tables
    assert "TABLE track" in s1["schema"] and "TABLE playlist_track" in s1["schema"]  # bridge rendered
    s2 = _table_relation({**state, **s1})
    assert "playlist_track.track_id = track.track_id" in s2["schema"]    # hints from edges only


# --- Fix 4: non-vacuous HITL config-survival probe ---

_PROBE = RetrievalConfig(name="checkpoint_probe", lexical=True, dense_backend="memory",
                         value_backend=None, selector=None, fusion="legacy_minmax",
                         relation_strategy="legacy_one_hop")


def test_hitl_serialized_config_survives_non_vacuous(det_index, tmp_path):
    db = build(tmp_path / "t.db")
    model = PlanningFakeModel(
        "SELECT customer_id, SUM(total) AS s FROM invoice GROUP BY customer_id ORDER BY s DESC LIMIT 5")
    thread_id, first = start_question_session(db, "who are the best customers?", model=model,
                                              retrieval_config=_PROBE)
    assert isinstance(first, dict) and first.get("clarification")
    _, mid = resume_question_session(thread_id, "sales")
    assert isinstance(mid, dict) and mid.get("plan")
    _, result = resume_question_session(thread_id, {"decision": "approve"})
    assert result.retrieval_result is not None
    assert result.retrieval_result.config_name == "checkpoint_probe"     # proves it survived, not fallback
