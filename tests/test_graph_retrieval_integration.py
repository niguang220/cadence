from conftest import PlanningFakeModel
from agent.db.build_demo_db import build
from agent.db.introspect import introspect
from agent.graph import _schema_recall, _table_relation, run_agent
from agent.pipeline import start_question_session, resume_question_session
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.serde import serialize_config


def _tables(tmp_path):
    return introspect(build(tmp_path / "t.db"))


def _state(tables, question, k=5):
    return {"question": question, "tables": tables, "k": k,
            "retrieval_config_serialized": serialize_config(RetrievalConfig.lexical_baseline())}


def test_typed_pipeline_refuses_offtopic(tmp_path):
    s = _schema_recall(_state(_tables(tmp_path), "what is the weather today"))
    assert s["retrieved_tables"] == [] and s["schema"] == ""


def test_runtime_k_honored(tmp_path):
    s = _schema_recall(_state(_tables(tmp_path), "tracks and invoices and customers", k=2))
    assert len(s["retrieved_tables"]) <= 2


def test_run_agent_default_populates_answerresult_via_the_rrf_path(tmp_path):
    """The PUBLIC default path: no retrieval_config passed. Asserts the shipped default is the
    typed RRF architecture -- real fusion scores and typed signals, not the legacy scaffold."""
    res = run_agent(str(build(tmp_path / "t.db")), "how many tracks are in each genre",
                    model=PlanningFakeModel("SELECT COUNT(*) FROM track"))
    assert res.retrieved_tables and res.answer and res.sql              # pre-existing fields intact
    assert res.retrieval_result is not None                             # reconstructed
    rr = res.retrieval_result
    assert rr.config_name == "governed_rrf"
    assert rr.relation_plan.strategy == "shortest_path"
    assert rr.signals, "the default path must emit typed retrieval signals"
    assert all(c.fusion_score is not None for c in rr.candidates), "real RRF fusion scores"


# --- _table_relation consumes the retrieval plan as its sole relation source ---

from agent.retrieval.contracts import (RelationEdge, RelationPlan, RetrievalResult, SelectionDecision)
from agent.retrieval.serde import serialize_result


def _state_with_result(tables, retrieved, result):
    return {"tables": tables, "retrieved_tables": retrieved, "schema": "SCHEMA",
            "retrieval_result_serialized": serialize_result(result)}


def test_table_relation_shortest_path_uses_edges(tmp_path):
    tables = _tables(tmp_path)
    result = RetrievalResult(config_name="legacy_sp", signals=[], candidates=[], metric_matches=[],
        selection=SelectionDecision(["track", "genre"], [], "noop", {}),
        relation_plan=RelationPlan("shortest_path", ["track", "genre"], ["playlist_track"],
            ["genre", "playlist_track", "track"],
            [RelationEdge("playlist_track", "track_id", "track", "track_id", "physical_fk")], [], []),
        stage_events=[])
    out = _table_relation(_state_with_result(tables, ["track", "genre"], result))
    assert "playlist_track.track_id = track.track_id" in out["schema"]      # hints from edges


# --- real RRF graph path ---

from agent.retrieval.contracts import TableCandidate


def test_rrf_graph_path_uses_anchors_bridges_and_edges(monkeypatch, tmp_path):
    import agent.graph as graph
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
    state = {"question": "q", "tables": tables, "k": 5,
             "retrieval_config_serialized": serialize_config(RetrievalConfig.rrf_hybrid())}
    s1 = _schema_recall(state)
    assert s1["retrieved_tables"] == ["genre", "track"]                  # == selection.anchor_tables
    assert "TABLE track" in s1["schema"] and "TABLE playlist_track" in s1["schema"]  # bridge rendered
    s2 = _table_relation({**state, **s1})
    assert "playlist_track.track_id = track.track_id" in s2["schema"]    # hints from edges only


# --- non-vacuous HITL config-survival probe ---

_PROBE = RetrievalConfig(name="checkpoint_probe", lexical=True, dense_backend="memory",
                         value_backend=None, selector=None, fusion="rrf",
                         relation_strategy="shortest_path")


def test_hitl_serialized_config_survives_non_vacuous(tmp_path):
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
