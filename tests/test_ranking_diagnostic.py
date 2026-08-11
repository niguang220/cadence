"""Ranking diagnostic — pure derivation from a typed RetrievalResult, then a deterministic real-retrieval
smoke on the expanded schema. It disentangles the candidate/RRF layer (Recall@k, gold fusion rank) from
the selector layer (selection recall, gold dropped by selector) from the relation layer (bridges added)."""
from agent.retrieval.contracts import (RelationPlan, RetrievalConfig, RetrievalResult,
                                        SelectionDecision, TableCandidate)
from evalharness.ranking_diagnostic import ranking_diagnostic


def _rr(cand_tables, *, selection=None, context=None):
    cands = [TableCandidate(t, {}, 1.0 / (i + 1), i + 1) for i, t in enumerate(cand_tables)]
    selection = cand_tables[:5] if selection is None else selection
    context = cand_tables[:5] if context is None else context
    return RetrievalResult(
        config_name="dense_value", signals=[], candidates=cands, metric_matches=[],
        selection=SelectionDecision(list(selection), [], "topk", {}),
        relation_plan=RelationPlan("shortest_path", list(selection), [], list(context), [], [], []),
        stage_events=[])


def test_recall_and_precision_at_k():
    # 12 candidates; gold 'plan' is rank 8 (in Top15/Top10, NOT Top5)
    ordered = ["a", "b", "c", "d", "e", "f", "g", "plan", "i", "j", "k", "l"]
    m = ranking_diagnostic(_rr(ordered), ["plan", "a"])
    assert m["gold_fusion_ranks"] == {"plan": 8, "a": 1}
    assert m["recall_at"][5] == 0.5 and m["recall_at"][10] == 1.0 and m["recall_at"][15] == 1.0
    assert m["recall_at"][6] == 0.5                        # offline Top6 diagnostic: plan still out at 6
    assert m["precision_at"][5] == 1 / 5                   # only 'a' of gold in top-5


def test_gold_absent_from_candidates_is_rank_none_and_low_recall():
    m = ranking_diagnostic(_rr(["a", "b", "c"]), ["plan"])
    assert m["gold_fusion_ranks"] == {"plan": None}       # never entered candidates -> upstream miss
    assert m["recall_at"][15] == 0.0


def test_gold_dropped_by_selector_when_in_candidates_but_not_selection():
    ordered = ["a", "b", "c", "d", "e", "plan"]            # plan is candidate rank 6
    d = ranking_diagnostic(_rr(ordered, selection=["a", "b", "c", "d", "e"],
                               context=["a", "b", "c", "d", "e"]), ["plan"])
    assert d["recall_at"][15] == 1.0                        # gold IS in candidates (Top15)
    assert d["recall_at"][6] == 1.0 and d["recall_at"][5] == 0.0   # in at Top6, out at Top5
    assert d["selection_recall"] == 0.0                     # ...but selector (top-5) dropped it
    # signature: Recall@15 high, selection recall low -> selector problem
    assert d["gold_dropped_by_selector"] == ["plan"]


def test_bridges_added_are_context_minus_selection():
    m = ranking_diagnostic(_rr(["a", "b", "c", "d", "e", "f"], selection=["a", "b"],
                               context=["a", "b", "bridge1", "bridge2"]), ["a"])
    assert sorted(m["bridges_added"]) == ["bridge1", "bridge2"]
    assert m["context_table_count"] == 4


# --- real-retrieval smoke on the expanded schema (deterministic; fastembed, no LLM) ---------------

def _expanded():
    import tempfile

    from agent.db.build_saas_db import build
    from agent.db.introspect import introspect
    from agent.retrieval.value_backend import FakeValueBackend
    d = tempfile.mkdtemp()
    db = str(build(f"{d}/e.db", confounders=True))
    return db, introspect(db), FakeValueBackend()


def test_real_retrieval_is_deterministic_across_runs():
    from evals.ranking_diagnostic import run_ranking_diagnostic
    from evalharness.golden import load_saas_metrics
    db, tables, be = _expanded()
    cases = [c for c in load_saas_metrics() if c.category == "arpu"][:2]
    a = run_ranking_diagnostic(tables, db, cases, be, RetrievalConfig.dense_value())
    b = run_ranking_diagnostic(tables, db, cases, be, RetrievalConfig.dense_value())
    for x, y in zip(a, b):
        assert x["candidate_tables_ordered"] == y["candidate_tables_ordered"]
        assert x["selection_tables"] == y["selection_tables"]
        assert x["context_tables"] == y["context_tables"]
    assert len(tables) >= 20                               # non-saturated: >candidate_k(15)


def test_arpu_report_flags_plan_endpoint_and_top15(tmp_path):
    from evals.ranking_diagnostic import run_ranking_diagnostic, summarize
    from evalharness.golden import load_saas_metrics
    db, tables, be = _expanded()
    arpu = [c for c in load_saas_metrics() if c.category == "arpu"]
    s = summarize(run_ranking_diagnostic(tables, db, arpu, be, RetrievalConfig.dense_value()))
    assert len(s["arpu"]) == len(arpu)
    for a in s["arpu"]:
        assert a["plan_role"] == "endpoint"               # plan is a required gold table for arpu
        assert set(a) >= {"plan_fusion_rank", "plan_in_top15", "plan_in_top5", "plan_dropped_by_selector"}


def test_value_fixture_stage3a_results_intact():
    # the diagnostic runs on the value domain too; value linking must still surface the owner table
    import tempfile

    from agent.db.build_value_db import build as build_value_db
    from agent.db.introspect import introspect
    from agent.retrieval.value_backend import FakeValueBackend
    from agent.retrieval.value_index import build_value_index
    from evals.ranking_diagnostic import run_ranking_diagnostic
    from evalharness.golden import load_value_linking
    d = tempfile.mkdtemp()
    db = build_value_db(f"{d}/v.db")
    tables = introspect(db)
    be = FakeValueBackend()
    build_value_index(tables, db, be)
    case = next(c for c in load_value_linking() if c.id == "en_globex_tickets")
    rec = run_ranking_diagnostic(tables, db, [case], be, RetrievalConfig.dense_value())[0]
    assert case.expected_table in rec["candidate_tables_ordered"]   # value hit preserved (Stage 3A intact)
