from agent.db.build_demo_db import build
from agent.db.introspect import introspect
from agent.retrieval.contracts import ChannelTableResult
from agent.retrieval.fusion import weighted_rrf, legacy_candidates


def _ctr(ch, table, rank, score=1.0):
    return ChannelTableResult(channel=ch, table=table, raw_table_score=score,
                              channel_rank=rank, signals=[])


def test_rrf_sums_reciprocal_ranks_across_channels():
    cr = {"lexical": [_ctr("lexical", "account", 1), _ctr("lexical", "plan", 2)],
          "dense":   [_ctr("dense", "account", 2), _ctr("dense", "invoice", 1)]}
    cands = weighted_rrf(cr, rrf_constant=60, candidate_k=15)
    scores = {c.table: round(c.fusion_score, 6) for c in cands}
    assert scores["account"] == round(1/61 + 1/62, 6)   # recalled by both channels
    assert scores["invoice"] == round(1/61, 6)
    assert cands[0].table == "account" and cands[0].fusion_rank == 1


def test_rrf_is_deterministic_and_ties_break_by_name():
    cr = {"lexical": [_ctr("lexical", "b", 1), _ctr("lexical", "a", 1)]}   # equal rank
    cands = weighted_rrf(cr, rrf_constant=60, candidate_k=15)
    assert [c.table for c in cands] == ["a", "b"]        # same score -> name asc
    assert weighted_rrf(cr) == weighted_rrf(cr)          # stable across runs


def test_legacy_candidates_reuse_hybrid_retrieve_order(tmp_path):
    import agent.hybrid_retriever as hr
    tables = introspect(build(tmp_path / "t.db"))

    class FakeIndex:                                     # deterministic, no fastembed
        def table_scores(self, q):
            return {"track": 0.9, "genre": 0.2}

    names = hr.hybrid_retrieve("how many tracks are in each genre", tables, FakeIndex(), k=5)
    cands = legacy_candidates("how many tracks are in each genre", tables, k=5, index=FakeIndex())
    assert [c.table for c in cands] == names            # byte-identical order, one source
    assert [c.fusion_rank for c in cands] == list(range(1, len(cands) + 1))
    # legacy min-max fusion is not this pipeline's RRF -- only fusion_rank is meaningful here.
    assert all(c.fusion_score is None for c in cands)


def test_rrf_candidates_keep_real_float_scores():
    cr = {"lexical": [_ctr("lexical", "account", 1), _ctr("lexical", "plan", 2)]}
    cands = weighted_rrf(cr, rrf_constant=60, candidate_k=15)
    assert cands and all(isinstance(c.fusion_score, float) for c in cands)
