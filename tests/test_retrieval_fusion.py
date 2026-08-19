from agent.retrieval.contracts import ChannelTableResult
from agent.retrieval.fusion import weighted_rrf


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
def test_rrf_candidates_keep_real_float_scores():
    cr = {"lexical": [_ctr("lexical", "account", 1), _ctr("lexical", "plan", 2)]}
    cands = weighted_rrf(cr, rrf_constant=60, candidate_k=15)
    assert cands and all(isinstance(c.fusion_score, float) for c in cands)
