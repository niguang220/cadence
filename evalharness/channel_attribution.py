"""Channel attribution — pure derivation. Given each gold table's per-channel ranks (lexical / dense /
fused RRF) and the deterministic selector/relation/governance outcome, classify WHERE it was lost, and
compute per-channel Recall@k / Precision@k from a channel's own ordered table list. No I/O, no LLM."""
from __future__ import annotations

_TOP = 5          # deterministic selector cutoff (context_anchor_k)
_CAND = 15        # candidate_k


def channel_recall_precision(ordered_tables: list[str], gold_tables, ks=(5, 10, 15)) -> dict:
    gold = set(gold_tables)
    def rec(k):
        return (len(gold & set(ordered_tables[:k])) / len(gold)) if gold else None
    def prec(k):
        top = ordered_tables[:k]
        return (len(gold & set(top)) / len(top)) if top else None
    return {"recall": {k: rec(k) for k in ks}, "precision": {k: prec(k) for k in ks}}


def classify_gold_table(*, lex_rank, dense_rank, fused_rank, selector_kept: bool,
                        context_recovered: bool, governance_protected: bool) -> list[str]:
    """Every applicable tag for one gold table. The tags map onto the decision rules: which upstream
    channel had it, whether RRF demoted it, whether only the Top-5 selector lost it, whether it never
    entered candidates, and whether metric-governance (protected anchor) is what recovered it."""
    def _in(rank, k):
        return rank is not None and rank <= k

    in_lex_top, in_dense_top = _in(lex_rank, _TOP), _in(dense_rank, _TOP)
    in_lex_cand, in_dense_cand = _in(lex_rank, _CAND), _in(dense_rank, _CAND)
    in_fused_top, in_fused_cand = _in(fused_rank, _TOP), _in(fused_rank, _CAND)

    tags: list[str] = []
    if not in_lex_cand and not in_dense_cand:
        tags.append("absent_from_both_channels_top15")           # recall-layer problem
    if in_lex_top and not in_fused_top:
        tags.append("lexical_top5_but_rrf_demoted_out_of_top5")  # fusion demoted a good lexical hit
    if in_dense_top and not in_fused_top:
        tags.append("dense_top5_but_rrf_demoted_out_of_top5")    # fusion demoted a good dense hit
    if in_lex_cand and in_dense_cand and in_fused_cand and not selector_kept:
        tags.append("both_recalled_but_selector_dropped_at_top5")  # selector-layer problem
    if governance_protected and (selector_kept or context_recovered) \
            and not (in_lex_top or in_dense_top):
        tags.append("governance_protected_recovery")             # only metric governance saves it
    return tags
