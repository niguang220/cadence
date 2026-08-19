"""Rank-sensitive retrieval metrics for the Stage 2 value-linking ablation — the metrics the
Stage 1 config ablation could NOT compute because it stored no ordered candidate list.

Given a typed ``RetrievalResult`` and the gold tables, these expose ranking quality (Fusion@5) and
precision, not just saturated recall, on a non-saturated schema. Pure derivation (no I/O)."""
from __future__ import annotations


def rank_sensitive_metrics(retrieval_result, gold_tables) -> dict:
    gold = set(gold_tables)
    ordered = [c.table for c in sorted(retrieval_result.candidates, key=lambda c: c.fusion_rank)]
    cand = set(ordered)
    top5 = set(ordered[:5])

    def recall(stage: set):
        return (len(gold & stage) / len(gold)) if gold else None       # gold-less negatives -> None

    selection = list(retrieval_result.selection.anchor_tables)
    context = list(retrieval_result.relation_plan.context_tables)
    return {
        "candidate_tables_ordered": ordered,
        "candidate_count": len(ordered),
        "candidate_recall": recall(cand),
        # fusion_at_5_recall = gold within the config's OWN top-5 candidate ordering (fusion_rank).
        # Every maintained retrieval preset uses RRF fusion, so this is one consistent rank.
        "fusion_at_5_recall": recall(top5),
        "candidate_precision": (len(gold & cand) / len(cand)) if cand else None,
        "selection_tables": selection,
        "context_tables": context,
        "selection_recall": recall(set(selection)),                     # gold kept after selector top-k
        "context_recall": recall(set(context)),                         # gold in the rendered context set
    }
