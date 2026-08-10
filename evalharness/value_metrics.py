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

    return {
        "candidate_tables_ordered": ordered,
        "candidate_count": len(ordered),
        "candidate_recall": recall(cand),
        "fusion_at_5_recall": recall(top5),                             # gold in top-5 by fusion rank
        "candidate_precision": (len(gold & cand) / len(cand)) if cand else None,
        "selection_tables": list(retrieval_result.selection.anchor_tables),
        "context_tables": list(retrieval_result.relation_plan.context_tables),
    }
