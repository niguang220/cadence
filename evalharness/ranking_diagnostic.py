"""Non-saturated ranking diagnostic for the retrieval layer (pure derivation; no I/O, no LLM).

Disentangles WHERE a gold table is lost on a >candidate_k schema:
  - candidate/RRF layer: does the gold table enter the fused Top-k at all? (Recall@k, gold fusion rank)
  - selector layer:       if it's a candidate, did the Top-context_anchor_k selector drop it?
  - relation layer:       which non-gold tables did the RelationPlanner add back as bridges?

Recall@6 is an OFFLINE diagnostic slice only (a "would Top6 have kept it?" probe) — it is NOT a preset
and changes no production default.
"""
from __future__ import annotations

_KS = (5, 6, 10, 15)


def ranking_diagnostic(retrieval_result, gold_tables) -> dict:
    ordered = [c.table for c in sorted(retrieval_result.candidates, key=lambda c: c.fusion_rank)]
    rank_of = {t: i + 1 for i, t in enumerate(ordered)}
    gold = list(dict.fromkeys(gold_tables))                 # de-dup, keep order
    gold_set = set(gold)
    selection = list(retrieval_result.selection.anchor_tables)
    context = list(retrieval_result.relation_plan.context_tables)
    sel_set, ctx_set = set(selection), set(context)

    def recall_at(k):
        top = set(ordered[:k])
        return (len(gold_set & top) / len(gold_set)) if gold_set else None

    def precision_at(k):
        top = ordered[:k]
        return (len(gold_set & set(top)) / len(top)) if top else None

    def recall_over(stage):
        return (len(gold_set & stage) / len(gold_set)) if gold_set else None

    return {
        "candidate_tables_ordered": ordered,
        "gold_fusion_ranks": {t: rank_of.get(t) for t in gold},   # None => never a candidate (upstream miss)
        "recall_at": {k: recall_at(k) for k in _KS},
        "precision_at": {k: precision_at(k) for k in (5, 10, 15)},
        "selection_recall": recall_over(sel_set),
        "context_recall": recall_over(ctx_set),
        "selection_tables": selection,
        "context_tables": context,
        # gold that IS a candidate but the selector dropped -> the "selector problem" signature
        "gold_dropped_by_selector": sorted(t for t in gold if t in rank_of and t not in sel_set),
        # non-selection tables the relation planner pulled into context (bridges)
        "bridges_added": sorted(ctx_set - sel_set),
        "context_table_count": len(context),
    }
