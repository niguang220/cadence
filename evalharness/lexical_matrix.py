"""Deterministic lexical-backend x RRF-weight selection matrix (pure derivation, no I/O, no LLM).

Answers one bounded question: which (lexical backend, lexical weight) pair should be the shipping
retrieval default? Scoring is delegated to the existing ``ranking_diagnostic``; this module owns
the cell definition, the aggregation, and the selection rule.

Surfaces, chosen so the matrix never asks semantic-OFF retrieval for knowledge it cannot have:

* ``explicit_clue``  -- the frozen control subset of the internal fixture, semantic OFF. Controls
  are plain structural questions whose tables are named in the question, so they are a fair OFF
  gate. Governed metric cases are deliberately NOT part of the OFF surface.
* ``spider``         -- the frozen external slice, semantic OFF. External schemas have no governed
  metrics, so this is raw ranking quality on unseen databases.
* ``governed``       -- the internal metric cases with governance ON. Not a selection criterion:
  it exists to verify the protected-anchor invariant (selection/context recall == 1.0).

Selection eliminates on regressions and invariants first and only then prefers; BM25 wins ties by
policy, because a standard maintained implementation is preferred over a bespoke scorer unless the
bespoke one is measurably better.
"""
from __future__ import annotations

from evalharness.ranking_diagnostic import ranking_diagnostic

LEXICAL_BACKENDS = ("hand_weighted", "bm25")
LEXICAL_WEIGHTS = (0.25, 0.5, 1.0)          # dense weight is fixed at 1.0; 0 is forbidden


def cells() -> list[dict]:
    """The frozen matrix: every (backend, positive lexical weight) pair, in a fixed order."""
    return [{"lexical_backend": b, "lexical_weight": w}
            for b in LEXICAL_BACKENDS for w in LEXICAL_WEIGHTS]


def cell_id(cell: dict) -> str:
    return f"{cell['lexical_backend']}@w{cell['lexical_weight']}"


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def score_case(retrieval_result, gold_tables) -> dict:
    """One case's rank-sensitive metrics plus whether retrieval admitted the question at all."""
    diag = ranking_diagnostic(retrieval_result, gold_tables)
    return {
        "recall_at": diag["recall_at"],
        "precision_at": diag["precision_at"],
        "selection_recall": diag["selection_recall"],
        "context_recall": diag["context_recall"],
        "context_table_count": len(diag["context_tables"]),
        "admitted": bool(retrieval_result.candidates),
    }


def summarize_surface(rows: list[dict]) -> dict:
    """Aggregate one surface of one cell. ``rows`` are ``score_case`` outputs."""
    return {
        "n": len(rows),
        **{f"recall_at_{k}": _mean(r["recall_at"][k] for r in rows) for k in (5, 10, 15)},
        "precision_at_5": _mean(r["precision_at"][5] for r in rows),
        "selection_recall": _mean(r["selection_recall"] for r in rows),
        "context_recall": _mean(r["context_recall"] for r in rows),
        "mean_context_tables": _mean(float(r["context_table_count"]) for r in rows),
        "admitted": sum(1 for r in rows if r["admitted"]),
    }


# --- selection rule -----------------------------------------------------------------------

SPIDER_RECALL_FLOOR = 0.95          # candidate and context recall on the external slice
GOVERNED_ANCHOR_TARGET = 1.0        # protected anchors are an invariant, not a target to approach


def eliminate(cell_summary: dict, *, baseline_explicit_clue: dict | None = None) -> list[str]:
    """Reasons this cell is ineligible. Empty list means eligible."""
    reasons: list[str] = []
    explicit, spider, governed = (cell_summary["explicit_clue"], cell_summary["spider"],
                                  cell_summary["governed"])

    if not cell_summary.get("deterministic", False):
        reasons.append("non_deterministic")

    # control/safety regression: every explicit-clue control must stay fully recalled and admitted
    if explicit["selection_recall"] is not None and explicit["selection_recall"] < 1.0:
        reasons.append("explicit_clue_selection_regression")
    if explicit["context_recall"] is not None and explicit["context_recall"] < 1.0:
        reasons.append("explicit_clue_context_regression")
    if explicit["admitted"] != explicit["n"]:
        reasons.append("explicit_clue_admission_regression")
    if baseline_explicit_clue is not None:
        for key in ("selection_recall", "context_recall"):
            base, got = baseline_explicit_clue.get(key), explicit.get(key)
            if base is not None and got is not None and got < base:
                reasons.append(f"explicit_clue_{key}_below_baseline")

    if (spider["recall_at_15"] or 0.0) < SPIDER_RECALL_FLOOR:
        reasons.append("spider_candidate_recall_below_floor")
    if (spider["context_recall"] or 0.0) < SPIDER_RECALL_FLOOR:
        reasons.append("spider_context_recall_below_floor")

    if governed["selection_recall"] != GOVERNED_ANCHOR_TARGET:
        reasons.append("governed_anchor_selection_not_perfect")
    if governed["context_recall"] != GOVERNED_ANCHOR_TARGET:
        reasons.append("governed_anchor_context_not_perfect")
    return reasons


# A quality difference below this margin is NOISE, not a measured advantage. One case out of
# thirty moving is ~0.033; the margin is set just above that so a single-case wobble on a
# saturated surface cannot outvote the standard-implementation policy.
CLEAR_ADVANTAGE_MARGIN = 0.05
NEUTRAL_LEXICAL_WEIGHT = 1.0        # equal weighting: the untuned choice


def quality_vector(cell_summary: dict) -> tuple[float, ...]:
    """Gate-relevant quality, higher is better. Ranking on the two SELECTION surfaces only --
    external Spider first (the only unseen-schema evidence), then internal explicit-clue.
    ``mean_context_tables`` is deliberately NOT in here: it is a guard checked by elimination,
    not a preference strong enough to choose a backend."""
    spider, explicit = cell_summary["spider"], cell_summary["explicit_clue"]
    return (
        spider["recall_at_5"] or 0.0,
        spider["recall_at_10"] or 0.0,
        spider["recall_at_15"] or 0.0,
        spider["context_recall"] or 0.0,
        explicit["recall_at_5"] or 0.0,
        explicit["context_recall"] or 0.0,
    )


def beats_clearly(a: dict, b: dict) -> bool:
    """True when ``a`` has a CLEAR measured advantage over ``b``: strictly better on at least one
    quality component by more than the margin, and not worse than the margin on any component."""
    va, vb = quality_vector(a), quality_vector(b)
    if any(x < y - CLEAR_ADVANTAGE_MARGIN for x, y in zip(va, vb)):
        return False
    return any(x > y + CLEAR_ADVANTAGE_MARGIN for x, y in zip(va, vb))


def select(summaries: list[dict], *, baseline_explicit_clue: dict | None = None) -> dict:
    """Apply the locked selection rule and return the decision with its full reasoning.

    Order: eliminate on regressions and invariants; then prefer BM25 unless the hand-weighted
    scorer beats it CLEARLY (see ``beats_clearly``); then, among equally-good cells, prefer the
    neutral weight. That last step matters: if the frozen surfaces cannot tell the weights apart,
    the honest output is the untuned equal weighting, not a weight fitted to a saturated fixture.
    """
    judged = []
    for s in summaries:
        reasons = eliminate(s, baseline_explicit_clue=baseline_explicit_clue)
        judged.append({**s, "eligible": not reasons, "eliminated_for": reasons})

    eligible = [s for s in judged if s["eligible"]]
    if not eligible:
        return {"selected": None, "reason": "no eligible cell", "cells": judged}

    def rank_key(cell):
        return (
            tuple(-v for v in quality_vector(cell)),          # better quality first
            0 if cell["lexical_backend"] == "bm25" else 1,     # policy: prefer the standard impl
            abs(cell["lexical_weight"] - NEUTRAL_LEXICAL_WEIGHT),   # prefer the untuned weight
            cell["lexical_weight"],                            # final deterministic tie-break
        )

    bm25_cells = [s for s in eligible if s["lexical_backend"] == "bm25"]
    best = min(eligible, key=rank_key)
    if bm25_cells:
        best_bm25 = min(bm25_cells, key=rank_key)
        # hand-weighted only wins by beating the best BM25 cell clearly
        best = best if beats_clearly(best, best_bm25) else best_bm25

    discriminating = len({quality_vector(s) for s in eligible}) > 1
    return {
        "selected": {"lexical_backend": best["lexical_backend"],
                     "lexical_weight": best["lexical_weight"], "dense_weight": 1.0},
        "reason": ("best eligible cell under the locked selection rule" if discriminating else
                   "frozen surfaces did not discriminate between eligible cells; fell back to the "
                   "standard BM25 backend at the neutral (untuned) equal weighting"),
        "surfaces_discriminated": discriminating,
        "cells": judged,
    }
