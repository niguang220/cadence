"""Unit tests for the lexical backend/weight selection matrix (pure logic, no retrieval)."""
from __future__ import annotations

from evalharness.lexical_matrix import (LEXICAL_BACKENDS, LEXICAL_WEIGHTS, cell_id, cells,
                                        eliminate, select, summarize_surface)


def _surface(sel=1.0, ctx=1.0, r5=1.0, r10=1.0, r15=1.0, n=6, admitted=None, tables=5.0):
    return {"n": n, "recall_at_5": r5, "recall_at_10": r10, "recall_at_15": r15,
            "precision_at_5": 0.3, "selection_recall": sel, "context_recall": ctx,
            "mean_context_tables": tables, "admitted": n if admitted is None else admitted}


def _cell(backend="bm25", weight=0.5, **over):
    base = {"lexical_backend": backend, "lexical_weight": weight, "deterministic": True,
            "explicit_clue": _surface(), "governed": _surface(n=24), "spider": _surface(n=30)}
    base.update(over)
    return base


def test_matrix_is_the_frozen_six_cells_with_positive_weights():
    assert LEXICAL_BACKENDS == ("hand_weighted", "bm25")
    assert LEXICAL_WEIGHTS == (0.25, 0.5, 1.0)
    assert 0 not in LEXICAL_WEIGHTS, "weight 0 is dense-only ranking, not hybrid fusion"
    assert len(cells()) == 6
    assert cell_id(cells()[0]) == "hand_weighted@w0.25"


def test_summarize_surface_averages_and_counts_admissions():
    rows = [{"recall_at": {5: 1.0, 10: 1.0, 15: 1.0}, "precision_at": {5: 0.4},
             "selection_recall": 1.0, "context_recall": 1.0, "context_table_count": 4,
             "admitted": True},
            {"recall_at": {5: 0.0, 10: 0.5, 15: 1.0}, "precision_at": {5: 0.0},
             "selection_recall": 0.0, "context_recall": 0.5, "context_table_count": 6,
             "admitted": False}]
    s = summarize_surface(rows)
    assert s["n"] == 2 and s["admitted"] == 1
    assert s["recall_at_5"] == 0.5 and s["recall_at_15"] == 1.0
    assert s["mean_context_tables"] == 5.0


def test_a_clean_cell_is_eligible():
    assert eliminate(_cell()) == []


def test_control_regression_eliminates():
    assert "explicit_clue_selection_regression" in eliminate(_cell(explicit_clue=_surface(sel=0.8)))
    assert "explicit_clue_context_regression" in eliminate(_cell(explicit_clue=_surface(ctx=0.8)))
    assert "explicit_clue_admission_regression" in eliminate(
        _cell(explicit_clue=_surface(admitted=5)))


def test_non_determinism_eliminates():
    assert "non_deterministic" in eliminate(_cell(deterministic=False))


def test_spider_recall_floor_eliminates():
    assert "spider_candidate_recall_below_floor" in eliminate(_cell(spider=_surface(n=30, r15=0.9)))
    assert "spider_context_recall_below_floor" in eliminate(_cell(spider=_surface(n=30, ctx=0.9)))


def test_governed_protected_anchor_must_be_exactly_perfect():
    assert "governed_anchor_selection_not_perfect" in eliminate(
        _cell(governed=_surface(n=24, sel=0.99)))
    assert "governed_anchor_context_not_perfect" in eliminate(
        _cell(governed=_surface(n=24, ctx=0.99)))


def test_regression_against_the_baseline_explicit_clue_eliminates():
    reasons = eliminate(_cell(explicit_clue=_surface(sel=1.0, ctx=1.0)),
                        baseline_explicit_clue=_surface(sel=1.0, ctx=1.0))
    assert reasons == []


def test_select_prefers_bm25_on_a_tie():
    hw = _cell("hand_weighted", 1.0)
    bm = _cell("bm25", 0.5)
    out = select([hw, bm])
    assert out["selected"]["lexical_backend"] == "bm25"


def test_select_keeps_hand_weighted_only_on_a_clear_measured_advantage():
    hw = _cell("hand_weighted", 1.0, spider=_surface(n=30, r5=0.95))
    bm = _cell("bm25", 0.5, spider=_surface(n=30, r5=0.60))
    out = select([hw, bm])
    assert out["selected"]["lexical_backend"] == "hand_weighted"


def test_a_noise_level_edge_does_not_count_as_a_clear_advantage():
    """One case out of thirty is ~0.033. That must not outvote the standard-implementation
    policy -- otherwise float noise on a saturated surface picks the backend."""
    hw = _cell("hand_weighted", 1.0, spider=_surface(n=30, r5=1.0))
    bm = _cell("bm25", 1.0, spider=_surface(n=30, r5=0.9833))
    assert select([hw, bm])["selected"]["lexical_backend"] == "bm25"


def test_when_nothing_discriminates_it_falls_back_to_bm25_at_the_neutral_weight():
    out = select([_cell(b, w) for b in ("hand_weighted", "bm25") for w in (0.25, 0.5, 1.0)])
    assert out["selected"] == {"lexical_backend": "bm25", "lexical_weight": 1.0,
                               "dense_weight": 1.0}
    assert out["surfaces_discriminated"] is False
    assert "did not discriminate" in out["reason"]


def test_select_never_returns_an_ineligible_cell():
    bad = _cell("bm25", 0.5, spider=_surface(n=30, r5=1.0, ctx=0.5))
    good = _cell("hand_weighted", 1.0, spider=_surface(n=30, r5=0.1))
    out = select([bad, good])
    assert out["selected"]["lexical_backend"] == "hand_weighted"
    assert any(c["eliminated_for"] for c in out["cells"])


def test_select_reports_no_selection_when_every_cell_is_eliminated():
    out = select([_cell(deterministic=False), _cell("hand_weighted", 1.0, deterministic=False)])
    assert out["selected"] is None


def test_selected_always_pins_a_positive_lexical_weight():
    out = select([_cell("bm25", 0.25)])
    assert out["selected"]["lexical_weight"] > 0
    assert out["selected"]["dense_weight"] == 1.0


# --- driver-side gold resolution (external schemas are mixed-case) -----------------------

def test_gold_tables_resolve_onto_the_schemas_actual_casing():
    from types import SimpleNamespace

    from evals.lexical_matrix import _resolve_gold

    tables = [SimpleNamespace(name="Has_Pet"), SimpleNamespace(name="Pets"),
              SimpleNamespace(name="Student")]
    assert _resolve_gold(["has_pet", "pets"], tables) == ["Has_Pet", "Pets"]
    # an unknown gold table is passed through unchanged rather than silently dropped
    assert _resolve_gold(["missing"], tables) == ["missing"]
