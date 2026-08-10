from agent.db.build_demo_db import build
from agent.db.introspect import introspect
from agent.lexical_retriever import lexical_scores, retrieve_tables, lexical_evidence


def _tables(tmp_path):
    return introspect(build(tmp_path / "t.db"))


# Frozen parity snapshots captured from the PRE-REFACTOR code. DO NOT EDIT THESE.
# Each: question -> (expected nonzero lexical_scores, expected retrieve_tables list)
PARITY = {
    "how many tracks are in each genre": (
        {"genre": 5, "invoice_line": 1, "media_type": 1, "playlist_track": 5,
         "review": 3, "supplier": 1, "track": 6, "track_supplier": 5},
        ["track", "genre", "playlist_track", "track_supplier", "review"],
    ),
    "total sales by billing country": (
        {"customer": 4, "employee": 3, "invoice": 18, "supplier": 3},
        ["invoice", "customer", "employee", "supplier"],
    ),
    "list the songs": ({"track": 5}, ["track"]),
    "total revenue per employee": (
        {"customer": 1, "employee": 5, "invoice": 7},
        ["invoice", "employee", "customer"],
    ),
    "support rep performance": (
        {"customer": 4, "employee": 12}, ["employee", "customer"],
    ),
    "which customers are from Singapore": (
        {"customer": 8, "employee": 1, "invoice": 4, "review": 1, "supplier": 3},
        ["customer", "invoice", "supplier", "employee", "review"],
    ),
    "what is the weather today": ({}, []),
}


import pytest


@pytest.mark.parametrize("question", list(PARITY))
def test_lexical_scores_parity(tmp_path, question):
    tables = _tables(tmp_path)
    expected_scores, expected_ranked = PARITY[question]
    scores = lexical_scores(question, tables)
    assert {k: v for k, v in scores.items() if v} == expected_scores
    assert len(scores) == 14                       # all tables still returned (incl zeros)
    assert retrieve_tables(question, tables) == expected_ranked


def test_evidence_sum_equals_lexical_scores(tmp_path):
    tables = _tables(tmp_path)
    for q in PARITY:
        by_table = {}
        for hit in lexical_evidence(q, tables):
            by_table[hit.table] = by_table.get(hit.table, 0) + hit.weight
        scores = lexical_scores(q, tables)
        assert by_table == {k: v for k, v in scores.items() if v}   # evidence folds to the score


def test_evidence_retains_column_and_target_type(tmp_path):
    tables = _tables(tmp_path)
    hits = lexical_evidence("which customers are from Singapore", tables)
    sample_hits = [h for h in hits if h.match_type == "sample"]
    assert sample_hits
    assert all(h.target_type == "value" and h.column is not None for h in sample_hits)
    singapore = [h for h in sample_hits if h.table == "customer"]
    assert singapore and singapore[0].column == "country"   # sampled value keeps its owner column
    name_hits = [h for h in hits if h.match_type == "name"]
    assert all(h.column is None and h.target_type == "table" for h in name_hits)


def test_token_hitting_multiple_columns_yields_one_representative(tmp_path):
    tables = _tables(tmp_path)
    # no (table, query_term, match_type in term-kinds) duplicated: dedup-per-token preserved
    hits = [h for h in lexical_evidence("total sales by billing country", tables)
            if h.match_type in ("name", "column", "fk_column", "sample", "desc")]
    keys = [(h.table, h.query_term) for h in hits]
    assert len(keys) == len(set(keys))             # one term representative per (table, token)
