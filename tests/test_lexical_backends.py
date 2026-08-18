"""The lexical backend boundary inside LexicalChannel, and the in-memory BM25 backend.

Two backends must be interchangeable behind one seam: the existing hand-weighted scorer
(kept as the comparison arm) and a standard maintained BM25 (rank_bm25). The invariants that
matter are (a) the hand-weighted arm stays byte-identical to the pre-boundary behaviour, and
(b) BM25 preserves the admission-gate contract -- a table with no query-token overlap produces
NO signal, so an off-topic question still has no lexical footing and is still refused.
"""
from __future__ import annotations

import pytest

from agent.db.introspect import introspect
from agent.lexical_retriever import lexical_evidence
from agent.retrieval.channels import LexicalChannel
from agent.retrieval.lexical_backends import (BM25LexicalBackend, HandWeightedLexicalBackend,
                                              lexical_backend_for)


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    from agent.db.build_saas_db import build
    return introspect(build(tmp_path_factory.mktemp("lex") / "saas.db"))


def test_hand_weighted_backend_is_byte_identical_to_the_previous_channel(tables):
    """The comparison arm must be the SAME scorer, not a reimplementation of it."""
    q = "how many active users per region"
    got = HandWeightedLexicalBackend().signals(q, tables)
    expected = lexical_evidence(q, tables)
    assert len(got) == len(expected)
    for signal, hit in zip(got, expected):
        assert signal.channel == "lexical"
        assert (signal.table, signal.column, signal.query_term) == (hit.table, hit.column, hit.query_term)
        assert signal.raw_score == float(hit.weight)
        assert signal.match_type == hit.match_type
        assert signal.target_type == hit.target_type


def test_channel_defaults_to_the_hand_weighted_backend(tables):
    q = "revenue by plan"
    assert LexicalChannel().signals(q, tables) == HandWeightedLexicalBackend().signals(q, tables)


def test_channel_delegates_to_an_injected_backend(tables):
    q = "revenue by plan"
    assert LexicalChannel(BM25LexicalBackend()).signals(q, tables) == \
        BM25LexicalBackend().signals(q, tables)


def test_backend_factory_routes_on_the_config_field():
    assert isinstance(lexical_backend_for("hand_weighted"), HandWeightedLexicalBackend)
    assert isinstance(lexical_backend_for("bm25"), BM25LexicalBackend)
    with pytest.raises(ValueError):
        lexical_backend_for("nope")


# --- BM25 admission-gate contract: the safety-critical property -------------------------

def test_bm25_emits_no_signal_for_an_offtopic_question(tables):
    """No query-token overlap anywhere -> no lexical footing -> the pipeline refuses.
    BM25Okapi returns 0.0 for zero-overlap docs, but it can also return NEGATIVE scores for
    matching docs, so emission is decided by token OVERLAP, never by a score threshold."""
    assert BM25LexicalBackend().signals("what is the weather in singapore today", tables) == []


def test_bm25_emits_signals_only_for_tables_with_query_token_overlap(tables):
    signals = BM25LexicalBackend().signals("how many invoices", tables)
    assert signals, "an in-domain question must produce lexical footing"
    named = {s.table for s in signals}
    assert "invoice" in named
    # every emitted table must genuinely share a normalized token with the question
    from agent.lexical_retriever import _tokenize
    q_tokens = _tokenize("how many invoices")
    for s in signals:
        table = next(t for t in tables if t.name == s.table)
        doc = set(_tokenize(" ".join(
            [table.name, table.description or ""]
            + [c.name for c in table.columns]
            + [c.description or "" for c in table.columns]
            + [v for c in table.columns for v in c.sample_values])))
        assert q_tokens & doc, f"{s.table} emitted without token overlap"


def test_bm25_signals_are_table_grained_and_typed(tables):
    signals = BM25LexicalBackend().signals("how many invoices", tables)
    for s in signals:
        assert s.channel == "lexical"
        assert s.target_type == "table"
        assert s.column is None
        assert s.match_type == "bm25"
        assert isinstance(s.raw_score, float)
    assert len({s.table for s in signals}) == len(signals), "one signal per table"


def test_bm25_is_deterministic(tables):
    q = "monthly recurring revenue by plan tier"
    first = BM25LexicalBackend().signals(q, tables)
    second = BM25LexicalBackend().signals(q, tables)
    assert [(s.table, s.raw_score) for s in first] == [(s.table, s.raw_score) for s in second]


def test_bm25_ranks_the_named_table_first(tables):
    signals = sorted(BM25LexicalBackend().signals("list the plans", tables),
                     key=lambda s: -s.raw_score)
    assert signals[0].table == "plan"


def test_bm25_handles_an_empty_question(tables):
    assert BM25LexicalBackend().signals("", tables) == []


def test_bm25_handles_an_empty_schema():
    assert BM25LexicalBackend().signals("anything", []) == []
