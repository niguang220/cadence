"""Tests for the semantic / hybrid schema retriever (Phase 3).

The fusion maths and the two guardrails -- off-topic refusal and graceful fallback
to lexical when the embedding backend is unavailable -- are deterministic and need
no model. The "semantic actually improves recall" claim lives in test_retrieval_recall.
"""
import pytest

from agent.db.build_demo_db import build
from agent.db.introspect import introspect
from agent.lexical_retriever import retrieve_tables
from agent.hybrid_retriever import _minmax, hybrid_retrieve, retrieve


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    return introspect(str(build(tmp_path_factory.mktemp("sem") / "demo.db")))


def test_minmax_normalises_to_unit_range():
    assert _minmax({}) == {}
    assert _minmax({"a": 5, "b": 5}) == {"a": 0.0, "b": 0.0}        # flat -> all zero
    assert _minmax({"a": 0.0, "b": 1.0, "c": 2.0}) == {"a": 0.0, "b": 0.5, "c": 1.0}


def test_hybrid_refuses_offtopic_even_when_semantic_scores_everything(tables):
    # pins the guard against the SEMANTIC signal itself (not via lexical fallback):
    # feed an index that scores every table with DISTINCT positive values (uniform
    # scores would min-max to zero and pass trivially). With no lexical hit, hybrid
    # must still return [] -- without the guard the top semantic table would survive.
    class FakeIndex:
        def table_scores(self, question):
            return {t.name: 0.1 + 0.01 * i for i, t in enumerate(tables)}

    assert hybrid_retrieve("what is the weather tomorrow", tables, FakeIndex()) == []


def test_offtopic_question_refuses_end_to_end(tables):
    # integration check: the full retrieve() path also refuses an off-topic question
    # (whether it runs hybrid or degrades to lexical, the answer is the same: []).
    assert retrieve("what is the weather tomorrow", tables) == []


def test_falls_back_to_lexical_when_embedding_unavailable(tables, monkeypatch):
    # force the embedding backend to fail; retrieve must degrade to the lexical result
    import agent.hybrid_retriever as sr

    def boom(*_a, **_k):
        raise RuntimeError("no embedding model")

    sr._INDEX_CACHE.clear()
    monkeypatch.setattr(sr, "SemanticIndex", boom)
    q = "total sales per genre"
    assert retrieve(q, tables) == retrieve_tables(q, tables)
    sr._INDEX_CACHE.clear()
