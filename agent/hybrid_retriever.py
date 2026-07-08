"""Semantic schema retrieval: a second, embedding-based signal for table selection.

The lexical retriever (lexical_retriever) matches question words to table/column
names plus a hand-curated alias layer. It misses tables a question never names --
e.g. "sales per genre" needs the bridge table `track`, which no question word hits.
This module embeds each column (table + column + description) with a small local
ONNX model (fastembed, no torch) and scores a table by its best column's cosine
similarity to the question, then FUSES that with the lexical score (min-max
normalised, weighted) -- augmenting the lexical layer, not replacing it.

Deterministic (same model + input -> same vector), so hybrid recall stays testable.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

_log = logging.getLogger(__name__)

from agent.db.introspect import Column, Table
from agent.lexical_retriever import lexical_scores, retrieve_tables

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_DIM = 384


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding  # lazy: heavy import + first-use model download
    return TextEmbedding(_MODEL_NAME)


def _embed(texts: list[str]) -> np.ndarray:
    vecs = np.asarray(list(_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)  # unit vectors -> dot product is cosine


def _column_doc(table: Table, col: Column) -> str:
    parts = [table.name, col.name]
    if col.description:
        parts.append(col.description)
    return " ".join(parts)


class SemanticIndex:
    """Column-level embedding index over a schema. Build once, reuse per question."""

    def __init__(self, tables: list[Table]):
        docs: list[str] = []
        self._owners: list[str] = []
        for t in tables:
            for c in t.columns:
                docs.append(_column_doc(t, c))
                self._owners.append(t.name)
        self._embs = _embed(docs) if docs else np.zeros((0, _DIM), np.float32)

    def table_scores(self, question: str) -> dict[str, float]:
        """Max cosine similarity of the question to any column of each table."""
        if self._embs.shape[0] == 0:
            return {}
        q = _embed([question])[0]
        sims = self._embs @ q
        scores: dict[str, float] = {}
        for name, s in zip(self._owners, sims):
            scores[name] = max(scores.get(name, -1.0), float(s))
        return scores


def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi <= lo:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


_DEFAULT_ALPHA = 0.4  # semantic weight; robust across 0.2-0.6 on the golden set


def hybrid_retrieve(question: str, tables: list[Table], index: SemanticIndex,
                    k: int = 5, alpha: float = _DEFAULT_ALPHA) -> list[str]:
    """Top-k tables by fused score = (1-alpha)*lexical_norm + alpha*semantic_norm.

    Both signals are min-max normalised across all tables first so their scales are
    comparable (lexical is small integers, semantic is cosine). alpha weights how
    much the semantic signal can move the ranking.

    Refusal is preserved on the LEXICAL signal: if no table has any lexical hit, the
    question is off-topic and we return [] rather than let the semantic signal (which
    scores every table) fabricate relevance from nothing. Semantic only *completes* a
    set the lexical layer already has a foothold in (e.g. rescuing a bridge table).
    """
    lex_raw = lexical_scores(question, tables)
    if not any(v > 0 for v in lex_raw.values()):
        return []
    lex = _minmax({n: float(v) for n, v in lex_raw.items()})
    sem = _minmax(index.table_scores(question))
    fused = {t.name: (1 - alpha) * lex.get(t.name, 0.0) + alpha * sem.get(t.name, 0.0)
             for t in tables}
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, score in ranked[:k] if score > 0]


def _schema_fingerprint(tables: list[Table]) -> tuple:
    """Hashable identity of a schema's embeddable content, for caching the index."""
    return tuple(sorted(_column_doc(t, c) for t in tables for c in t.columns))


_INDEX_CACHE: dict[tuple, SemanticIndex] = {}


def retrieve(question: str, tables: list[Table], k: int = 5) -> list[str]:
    """Schema retrieval for the agent: hybrid (lexical + semantic) when the embedding
    backend is available, falling back to the lexical retriever if fastembed / its
    model can't load. The semantic index is built once per schema and cached.
    """
    try:
        fp = _schema_fingerprint(tables)
        index = _INDEX_CACHE.get(fp)
        if index is None:
            index = _INDEX_CACHE[fp] = SemanticIndex(tables)
        return hybrid_retrieve(question, tables, index, k=k)
    except Exception:  # missing dep, model download failure, runtime error -> degrade
        # broad on purpose: the agent must never crash on retrieval. Logged (with
        # traceback) rather than swallowed silently, so a real hybrid bug still shows.
        _log.warning("semantic retrieval failed; falling back to lexical", exc_info=True)
        return retrieve_tables(question, tables, k=k)
