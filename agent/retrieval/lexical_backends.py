"""Lexical scoring backends. A backend owns HOW a question is scored against a schema; the
LexicalChannel (channels.py) owns the boundary and stays scoring-agnostic -- the same split
DenseBackend/DenseChannel already use.

Two backends exist deliberately:

* ``HandWeightedLexicalBackend`` -- the project's original hand-curated scorer (table-name /
  column / sample-value weights plus a domain alias layer). Kept as the COMPARISON arm.
* ``BM25LexicalBackend`` -- a standard, maintained in-memory BM25 (``rank_bm25``). No scoring
  formula is written here; ranking is delegated to the library.

Both share ``lexical_retriever._tokenize`` for normalisation and stopwords, so the arms differ
only in scoring, never in preprocessing -- otherwise the comparison would not be honest.

ADMISSION-GATE CONTRACT (safety-critical). The retrieval pipeline decides whether a question has
any footing in this schema from whether the lexical channel produced ANY result; with no footing
an off-topic question is refused. BM25Okapi scores a zero-overlap document exactly 0.0 but can
score a *matching* document negatively (a term present in most documents gets negative IDF), so
emission is decided by TOKEN OVERLAP and never by a score threshold. A table with no overlapping
query token emits no signal; a table with overlap always does, whatever its score.
"""
from __future__ import annotations

from typing import Literal, Protocol

from agent.db.introspect import Table
from agent.lexical_retriever import _tokenize, lexical_evidence
from agent.retrieval.contracts import RetrievalSignal

LexicalBackendName = Literal["hand_weighted", "bm25"]


class LexicalBackend(Protocol):
    def signals(self, question: str, tables: list[Table]) -> list[RetrievalSignal]: ...


class HandWeightedLexicalBackend:
    """The original scorer, unchanged: one signal per (table, matching question token) at that
    token's best field weight, plus domain word/phrase alias boosts."""

    def signals(self, question: str, tables: list[Table]) -> list[RetrievalSignal]:
        return [
            RetrievalSignal(
                channel="lexical", target_type=hit.target_type, table=hit.table,
                column=hit.column, query_term=hit.query_term,
                raw_score=float(hit.weight), match_type=hit.match_type)
            for hit in lexical_evidence(question, tables)
        ]


def _table_doc_tokens(table: Table) -> list[str]:
    """The BM25 document for one table: a token LIST (repetition intact, because term frequency
    is exactly what BM25 consumes) over the table name, its description, and every column's name,
    description, and sampled values."""
    parts: list[str] = [table.name, table.description or ""]
    for col in table.columns:
        parts.append(col.name)
        parts.append(col.description or "")
        parts.extend(col.sample_values)
    tokens: list[str] = []
    for part in parts:
        tokens.extend(sorted(_tokenize(part)))   # sorted: deterministic doc construction
    return tokens


class BM25LexicalBackend:
    """Okapi BM25 over one document per table, delegated to ``rank_bm25``.

    Emits one table-grained signal per table that shares at least one normalised token with the
    question (see the admission-gate contract above). The index is rebuilt per call: schemas here
    are prompt-sized (tens of tables), and a cache keyed on schema identity would add a staleness
    surface for no measurable gain at this scale.
    """

    def signals(self, question: str, tables: list[Table]) -> list[RetrievalSignal]:
        q_tokens = _tokenize(question)
        if not q_tokens or not tables:
            return []
        docs = [_table_doc_tokens(t) for t in tables]
        if not any(docs):
            return []
        from rank_bm25 import BM25Okapi

        query = sorted(q_tokens)                     # deterministic query order
        scores = BM25Okapi(docs).get_scores(query)
        out: list[RetrievalSignal] = []
        for table, doc, score in zip(tables, docs, scores):
            if not q_tokens & set(doc):              # no overlap -> no lexical footing
                continue
            out.append(RetrievalSignal(
                channel="lexical", target_type="table", table=table.name, column=None,
                query_term=question, raw_score=float(score), match_type="bm25"))
        return out


def lexical_backend_for(name: LexicalBackendName) -> LexicalBackend:
    if name == "hand_weighted":
        return HandWeightedLexicalBackend()
    if name == "bm25":
        return BM25LexicalBackend()
    raise ValueError(f"unknown lexical backend {name!r}")
