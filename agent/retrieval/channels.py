"""Retrieval channels: turn raw scoring into typed RetrievalSignals whose owner is always a table.
A column/value hit votes for its owner table via signal.table but keeps its target_type/column."""
from __future__ import annotations

from agent.db.introspect import Table
from agent.retrieval.backends import DenseBackend
from agent.retrieval.contracts import RetrievalSignal
from agent.retrieval.lexical_backends import HandWeightedLexicalBackend, LexicalBackend
from agent.retrieval.value_backend import ValueBackend
from agent.retrieval.value_policy import resolve_searchable_columns


class LexicalChannel:
    """Boundary over a pluggable lexical backend (see lexical_backends.py). The channel owns the
    seam; the backend owns the scoring. Defaults to the hand-weighted scorer so a direct
    construction keeps the pre-boundary behaviour."""

    def __init__(self, backend: LexicalBackend | None = None):
        self._backend = backend or HandWeightedLexicalBackend()

    def signals(self, question: str, tables: list[Table]) -> list[RetrievalSignal]:
        return self._backend.signals(question, tables)


class DenseChannel:
    def __init__(self, backend: DenseBackend):
        self._backend = backend

    def signals(self, question: str, tables: list[Table]) -> list[RetrievalSignal]:
        # Do NOT catch DenseBackendError here — the pipeline catches it to emit dense_degraded (G5).
        out: list[RetrievalSignal] = []
        for tbl, col, score in self._backend.column_scores(question, tables):
            out.append(RetrievalSignal(
                channel="dense", target_type="column", table=tbl, column=col,
                query_term=question, raw_score=score, match_type="cosine"))
        return out


class ValueChannel:
    """Entity-value search: a hit on a searchable column's VALUE (e.g. a product/company name)
    votes for its owner table. Search is restricted to the searchable allowlist. A backend error
    is NOT caught here — the pipeline catches it to emit a value_degraded event."""

    def __init__(self, backend: ValueBackend):
        self._backend = backend

    def signals(self, question: str, tables: list[Table]) -> list[RetrievalSignal]:
        allowed = resolve_searchable_columns(tables)
        if not allowed:
            return []
        out: list[RetrievalSignal] = []
        for hit in self._backend.search(question, allowed=allowed):
            # matched_value is set ONLY here (searchable value hits): search is restricted to the
            # searchable allowlist, so a pii/public column can never carry a canonical value forward.
            out.append(RetrievalSignal(
                channel="value", target_type="value", table=hit.table, column=hit.column,
                query_term=question, raw_score=float(hit.score), match_type=hit.match_type,
                document_id=hit.document_id, matched_value=hit.value))
        return out
