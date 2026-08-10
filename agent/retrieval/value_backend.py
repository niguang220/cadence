"""Value-search backends. A backend owns index storage/search of high-cardinality entity VALUES
(product/company/contract names, ids). A channel (channels.py) turns backend hits into
RetrievalSignals. Keeping them separate means the real ElasticsearchValueBackend drops in behind
the same ValueChannel without touching signal production.

``FakeValueBackend`` is a deterministic in-memory stand-in so the channel/pipeline/ingestion logic
is fully covered without Docker. It reproduces the four match tiers the ES backend will emit:
exact_keyword > exact_phrase > token_match > fuzzy (the same order aggregate.py buckets on)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

_BUCKET = {"exact_keyword": 4, "exact_phrase": 3, "token_match": 2, "fuzzy": 1}


@dataclass(frozen=True)
class ValueDoc:
    document_id: str
    table: str
    column: str
    value: str


@dataclass(frozen=True)
class ValueHit:
    table: str
    column: str
    value: str
    match_type: str            # exact_keyword | exact_phrase | token_match | fuzzy
    score: float
    document_id: str


class ValueBackendError(Exception):
    """Value backend failed (ES unreachable / query error). The pipeline catches this and records a
    value_degraded stage event; the channel must let it propagate."""


class ValueBackend(Protocol):
    def ensure_index(self) -> None: ...
    def upsert(self, docs: list[ValueDoc]) -> None: ...
    def prune(self, keep_ids: set[str]) -> None: ...
    def search(self, question: str, *, allowed: list[tuple[str, str]],
               limit: int = 20) -> list[ValueHit]: ...


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _within_edit1(a: str, b: str) -> bool:
    """True if a and b are within Levenshtein distance 1 (small, deterministic)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    i = j = edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            if la == lb:
                i += 1
                j += 1
            elif la > lb:
                i += 1
            else:
                j += 1
    return edits + (la - i) + (lb - j) <= 1


def _has_run(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def _classify(value: str, question: str) -> tuple[str, float] | None:
    """Deterministic match tier of ``value`` against ``question`` (mirrors the ES query tiers).
    Matching is token-boundary based, so 'Widget' does NOT exact-match inside 'Widgett' (that is a
    fuzzy hit, not an exact one)."""
    vtok = _tokens(value)
    if not vtok:
        return None
    qtok = _tokens(question)
    qset = set(qtok)
    if len(vtok) == 1:
        if vtok[0] in qset:                                # whole-word single-token value
            return ("exact_keyword", 1.0)
    elif _has_run(qtok, vtok):                             # multi-token value as a consecutive run
        return ("exact_phrase", 1.0)
    overlap = set(vtok) & qset
    if overlap:
        return ("token_match", len(overlap) / len(set(vtok)))
    for a in set(vtok):                                    # near-spelling of a value token
        if len(a) >= 4 and any(_within_edit1(a, b) for b in qset):
            return ("fuzzy", 0.5)
    return None


class FakeValueBackend:
    """In-memory value index for tests. Search restricts to the (table, column) allowlist, so a
    non-searchable column can never surface even if a doc for it were present."""

    def __init__(self) -> None:
        self._docs: dict[str, ValueDoc] = {}

    def ensure_index(self) -> None:
        pass

    def upsert(self, docs: list[ValueDoc]) -> None:
        for d in docs:
            self._docs[d.document_id] = d                  # idempotent by document_id

    def prune(self, keep_ids: set[str]) -> None:
        self._docs = {i: d for i, d in self._docs.items() if i in keep_ids}

    def search(self, question: str, *, allowed: list[tuple[str, str]],
               limit: int = 20) -> list[ValueHit]:
        allow = set(allowed)
        hits: list[ValueHit] = []
        for d in self._docs.values():
            if (d.table, d.column) not in allow:           # allowlist restriction (fail-closed)
                continue
            m = _classify(d.value, question)
            if m:
                hits.append(ValueHit(d.table, d.column, d.value, m[0], m[1], d.document_id))
        hits.sort(key=lambda h: (-_BUCKET.get(h.match_type, 0), -h.score, h.table, h.column,
                                 h.document_id))
        return hits[:limit]
