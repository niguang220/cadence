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
    """Value backend failed (ES unreachable / query error / partial ingestion failure). The query
    pipeline catches search-time failures and records a value_degraded stage event; ingestion
    failures (upsert/prune) propagate out of build_value_index so the caller fails loud instead of
    querying a silently-incomplete index."""


def _bulk_error_summary(body: dict) -> tuple[int, int, list[int]] | None:
    """Inspect an Elasticsearch _bulk response BODY for per-item failures. ``client.bulk`` returns
    HTTP 200 with ``errors: true`` when some items fail; it does NOT raise. Returns
    ``(failed, total, sorted_status_codes)`` if any item carries an ``error``, else ``None``. A
    delete of a missing doc (404 not_found, no ``error`` key) is not a failure. The caller must not
    put any part of the item bodies / error reasons into a raised message (they can echo values)."""
    if not body.get("errors"):
        return None
    items = body.get("items", [])
    statuses: set[int] = set()
    failed = 0
    for item in items:
        result = next(iter(item.values())) if item else {}     # {op: {...}} -> the op result dict
        if result.get("error") is not None:                    # ES sets 'error' only on real failures
            failed += 1
            status = result.get("status")
            if isinstance(status, int):
                statuses.add(status)
    if not failed:
        return None
    return failed, len(items), sorted(statuses)


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
        return _rank(hits)[:limit]


def _rank(hits: list[ValueHit]) -> list[ValueHit]:
    return sorted(hits, key=lambda h: (-_BUCKET.get(h.match_type, 0), -h.score, h.table, h.column,
                                       h.document_id))


class ElasticsearchValueBackend:
    """Real value backend over Elasticsearch. ES does the recall (which candidate values share
    tokens with / are near-spellings of the question); the match TIER is then assigned by the same
    deterministic ``_classify`` the fake uses, so the two backends agree on match_type (parity).

    ``elasticsearch`` is imported lazily so this module stays importable without the extra."""

    def __init__(self, client, index: str):
        self._es = client
        self._index = index

    @classmethod
    def from_url(cls, url: str, index: str) -> "ElasticsearchValueBackend":
        from elasticsearch import Elasticsearch          # lazy: only when a real backend is built
        return cls(Elasticsearch(url), index)

    def ensure_index(self) -> None:
        try:
            if not self._es.indices.exists(index=self._index):
                self._es.indices.create(
                    index=self._index,
                    settings={"analysis": {"normalizer": {
                        "lc": {"type": "custom", "filter": ["lowercase"]}}}},
                    mappings={"properties": {
                        "table": {"type": "keyword"},
                        "column": {"type": "keyword"},
                        "value": {"type": "text",
                                  "fields": {"raw": {"type": "keyword", "normalizer": "lc"}}}}})
        except Exception as e:
            raise ValueBackendError(str(e)) from e

    def _bulk(self, ops: list[dict], op: str) -> None:
        """Run a refresh-blocking bulk and FAIL LOUD on either a transport error or a per-item
        partial failure. The raised message is sanitized to op type / failed-count / HTTP statuses
        only — never a raw value, document source, or ES error reason (those can echo the data)."""
        try:
            resp = self._es.bulk(operations=ops, refresh="wait_for")
        except Exception as e:                                 # transport / request-level failure
            raise ValueBackendError(f"bulk {op} request failed") from e
        summary = _bulk_error_summary(getattr(resp, "body", resp))
        if summary is not None:
            failed, total, statuses = summary
            raise ValueBackendError(
                f"bulk {op} partial failure: {failed}/{total} items failed; http_status={statuses}")

    def upsert(self, docs: list[ValueDoc]) -> None:
        if not docs:
            return
        ops: list[dict] = []
        for d in docs:
            ops.append({"index": {"_index": self._index, "_id": d.document_id}})
            ops.append({"table": d.table, "column": d.column, "value": d.value})
        self._bulk(ops, "upsert")

    def prune(self, keep_ids: set[str]) -> None:
        try:
            res = self._es.search(index=self._index, size=10000, source=False,
                                  query={"match_all": {}})
        except Exception as e:
            raise ValueBackendError("prune search failed") from e
        stale = {h["_id"] for h in res["hits"]["hits"]} - set(keep_ids)
        if stale:
            ops = [{"delete": {"_index": self._index, "_id": i}} for i in stale]
            self._bulk(ops, "prune")

    def search(self, question: str, *, allowed: list[tuple[str, str]],
               limit: int = 20) -> list[ValueHit]:
        if not allowed:
            return []
        pairs = [{"bool": {"must": [{"term": {"table": t}}, {"term": {"column": c}}]}}
                 for t, c in allowed]
        query = {"bool": {
            "filter": [{"bool": {"should": pairs, "minimum_should_match": 1}}],   # allowlist only
            "should": [{"match": {"value": {"query": question}}},                 # token/phrase recall
                       {"match": {"value": {"query": question, "fuzziness": "AUTO"}}}],  # near-spell
            "minimum_should_match": 1}}
        try:
            res = self._es.search(index=self._index, size=max(limit * 5, 50), query=query)
        except Exception as e:
            raise ValueBackendError(str(e)) from e
        hits: list[ValueHit] = []
        for h in res["hits"]["hits"]:
            src = h["_source"]
            m = _classify(src["value"], question)          # deterministic tier -> fake/ES parity
            if m:
                hits.append(ValueHit(src["table"], src["column"], src["value"], m[0], m[1], h["_id"]))
        return _rank(hits)[:limit]
