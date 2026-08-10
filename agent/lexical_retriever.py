"""Lexical schema retriever: pick the tables most relevant to a question.

Real databases have far too many tables to dump into a prompt, so before
generating SQL we select the top-k relevant tables. This is the deterministic
*lexical* baseline (no LLM, no embeddings): score each table by how well the
question's words overlap its name, columns, descriptions and sample values,
plus a small hand-curated layer of domain synonyms/phrases. Embedding / hybrid
retrieval can be added later as an upgrade.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from agent.db.introspect import Table

# Generic words that carry no schema-linking signal. Deliberately keeps words
# that ARE column-meaningful (total, count, name, date, price, country, ...).
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "by", "for", "to", "and", "or",
    "with", "how", "many", "much", "what", "which", "who", "whose", "is", "are",
    "was", "were", "do", "does", "did", "show", "list", "give", "me", "all",
    "that", "this", "from", "there", "their", "per", "each", "as", "be",
}

# Weights: a hit on a table name is the strongest signal, then real columns and
# concrete sample values; FK columns (track_id, customer_id, ...) are weak — they
# appear in many tables and must NOT rival the actual entity table's name.
_W_TABLE_NAME = 5
_W_COLUMN = 3
_W_FK_COLUMN = 1
_W_SAMPLE = 3
_W_DESC = 1

# The demo's business vocabulary (normalized form), mapping a word to the tables
# it implies. These are domain synonyms, not morphology: "sales"/"revenue" mean
# the invoice table even though that word appears in no column.
WORD_ALIASES: dict[str, dict[str, int]] = {
    "sale": {"invoice": 4},        # normalized form of "sales"
    "revenue": {"invoice": 4},
    "spending": {"invoice": 3, "customer": 2},
    "order": {"invoice": 4},
    "purchase": {"invoice": 4},
    "song": {"track": 4},          # normalized form of "songs"
    "rep": {"employee": 3},
}

# Multi-word phrases (matched as substrings of the lowercased question) that a
# bag of tokens can't disambiguate — e.g. "billing country" is invoice, not the
# customer's own country.
PHRASE_ALIASES: dict[str, dict[str, int]] = {
    "billing country": {"invoice": 5},
    "home country": {"customer": 5},
    "customer country": {"customer": 5},
    "support rep": {"employee": 5, "customer": 2},
    "line item": {"invoice_line": 5},
}


LexicalMatchType = Literal["name", "column", "fk_column", "sample", "desc", "alias", "phrase"]


@dataclass
class LexicalHit:
    """One scoring signal linking a table to the question. ``lexical_scores`` is the fold
    (sum-by-table) over ``lexical_evidence``; this is the single source both derive from."""

    table: str                 # owner table this hit votes for
    query_term: str            # a normalized question token, or a matched phrase string
    weight: int
    match_type: LexicalMatchType
    column: str | None = None  # owner column for column/fk_column/desc(on a column)/sample hits

    @property
    def target_type(self) -> Literal["table", "column", "value"]:
        # granularity depends on BOTH match_type AND whether a column is attached:
        #   name / table-description / alias / phrase -> "table"
        #   column / fk_column / column-description    -> "column"
        #   sampled field value                        -> "value" (keeps its owner column)
        if self.match_type == "sample":
            return "value"
        if self.match_type in ("column", "fk_column"):
            return "column"
        if self.match_type == "desc":
            return "column" if self.column is not None else "table"
        return "table"   # name, alias, phrase


# Deterministic tie-break order when several fields of the same table produce
# the same max weight for a token: prefer the field kind listed first, then
# (for equal kind) the lexicographically smaller column name. This only picks
# which field is *recorded* as the representative hit — the weight itself is
# identical either way, so it cannot change any score.
_MATCH_TYPE_RANK = {"name": 0, "column": 1, "fk_column": 2, "sample": 3, "desc": 4}


def _normalize(token: str) -> str:
    """Lowercase and crudely singularize so 'tracks'/'countries' match
    'track'/'country'. Applied to both question and schema terms."""
    token = token.lower()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"          # countries -> country
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]                # tracks -> track, sales -> sale
    return token


def _tokenize(text: str) -> set[str]:
    """Split text into a set of normalized, non-stopword tokens."""
    out = set()
    for raw in re.split(r"[^a-zA-Z0-9]+", text or ""):
        if not raw:
            continue
        norm = _normalize(raw)
        if norm and raw.lower() not in _STOPWORDS and norm not in _STOPWORDS:
            out.add(norm)
    return out


def _table_term_hits(table: Table) -> dict[str, tuple[int, str, str | None]]:
    """Map each of a table's terms to (weight, match_type, column) of its best
    (max-weight) field. When several fields tie at the max weight, the field
    kept is picked deterministically (see ``_MATCH_TYPE_RANK``) — the weight
    is identical regardless of the pick, so this only fixes provenance."""
    best: dict[str, tuple[int, str, str | None]] = {}

    def _sort_key(hit: tuple[int, str, str | None]) -> tuple[int, int, str]:
        weight, match_type, column = hit
        return (-weight, _MATCH_TYPE_RANK[match_type], column or "")

    def add(text: str, weight: int, match_type: str, column: str | None) -> None:
        for tok in _tokenize(text):
            candidate = (weight, match_type, column)
            current = best.get(tok)
            if current is None or _sort_key(candidate) < _sort_key(current):
                best[tok] = candidate

    add(table.name, _W_TABLE_NAME, "name", None)
    add(table.description, _W_DESC, "desc", None)
    fk_cols = {fk.column for fk in table.foreign_keys}
    for col in table.columns:
        is_fk = col.name in fk_cols
        add(col.name, _W_FK_COLUMN if is_fk else _W_COLUMN, "fk_column" if is_fk else "column", col.name)
        add(col.description, _W_DESC, "desc", col.name)
        for value in col.sample_values:
            add(value, _W_SAMPLE, "sample", col.name)
    return best


def _table_terms(table: Table) -> dict[str, int]:
    """Map each of a table's terms to its best (max) weight."""
    return {tok: weight for tok, (weight, _, _) in _table_term_hits(table).items()}


def score_table(question_tokens: set[str], table: Table) -> int:
    """Sum the weights of a table's terms that appear in the question (term
    overlap only; alias/phrase boosts are applied in ``retrieve_tables``)."""
    terms = _table_terms(table)
    return sum(terms.get(tok, 0) for tok in question_tokens)


def lexical_evidence(question: str, tables: list[Table]) -> list[LexicalHit]:
    """Every individual signal ``lexical_scores`` sums: one term hit per
    (table, matching question token) at that token's max field weight, plus
    one hit per matching word-alias and phrase-alias. This is the single
    source of truth — ``lexical_scores`` is a fold (sum-by-table) over it —
    so other retrieval channels can consume the same evidence without
    re-deriving the scoring math."""
    q_lower = question.lower()
    q_tokens = _tokenize(question)
    hits: list[LexicalHit] = []

    for table in tables:
        term_hits = _table_term_hits(table)
        for tok in q_tokens:
            hit = term_hits.get(tok)
            if hit is None:
                continue
            weight, match_type, column = hit
            hits.append(LexicalHit(table.name, tok, weight, match_type, column))

        for tok in q_tokens:
            weight = WORD_ALIASES.get(tok, {}).get(table.name)
            if weight:
                hits.append(LexicalHit(table.name, tok, weight, "alias"))

        for phrase, mapping in PHRASE_ALIASES.items():
            if phrase not in q_lower:
                continue
            weight = mapping.get(table.name)
            if weight:
                hits.append(LexicalHit(table.name, phrase, weight, "phrase"))

    return hits


def lexical_scores(question: str, tables: list[Table]) -> dict[str, int]:
    """Lexical relevance score for every table: term overlap + domain word-alias
    boosts + phrase-alias boosts. Returns ALL tables (including 0 scores) so callers
    that fuse this with another signal can normalise across the full candidate set."""
    scores = {t.name: 0 for t in tables}
    for hit in lexical_evidence(question, tables):
        scores[hit.table] += hit.weight
    return scores


def retrieve_tables(question: str, tables: list[Table], k: int = 5) -> list[str]:
    """Return up to ``k`` table names most relevant to ``question``.

    Score = term overlap + domain word-alias boosts + phrase-alias boosts.
    Tables with no signal are excluded (an off-topic question can return
    fewer than k, possibly none — the caller decides the fallback). Ties break
    by table name for determinism.
    """
    scored = [(s, name) for name, s in lexical_scores(question, tables).items()]
    ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))
    return [name for _, name in ranked[:k]]
