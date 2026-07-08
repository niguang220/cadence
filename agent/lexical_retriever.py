"""Lexical schema retriever: pick the tables most relevant to a question.

Real databases have far too many tables to dump into a prompt, so before
generating SQL we select the top-k relevant tables. This is the deterministic
*lexical* baseline (no LLM, no embeddings): score each table by how well the
question's words overlap its name, columns, descriptions and sample values,
plus a small hand-curated layer of domain synonyms/phrases. Embedding / hybrid
retrieval is a later (Phase 4) upgrade.
"""
from __future__ import annotations

import re

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


def _table_terms(table: Table) -> dict[str, int]:
    """Map each of a table's terms to its best (max) weight."""
    terms: dict[str, int] = {}

    def add(text: str, weight: int) -> None:
        for tok in _tokenize(text):
            terms[tok] = max(terms.get(tok, 0), weight)

    add(table.name, _W_TABLE_NAME)
    add(table.description, _W_DESC)
    fk_cols = {fk.column for fk in table.foreign_keys}
    for col in table.columns:
        add(col.name, _W_FK_COLUMN if col.name in fk_cols else _W_COLUMN)
        add(col.description, _W_DESC)
        for value in col.sample_values:
            add(value, _W_SAMPLE)
    return terms


def score_table(question_tokens: set[str], table: Table) -> int:
    """Sum the weights of a table's terms that appear in the question (term
    overlap only; alias/phrase boosts are applied in ``retrieve_tables``)."""
    terms = _table_terms(table)
    return sum(terms.get(tok, 0) for tok in question_tokens)


def lexical_scores(question: str, tables: list[Table]) -> dict[str, int]:
    """Lexical relevance score for every table: term overlap + domain word-alias
    boosts + phrase-alias boosts. Returns ALL tables (including 0 scores) so callers
    that fuse this with another signal can normalise across the full candidate set."""
    q_lower = question.lower()
    q_tokens = _tokenize(question)

    boost: dict[str, int] = {}
    for tok in q_tokens:
        for name, w in WORD_ALIASES.get(tok, {}).items():
            boost[name] = boost.get(name, 0) + w
    for phrase, mapping in PHRASE_ALIASES.items():
        if phrase in q_lower:
            for name, w in mapping.items():
                boost[name] = boost.get(name, 0) + w

    return {t.name: score_table(q_tokens, t) + boost.get(t.name, 0) for t in tables}


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
