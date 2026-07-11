"""detect_ambiguity: a cheap heuristic that catches a question whose answer
depends on an unstated choice, so the agent can ASK instead of guessing.

It fires when the question uses a vague quality word ("best", "popular", ...)
without naming a metric to rank by. Most text2SQL agents just guess here; asking
is both more correct and a differentiator (AmbiSQL: 42.5% -> 92.5% on ambiguous
questions with clarification).

Deliberately conservative: a missed ambiguity just falls through to a normal
answer, but a false clarification annoys the user -- so we only flag a clearly
*unqualified* superlative. If a metric word appears anywhere ("best-SELLING",
"highest-RATED", "most ORDERS"), the question is treated as specific.
"""
from __future__ import annotations

import re
from typing import Any

from agent.db.introspect import Table
from agent.semantic_layer import MetricDef

# Inherently vague quality words: each needs "by what measure?" to be answerable.
_VAGUE = ("best", "worst", "popular", "good", "successful", "valuable",
          "important", "engaged", "active", "top")

# A metric word OR a concrete ranking dimension makes the question specific:
# "best-SELLING", "top 5 LONGEST", "HIGHEST-priced" all name what to rank by. The
# dimension words matter especially for "top", which is otherwise vague ("top
# tracks") but specific once a dimension follows ("top longest tracks").
_METRIC_HINTS = ("spend", "spent", "revenue", "sales", "sold", "selling", "rating",
                 "rated", "count", "number", "total", "amount", "price", "quantity",
                 "order", "invoice", "frequency", "duration", "length", "play",
                 "longest", "shortest", "highest", "lowest", "newest", "oldest",
                 "latest", "earliest", "largest", "smallest")


def detect_ambiguity(question: str) -> str | None:
    """Return a clarification question if the wording is ambiguous, else None."""
    q = question.lower()
    vague = next((w for w in _VAGUE if re.search(rf"\b{w}\b", q)), None)
    if vague and not any(hint in q for hint in _METRIC_HINTS):
        # message is intentionally entity-neutral (not schema-aware): the value is
        # that the agent asks at all. Tailoring options to the entity is a future upgrade.
        return (f'"{vague}" can be measured several ways (e.g. by a total amount, '
                f'by a count, or by a rating). Which metric do you mean?')
    return None


def _numeric_measure_columns(tables: list[Table]) -> list[tuple[str, str, str]]:
    """Return (table, col_name, pretty_label) for aggregatable numeric non-key columns.

    Scans actual column types rather than matching against a fixed name set, so it
    works on any schema, not just the demo database.
    """
    _NUMERIC = ("INT", "REAL", "FLOAT", "NUMERIC", "DECIMAL", "DOUBLE", "MONEY")
    found = []
    for table in tables:
        fk_cols = {fk.column for fk in table.foreign_keys}
        for col in table.columns:
            if col.policy == "pii" or col.pk or col.name in fk_cols:
                continue
            if any(t in col.type.upper() for t in _NUMERIC):
                label = col.name.replace("_", " ").title()
                found.append((table.name, col.name, label))
    return found


def build_clarification_options(
    question: str,
    *,
    tables: list[Table] | None = None,
    metrics: list[MetricDef] | None = None,
) -> list[dict[str, Any]]:
    """Return serializable metric options that can be shown to a user.

    Governed semantic metrics always take priority. When none match, the schema
    fallback scans for actual numeric columns by type — not by a hardcoded name
    set — so it works on any schema, not just the demo database.
    """
    options: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        label: str,
        value: str,
        detail: str,
        source: str,
        intent: dict[str, str] | None = None,
        confidence: str = "fallback",
    ) -> None:
        key = label.lower()
        if key not in seen:
            option: dict[str, Any] = {
                "label": label,
                "value": value,
                "detail": detail,
                "source": source,
                "confidence": confidence,
            }
            if intent:
                option["intent"] = intent
            options.append(option)
            seen.add(key)

    for metric in metrics or []:
        label = metric.aliases[0] if metric.aliases else metric.name.replace("_", " ")
        add(
            label if label.isupper() else label.title(),
            metric.name.replace("_", " "),
            metric.measure,
            "semantic_layer",
            {
                "metric": metric.name,
                "aggregation": "governed",
                "measure": metric.measure,
                "grain": metric.grain,
                "sort": "desc",
                "source": "semantic_layer",
            },
            confidence="governed",
        )

    if options:
        return options

    if tables is None:
        return options

    # Schema-agnostic fallback: scan actual numeric columns by type.
    # Hints rank within real column names; they do not gate on fixed strings.
    _SUM_HINTS = ("total", "amount", "revenue", "sales", "price", "spend", "cost", "value")
    _AVG_HINTS = ("rating", "score", "grade", "star")
    measure_cols = _numeric_measure_columns(tables)

    sum_col = next(
        (c for c in measure_cols if any(h in c[1].lower() for h in _SUM_HINTS)),
        measure_cols[0] if measure_cols else None,
    )
    if sum_col:
        table, col, label = sum_col
        add(label, "total", f"SUM({table}.{col})", "schema",
            {"metric": col, "aggregation": "sum", "measure": f"{table}.{col}",
             "sort": "desc", "source": "schema"})

    add("Count", "count", "COUNT(*)", "schema",
        {"metric": "count", "aggregation": "count", "measure": "*",
         "sort": "desc", "source": "schema"})

    avg_col = next(
        (c for c in measure_cols if any(h in c[1].lower() for h in _AVG_HINTS)),
        None,
    )
    if avg_col:
        table, col, label = avg_col
        add(label, "average", f"AVG({table}.{col})", "schema",
            {"metric": col, "aggregation": "avg", "measure": f"{table}.{col}",
             "sort": "desc", "source": "schema"})

    return options


def format_clarification_prompt(clarification: str, options: list[dict[str, Any]]) -> str:
    """Append available schema/semantic options without changing old callers."""
    if not options:
        return clarification
    rendered = "\n".join(f"- {o['label']}: {o['detail']}" for o in options)
    return f"{clarification}\nOptions:\n{rendered}"


def normalize_clarification_response(response: str) -> str:
    """Turn short HITL replies into generation-ready ranking instructions."""
    raw = response.strip()
    if not raw:
        return ""

    text = raw.lower()
    if any(hint in text for hint in ("count", "number", "quantity", "volume", "orders")):
        return "rank the requested entities by count, descending"
    if any(hint in text for hint in ("rating", "rated", "score")):
        return "rank the requested entities by rating or score, descending"
    if any(hint in text for hint in ("sale", "revenue", "spend", "spent", "amount", "total")):
        return "rank the requested entities by total sales or revenue, descending"
    if text.startswith("by ") and raw[3:].strip():
        return f"rank the requested entities by {raw[3:].strip()}, descending"
    return raw


def parse_clarification_intent(
    response: str,
    options: list[dict[str, Any]] | None = None,
) -> dict[str, str] | None:
    """Map a clarification reply to a deterministic, serializable metric intent."""
    raw = response.strip()
    if not raw:
        return None

    text = raw.lower()
    for option in options or []:
        candidates = {
            str(option.get("label", "")).lower(),
            str(option.get("value", "")).lower(),
        }
        if option.get("intent") and any(text == c or text in c or c in text for c in candidates if c):
            intent = dict(option["intent"])
            intent["raw_response"] = raw
            return intent

    if any(hint in text for hint in ("count", "number", "quantity", "volume", "orders")):
        return {
            "metric": "count",
            "aggregation": "count",
            "measure": "* or distinct entity id",
            "sort": "desc",
            "source": "freeform",
            "raw_response": raw,
        }
    if any(hint in text for hint in ("rating", "rated", "score")):
        return {
            "metric": "rating",
            "aggregation": "avg",
            "measure": "rating or score column",
            "sort": "desc",
            "source": "freeform",
            "raw_response": raw,
        }
    if any(hint in text for hint in ("sale", "revenue", "spend", "spent", "amount", "total")):
        return {
            "metric": "total_sales",
            "aggregation": "sum",
            "measure": "sales, revenue, amount, or total column",
            "sort": "desc",
            "source": "freeform",
            "raw_response": raw,
        }
    if text.startswith("by ") and raw[3:].strip():
        return {
            "metric": raw[3:].strip(),
            "aggregation": "unspecified",
            "measure": raw[3:].strip(),
            "sort": "desc",
            "source": "freeform",
            "raw_response": raw,
        }
    return None


def format_clarification_intent(intent: dict[str, str] | None) -> str:
    """Render typed clarification intent as prompt guidance for SQL generation."""
    if not intent:
        return ""
    lines = [
        "Use this resolved clarification intent:",
        f"- metric: {intent.get('metric', '')}",
        f"- aggregation: {intent.get('aggregation', '')}",
        f"- measure: {intent.get('measure', '')}",
        f"- sort: {intent.get('sort', 'desc')}",
    ]
    grain = intent.get("grain")
    if grain:
        lines.append(f"- grain: {grain}")
    return "\n".join(lines) + "\n\n"
