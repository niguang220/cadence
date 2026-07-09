"""Column-level data-governance checks.

The SQL safety gate answers "is this read-only?" This module answers a separate
question: "does this query touch columns our policy forbids exposing to the
model/user?" v1 is deliberately conservative and blocks any reference to PII,
including filters and aggregates over PII-derived values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from agent.db.introspect import Table

DENIED_POLICIES = {"pii"}


@dataclass
class GovernanceResult:
    ok: bool
    reason: str = ""
    columns: list[str] = field(default_factory=list)


def _denied_columns(tables: list[Table], denied: set[str]) -> dict[tuple[str, str], str]:
    out = {}
    for table in tables:
        for col in table.columns:
            if col.policy in denied:
                out[(table.name.lower(), col.name.lower())] = col.policy
    return out


def _alias_map(tree: exp.Expression, tables: list[Table]) -> dict[str, set[str]]:
    known = {t.name.lower() for t in tables}
    aliases: dict[str, set[str]] = {}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name not in known:
            continue
        aliases.setdefault(name, set()).add(name)
        aliases.setdefault(table.alias_or_name.lower(), set()).add(name)
    return aliases


def _table_has_denied(table: str, denied_cols: dict[tuple[str, str], str]) -> bool:
    table = table.lower()
    return any(t == table for t, _ in denied_cols)


def _format_col(table: str, column: str) -> str:
    return f"{table}.{column}" if table else column


def check_sql_governance(
    sql: str,
    tables: list[Table],
    *,
    denied_policies: set[str] | None = None,
) -> GovernanceResult:
    """Reject SQL that references denied columns.

    Bare columns are resolved conservatively: if any table in the query exposes a
    denied column with that name, the query is blocked. ``COUNT(*)`` is allowed,
    but ``SELECT *``/``alias.*`` is blocked when it would include PII.
    """
    denied = denied_policies or DENIED_POLICIES
    denied_cols = _denied_columns(tables, denied)
    if not denied_cols:
        return GovernanceResult(True)

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception as e:
        return GovernanceResult(False, f"could not parse SQL for governance: {e}")

    aliases = _alias_map(tree, tables)
    query_tables = {name for names in aliases.values() for name in names}
    blocked: set[str] = set()

    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        qualifier = (col.table or "").lower()
        if name == "*":
            candidate_tables = aliases.get(qualifier, {qualifier} if qualifier else set())
            for table in candidate_tables:
                if table and _table_has_denied(table, denied_cols):
                    blocked.add(f"{table}.*")
            continue

        if qualifier:
            candidate_tables = aliases.get(qualifier, {qualifier})
            for table in candidate_tables:
                if (table, name) in denied_cols:
                    blocked.add(_format_col(table, name))
            continue

        # Unqualified column: block if any in-scope schema table has this denied
        # column. This errs on the side of refusing instead of silently leaking.
        for table in query_tables:
            if (table, name) in denied_cols:
                blocked.add(_format_col(table, name))

    for star in tree.find_all(exp.Star):
        if isinstance(star.parent, exp.Count):
            continue
        if isinstance(star.parent, exp.Column):
            continue
        for table in query_tables:
            if _table_has_denied(table, denied_cols):
                blocked.add(f"{table}.*")

    if blocked:
        cols = sorted(blocked)
        return GovernanceResult(
            False,
            "query references blocked PII columns: " + ", ".join(cols),
            cols,
        )
    return GovernanceResult(True)


def check_result_governance(
    columns: list[str],
    tables: list[Table],
    *,
    denied_policies: set[str] | None = None,
) -> GovernanceResult:
    """Conservative answer-layer fallback based on output column names."""
    denied = denied_policies or DENIED_POLICIES
    denied_names = {
        col.name.lower()
        for table in tables
        for col in table.columns
        if col.policy in denied
    }
    blocked = sorted(c for c in columns if c.lower() in denied_names)
    if blocked:
        return GovernanceResult(
            False,
            "result contains blocked PII columns: " + ", ".join(blocked),
            blocked,
        )
    return GovernanceResult(True)
