"""Read a SQLite database's schema and merge in the human-written descriptions.

This is the raw material for schema linking: the schema retriever ranks tables
from here, and the SQL-generation prompt renders (a subset of) it. Descriptions
come from schema_meta, and low-cardinality columns carry sample values so the
model can link NL values ("Singapore customers", "Rock tracks") to columns.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from agent.db.schema_meta import COLUMN_DESCRIPTIONS, COLUMN_POLICIES, TABLE_DESCRIPTIONS

# A column is "low cardinality" (worth sampling for schema linking) if it has at
# most this many distinct values — captures country / genre / rating / title,
# skips emails, names and ids.
_MAX_DISTINCT_FOR_SAMPLES = 12
# show ALL values of a low-cardinality column (we only sample when distinct count
# is already <= the threshold), so common values like USA/UK aren't truncated away.
_MAX_SAMPLES = _MAX_DISTINCT_FOR_SAMPLES


@dataclass
class Column:
    name: str
    type: str
    pk: bool
    notnull: bool
    description: str = ""
    sample_values: tuple[str, ...] = ()
    policy: str = "public"


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Table:
    name: str
    description: str = ""
    row_count: int = 0
    columns: list[Column] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)


def _sample_values(conn: sqlite3.Connection, table: str, column: str) -> tuple[str, ...]:
    """Return up to _MAX_SAMPLES distinct values for a low-cardinality column,
    or () if it has too many distinct values (high cardinality / noisy)."""
    n = conn.execute(
        f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"'
    ).fetchone()[0]
    if not 1 <= n <= _MAX_DISTINCT_FOR_SAMPLES:
        return ()
    rows = conn.execute(
        f'SELECT DISTINCT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL ORDER BY "{column}" LIMIT {_MAX_SAMPLES}'
    ).fetchall()
    return tuple(str(r[0]) for r in rows)


def introspect(db_path: str | Path) -> list[Table]:
    """Return one Table per user table, with columns, FKs, descriptions, a row
    count, and sample values for low-cardinality (non-key) columns."""
    conn = sqlite3.connect(str(db_path))
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables = []
        for name in names:
            col_desc = COLUMN_DESCRIPTIONS.get(name, {})
            col_policy = COLUMN_POLICIES.get(name, {})
            fks = [
                ForeignKey(column=row[3], ref_table=row[2], ref_column=row[4])
                for row in conn.execute(f"PRAGMA foreign_key_list({name})")
            ]
            fk_cols = {fk.column for fk in fks}
            columns = []
            for row in conn.execute(f"PRAGMA table_info({name})"):
                col_name, col_type, pk = row[1], row[2] or "", bool(row[5])
                policy = col_policy.get(col_name, "public")
                # sample only non-key columns (ids/FKs aren't useful as NL values)
                # and never sample PII into schema metadata.
                samples = (
                    _sample_values(conn, name, col_name)
                    if policy != "pii" and not pk and col_name not in fk_cols
                    else ()
                )
                columns.append(
                    Column(
                        name=col_name,
                        type=col_type,
                        notnull=bool(row[3]),
                        pk=pk,
                        description=col_desc.get(col_name, ""),
                        sample_values=samples,
                        policy=policy,
                    )
                )
            row_count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            tables.append(
                Table(
                    name=name,
                    description=TABLE_DESCRIPTIONS.get(name, ""),
                    row_count=row_count,
                    columns=columns,
                    foreign_keys=fks,
                )
            )
        return tables
    finally:
        conn.close()


def render_table(table: Table) -> str:
    """Render one table as a compact, prompt-friendly schema block."""
    header = f"TABLE {table.name} ({table.row_count} rows)"
    if table.description:
        header += f"  -- {table.description}"
    lines = [header]
    for c in table.columns:
        if c.policy == "pii":
            continue
        flags = []
        if c.pk:
            flags.append("PK")
        if c.notnull:
            flags.append("NOT NULL")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        parts = [f"  {c.name} {c.type}{flag_str}"]
        if c.description:
            parts.append(f"-- {c.description}")
        if c.sample_values:
            parts.append("e.g. {" + ", ".join(c.sample_values) + "}")
        lines.append("  ".join(parts))
    for fk in table.foreign_keys:
        lines.append(f"  FK {fk.column} -> {fk.ref_table}.{fk.ref_column}")
    return "\n".join(lines)


def expand_with_fk_neighbors(tables: list[Table], requested: list[str]) -> set[str]:
    """Return ``requested`` plus its one-hop FK neighbours in BOTH directions:
    parent tables a requested table references, AND child tables that reference a
    requested one (e.g. ``invoice`` also pulls in ``customer`` and ``invoice_line``).

    This is exactly the table set ``render_schema`` shows with
    ``include_fk_neighbors=True``; exposed separately so callers — e.g. the
    retrieval-recall metric — can ask "which tables would actually reach the
    prompt" without parsing the rendered text.
    """
    requested_set = set(requested)
    names = set(requested_set)
    for t in tables:
        if t.name in requested_set:  # outgoing: parents this table references
            names.update(fk.ref_table for fk in t.foreign_keys)
        if any(fk.ref_table in requested_set for fk in t.foreign_keys):
            names.add(t.name)  # incoming: child tables referencing a requested one
    return names


def render_schema(
    tables: list[Table],
    only: list[str] | None = None,
    include_fk_neighbors: bool = False,
) -> str:
    """Render tables for the prompt.

    If ``only`` is given, render just those (used by the schema retriever to pass
    top-k tables instead of the whole DB). With ``include_fk_neighbors``, also
    render the one-hop FK neighbours in BOTH directions — parent tables this one
    references AND child tables that reference it (e.g. picking ``invoice`` also
    brings in ``customer`` and ``invoice_line``) — so the model never sees a
    foreign key into a table whose columns are missing (which invites hallucination).
    """
    if only is None:
        selected = list(tables)
    else:
        if include_fk_neighbors:
            names = expand_with_fk_neighbors(tables, only)
        else:
            names = set(only)
        selected = [t for t in tables if t.name in names]
    return "\n\n".join(render_table(t) for t in selected)
