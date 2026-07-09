"""Tools the model can call while generating SQL.

Right now there is one: ``get_schema``. The deterministic retriever picks the
tables it thinks are relevant, but it can miss one (a bridge table, a table whose
name doesn't lexically match the question). ``get_schema`` lets the model pull in
any table by name before writing SQL, so a schema-linking miss is *recoverable*
instead of fatal -- and it makes the agent genuinely tool-calling, not a fixed
prompt template.

The tool is read-only: it only renders schema text. It validates names so a typo
comes back as a helpful message instead of nothing.
"""
from __future__ import annotations

from langchain_core.tools import tool

from agent.db.introspect import Table, render_schema


def build_get_schema_tool(tables: list[Table], requested: list[str]):
    """Build a ``get_schema`` tool bound to this database's tables.

    ``requested`` is a list the tool appends to, so the caller can record which
    tables the model asked for (useful in the trace and in tests).
    """
    known = {t.name for t in tables}

    @tool
    def get_schema(table_names: list[str]) -> str:
        """Return the CREATE statement and a few sample rows for each named table.

        Call this when the schema you were given is missing a table you need to
        answer the question. Pass exact table names.
        """
        requested.extend(table_names)
        unknown = [n for n in table_names if n not in known]
        if unknown:
            return (f"Unknown table(s): {', '.join(unknown)}. "
                    f"Available tables: {', '.join(sorted(known))}.")
        return render_schema(tables, only=table_names, include_fk_neighbors=False)

    return get_schema
