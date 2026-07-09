"""SQL safety gate: only let read-only SELECT queries through.

The agent generates SQL from an LLM, so before we ever execute it we parse it
with a real SQL parser (sqlglot, not regex) and reject anything that isn't a
single read-only query: no INSERT/UPDATE/DELETE/DDL, no multiple statements,
no ATTACH/PRAGMA, no dangerous functions. Execution then adds a second line of
defence (a read-only connection), but this gate is the first.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

# Read-only query roots we allow (built defensively so it works across sqlglot
# versions that may not define every set-operation node).
_ALLOWED_ROOTS = tuple(
    t for t in (
        getattr(exp, "Select", None),
        getattr(exp, "Union", None),
        getattr(exp, "Intersect", None),
        getattr(exp, "Except", None),
    ) if t is not None
)

# Statement types that mutate data/schema or run engine commands (ATTACH, PRAGMA,
# VACUUM parse as Command). If any appears anywhere in the tree, reject.
_FORBIDDEN = tuple(
    t for t in (
        getattr(exp, "Insert", None),
        getattr(exp, "Update", None),
        getattr(exp, "Delete", None),
        getattr(exp, "Drop", None),
        getattr(exp, "Create", None),
        getattr(exp, "Alter", None),
        getattr(exp, "Command", None),
    ) if t is not None
)

_FORBIDDEN_FUNCS = {"load_extension"}


@dataclass
class SafetyResult:
    safe: bool
    reason: str = ""


def check_sql_safety(sql: str) -> SafetyResult:
    """Return whether ``sql`` is a single read-only query, and why not if it isn't."""
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1]  # tolerate ONE trailing semicolon (LLMs emit them); real
                        # multi-statements still parse to >1 and are rejected below
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except Exception as e:  # sqlglot raises ParseError and friends
        return SafetyResult(False, f"could not parse SQL: {e}")

    if len(statements) != 1:
        return SafetyResult(False, f"expected exactly one statement, got {len(statements)}")

    stmt = statements[0]
    if not isinstance(stmt, _ALLOWED_ROOTS):
        return SafetyResult(
            False, f"only read-only SELECT queries are allowed, got {type(stmt).__name__}"
        )

    forbidden = next(iter(stmt.find_all(*_FORBIDDEN)), None) if _FORBIDDEN else None
    if forbidden is not None:
        return SafetyResult(False, f"forbidden statement: {type(forbidden).__name__}")

    # table-valued PRAGMA functions (pragma_table_info(...), pragma_database_list)
    # parse as ordinary SELECTs; reject them here too, not only at execution.
    for table in stmt.find_all(exp.Table):
        if (table.name or "").lower().startswith("pragma_"):
            return SafetyResult(False, f"forbidden pragma function: {table.name}")

    for func in stmt.find_all(exp.Anonymous):
        name = str(func.this).lower()
        if name in _FORBIDDEN_FUNCS or name.startswith("pragma_"):
            return SafetyResult(False, f"forbidden function: {func.this}")

    return SafetyResult(True)
