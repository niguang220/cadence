"""Run a SQL query against a SQLite database with three layers of defence.

1. The **safety gate** (parser) — ``run_query`` calls ``check_sql_safety`` by
   default, so a caller can't forget it (pass ``assume_safe=True`` only if the
   SQL was already vetted upstream).
2. A SQLite **authorizer** — rejects writes / schema changes / ATTACH / PRAGMA /
   load_extension at execution time, catching anything the parser missed
   (a read-only connection alone does NOT block ATTACH's file side effect).
3. A **read-only** connection — last-ditch guard against writes to the main DB.

Plus an output row cap and a wall-clock timeout. Returns a structured result
(columns + rows, or an error) instead of raising, so the agent can branch on it.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.db.introspect import Table
from agent.governance import check_result_governance, check_sql_governance
from agent.safety import check_sql_safety

DEFAULT_MAX_ROWS = 1000
DEFAULT_TIMEOUT_SECONDS = 5.0

# SQLite authorizer action codes that must never run in a read-only query engine.
# Built defensively so it works across Python versions / build options.
_DENIED_ACTIONS = {
    getattr(sqlite3, name)
    for name in (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE",
        "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_INDEX", "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_VIEW", "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_DROP_TABLE", "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_INDEX", "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_VIEW", "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER", "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_ALTER_TABLE", "SQLITE_ATTACH", "SQLITE_DETACH", "SQLITE_PRAGMA",
        "SQLITE_ANALYZE", "SQLITE_REINDEX",
        "SQLITE_CREATE_VTABLE", "SQLITE_DROP_VTABLE",
        "SQLITE_TRANSACTION", "SQLITE_SAVEPOINT",
    )
    if hasattr(sqlite3, name)
}


def _readonly_authorizer(action, arg1, arg2, db_name, trigger):
    if action in _DENIED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


@dataclass
class ExecutionResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False           # True if more rows than max_rows were available
    error: str = ""


def run_query(
    db_path: str | Path,
    sql: str,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    assume_safe: bool = False,
    tables: list[Table] | None = None,
) -> ExecutionResult:
    """Execute ``sql`` read-only against ``db_path`` and return rows or an error.

    ``max_rows`` caps the rows handed back (an OUTPUT cap, not a query-cost cap —
    SQLite may still compute a full ORDER BY/GROUP BY); the timeout is the cost
    guard. Unless ``assume_safe``, the SQL is run through ``check_sql_safety``
    first and rejected if it isn't a single read-only query. Governance checks
    still run whenever ``tables`` are supplied; ``assume_safe`` does not waive
    column policy.
    """
    if max_rows < 1:
        return ExecutionResult(False, error="max_rows must be >= 1")
    if timeout_seconds <= 0:
        return ExecutionResult(False, error="timeout_seconds must be > 0")
    if not assume_safe:
        safety = check_sql_safety(sql)
        if not safety.safe:
            return ExecutionResult(False, error=f"unsafe SQL: {safety.reason}")
    if tables is not None:
        governance = check_sql_governance(sql, tables)
        if not governance.ok:
            return ExecutionResult(False, error=f"governance violation: {governance.reason}")

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        return ExecutionResult(False, error=f"could not open database read-only: {e}")

    deadline = time.monotonic() + timeout_seconds
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    conn.set_authorizer(_readonly_authorizer)
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        fetched = cur.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = [tuple(r) for r in fetched[:max_rows]]
        if tables is not None:
            governance = check_result_governance(columns, tables)
            if not governance.ok:
                return ExecutionResult(False, error=f"governance violation: {governance.reason}")
        return ExecutionResult(True, columns=columns, rows=rows, truncated=truncated)
    except sqlite3.Error as e:
        return ExecutionResult(False, error=str(e))
    finally:
        conn.close()
