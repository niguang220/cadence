"""Value-index governance: which columns may leave the database into a value-search index.

Fail-closed. Only columns explicitly marked ``policy == "searchable"`` are indexable; ``public``
and unknown columns are NOT indexed by default; ``pii`` is NEVER indexable. This module is the
single source of truth for the value-index allowlist — the ingestion (value_index) and the ES
backend must obey it, and it validates the policy config before anything is indexed so a typo
fails loudly instead of silently exposing data.
"""
from __future__ import annotations

from agent.db.introspect import Table
from agent.db.schema_meta import COLUMN_POLICIES

INDEXABLE_POLICY = "searchable"
_VALID_POLICIES = {"public", "pii", "searchable"}


def validate_value_policy(tables: list[Table]) -> None:
    """Fail-fast on a misconfigured governance policy, before any value is indexed:

    1. every resolved column policy is a known value (a typo like ``"piii"`` must not fall back to
       ``public`` and expose data);
    2. every ``COLUMN_POLICIES`` entry for a table present in this catalog names a real column (a
       typo'd column name would leave the intended PII column unmarked and eligible for indexing).
    """
    catalog = {t.name: {c.name for c in t.columns} for t in tables}
    for t in tables:
        for c in t.columns:
            if c.policy not in _VALID_POLICIES:
                raise ValueError(
                    f"{t.name}.{c.name}: invalid column policy {c.policy!r} "
                    f"(must be one of {sorted(_VALID_POLICIES)})")
    for table, cols in COLUMN_POLICIES.items():
        if table not in catalog:
            continue                      # a policy for a different fixture/db — not our concern here
        for col in cols:
            if col not in catalog[table]:
                raise ValueError(
                    f"COLUMN_POLICIES[{table!r}][{col!r}]: column does not exist in the catalog "
                    f"(a policy typo would silently unmark data)")


def resolve_searchable_columns(tables: list[Table]) -> list[tuple[str, str]]:
    """The value-index allowlist: sorted ``(table, column)`` for every column explicitly marked
    ``searchable``. Validates first, so a bad policy fails closed rather than indexing the wrong
    thing. ``pii``/``public``/unknown columns are never returned."""
    validate_value_policy(tables)
    return sorted((t.name, c.name) for t in tables for c in t.columns
                  if c.policy == INDEXABLE_POLICY)
