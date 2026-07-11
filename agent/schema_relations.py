"""Deterministic FK-graph join paths among the recalled tables -- the real work of the
SchemaRecall/TableRelation split (the recall half is an unchanged refactor). Zero LLM."""
from __future__ import annotations

from agent.db.introspect import Table


def join_paths(tables: list[Table], recalled: list[str]) -> list[dict]:
    """Direct FK edges whose both ends are in ``recalled`` -- a planner/SQL hint, not
    an exhaustive path set (multi-hop paths through non-recalled tables are not
    enumerated) and not a basis for a hard refusal."""
    recalled_set = set(recalled)
    by_name = {t.name: t for t in tables}
    paths = []
    for name in recalled:
        t = by_name.get(name)
        if not t:
            continue
        for fk in t.foreign_keys:
            if fk.ref_table in recalled_set:
                paths.append({"from": name, "to": fk.ref_table, "on": fk.column})
    return paths
