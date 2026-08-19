from __future__ import annotations

from collections import defaultdict

from agent.db.introspect import Table
from agent.retrieval.contracts import RelationEdge, RelationPlan


def _adjacency(tables: list[Table]) -> dict[str, list[str]]:
    names = {t.name for t in tables}
    adj: dict[str, set[str]] = defaultdict(set)
    for t in tables:
        for fk in t.foreign_keys:
            if fk.ref_table in names:
                adj[t.name].add(fk.ref_table)
                adj[fk.ref_table].add(t.name)   # undirected: a bridge is traversable either way
    return {k: sorted(v) for k, v in adj.items()}


def _shortest_paths(adj, src: str, dst: str, max_hops: int) -> list[list[str]]:
    """All shortest simple paths (as node lists) from src to dst using <= max_hops edges."""
    if src == dst:
        return [[src]]
    frontier = [[src]]
    for _ in range(max_hops):
        found, nxt = [], []
        for path in frontier:
            for nb in adj.get(path[-1], []):
                if nb in path:
                    continue
                newpath = path + [nb]
                (found if nb == dst else nxt).append(newpath)
        if found:
            return sorted(found)              # deterministic; all are same (shortest) length
        frontier = nxt
    return []


def _physical_edges_between(by_name: dict[str, Table], x: str, y: str) -> list[RelationEdge]:
    out = []
    for fk in by_name[x].foreign_keys:
        if fk.ref_table == y:
            out.append(RelationEdge(x, fk.column, y, fk.ref_column, "physical_fk"))
    for fk in by_name[y].foreign_keys:
        if fk.ref_table == x:
            out.append(RelationEdge(y, fk.column, x, fk.ref_column, "physical_fk"))
    return out


def _edges_along(by_name, paths) -> list[RelationEdge]:
    seen, out = set(), []
    for path in paths:
        for a, b in zip(path, path[1:]):
            for e in _physical_edges_between(by_name, a, b):
                key = (e.from_table, e.from_column, e.to_table, e.to_column)
                if key not in seen:
                    seen.add(key)
                    out.append(e)
    return sorted(out, key=lambda e: (e.from_table, e.from_column, e.to_table, e.to_column))


def plan_relations(tables: list[Table], anchors: list[str], *, max_hops: int = 3) -> RelationPlan:
    by_name = {t.name: t for t in tables}
    anchors = sorted(set(anchors))
    adj = _adjacency(tables)
    bridges: set[str] = set()
    ambiguous: list[list[str]] = []
    accepted_paths: list[list[str]] = []

    # Union-find over anchors: union u,v when a shortest path <= max_hops exists between
    # them. The resulting components partition the anchors; the "primary" component is the
    # one containing the lexicographically-smallest anchor, and every anchor OUTSIDE it is
    # unconnected -- even if it connects to a peer inside its own (non-primary) component.
    # Bridges/ambiguous/edges are unaffected: within-component bridges are still rendered
    # (recall-first).
    parent = {a: a for a in anchors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, u in enumerate(anchors):
        for v in anchors[i + 1:]:
            paths = _shortest_paths(adj, u, v, max_hops)
            if not paths:
                continue
            union(u, v)
            accepted_paths.extend(paths)
            for p in paths:
                bridges.update(p[1:-1])
            if len(paths) > 1:
                ambiguous.extend(paths)

    if len(anchors) >= 2:
        primary = find(min(anchors))
        unconnected = sorted(a for a in anchors if find(a) != primary)
    else:
        unconnected = []

    bridges -= set(anchors)
    context = sorted(set(anchors) | bridges)
    return RelationPlan(
        strategy="shortest_path", anchors=anchors, bridges=sorted(bridges),
        context_tables=context, edges=_edges_along(by_name, accepted_paths),
        unconnected_anchors=unconnected, ambiguous_paths=sorted(ambiguous))
