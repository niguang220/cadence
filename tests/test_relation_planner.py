from collections import defaultdict

from agent.db.build_saas_db import build
from agent.db.introspect import ForeignKey, Table, introspect, expand_with_fk_neighbors
from agent.retrieval.relation import plan_relations, legacy_one_hop_plan


def _graph(edges, extra_nodes=()):
    """edges: list of (child, parent) meaning child has an FK -> parent. Undirected in the planner."""
    fks = defaultdict(list)
    names = set(extra_nodes)
    for c, p in edges:
        names |= {c, p}
        fks[c].append(ForeignKey(column=f"{p}_id", ref_table=p, ref_column="id"))
    return [Table(name=n, foreign_keys=fks.get(n, [])) for n in sorted(names)]


def test_unique_multi_hop_path_adds_internal_bridge():
    tables = _graph([("a", "m"), ("m", "b")])
    plan = plan_relations(tables, ["a", "b"], max_hops=3)
    assert plan.strategy == "shortest_path"
    assert plan.bridges == ["m"] and set(plan.context_tables) == {"a", "b", "m"}
    assert plan.ambiguous_paths == [] and plan.unconnected_anchors == []


def test_two_equal_shortest_paths_recorded_and_union_bridged():
    tables = _graph([("a", "m1"), ("m1", "b"), ("a", "m2"), ("m2", "b")])
    plan = plan_relations(tables, ["a", "b"], max_hops=3)
    assert plan.ambiguous_paths == [["a", "m1", "b"], ["a", "m2", "b"]]
    assert set(plan.bridges) == {"m1", "m2"}
    assert set(plan.context_tables) == {"a", "b", "m1", "m2"}


def test_path_longer_than_max_hops_is_unconnected():
    tables = _graph([("a", "x"), ("x", "y"), ("y", "z"), ("z", "b")])   # 4 hops
    plan = plan_relations(tables, ["a", "b"], max_hops=3)
    # Union-find semantics: with no path <= max_hops between a and b, each anchor is its own
    # component. The "primary" component is the one containing the lexicographically-smallest
    # anchor ("a"); every anchor OUTSIDE that component is unconnected -- "a" itself is inside
    # its own (primary) component, so only "b" is flagged.
    assert plan.unconnected_anchors == ["b"] and plan.bridges == []


def test_two_disconnected_multinode_components_flags_secondary():
    # anchors a,b (joined via m1) and c,d (joined via m2); the two components are disconnected.
    # Semantics: the "primary" component is the one containing the lexicographically-smallest anchor
    # (a) -> {a,b}. c and d are unconnected EVEN THOUGH each connects to a peer in its own component.
    tables = _graph([("a", "m1"), ("m1", "b"), ("c", "m2"), ("m2", "d")])
    plan = plan_relations(tables, ["a", "b", "c", "d"], max_hops=3)
    assert plan.unconnected_anchors == ["c", "d"]
    assert {"m1", "m2"} <= set(plan.bridges)          # within-component bridges still rendered
    assert set(plan.context_tables) == {"a", "b", "c", "d", "m1", "m2"}


def test_disconnected_anchor_is_flagged():
    tables = _graph([("a", "b")], extra_nodes=("c",))
    plan = plan_relations(tables, ["a", "b", "c"], max_hops=3)
    assert plan.unconnected_anchors == ["c"]
    assert set(plan.context_tables) == {"a", "b", "c"}


def test_single_anchor_has_no_bridges_or_unconnected():
    tables = _graph([("a", "b")])
    plan = plan_relations(tables, ["a"], max_hops=3)
    assert plan.bridges == [] and plan.unconnected_anchors == [] and plan.context_tables == ["a"]


def test_planner_is_deterministic():
    tables = _graph([("a", "m1"), ("m1", "b"), ("a", "m2"), ("m2", "b")])
    p1 = plan_relations(tables, ["a", "b"], max_hops=3)
    p2 = plan_relations(tables, ["b", "a"], max_hops=3)
    assert (p1.context_tables, p1.bridges, p1.ambiguous_paths, p1.unconnected_anchors) == \
           (p2.context_tables, p2.bridges, p2.ambiguous_paths, p2.unconnected_anchors)


def test_context_equals_anchors_union_bridges_on_real_db(tmp_path):
    tables = introspect(str(build(tmp_path / "saas.db")))
    plan = plan_relations(tables, ["invoice_line", "account"], max_hops=3)
    assert set(plan.context_tables) == set(plan.anchors) | set(plan.bridges)
    assert all(b not in plan.anchors for b in plan.bridges)


def test_legacy_closure_matches_expand_and_no_bridges(tmp_path):
    tables = introspect(str(build(tmp_path / "saas.db")))
    plan = legacy_one_hop_plan(tables, ["subscription"])
    assert plan.strategy == "legacy_one_hop" and plan.bridges == []
    assert set(plan.context_tables) == expand_with_fk_neighbors(tables, ["subscription"])
