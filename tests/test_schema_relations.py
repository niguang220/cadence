from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.schema_relations import join_paths


def test_join_paths_finds_fk_edge_between_recalled_tables(tmp_path):
    tables = introspect(build(tmp_path / "s.db"))
    names = [t.name for t in tables]
    # pick a child+parent pair known to share an FK
    paths = join_paths(tables, names)
    # each path is a COMPLETE join condition (from.on = to.ref_on), so the LLM never has
    # to guess the parent column even when the two column names differ.
    assert paths and all({"from", "to", "on", "ref_on"} <= set(p) for p in paths)


def test_no_join_path_when_tables_unrelated(tmp_path):
    tables = introspect(build(tmp_path / "s.db"))
    # a single table has no join path to anything in the recalled set
    solo = [tables[0].name]
    assert join_paths(tables, solo) == []


def test_join_path_carries_both_columns_when_fk_names_differ():
    # a synthetic FK whose two column names DIFFER -- the SaaS fixture happens to use the
    # same name both ends, so this pins that ``ref_on`` is the parent column, not the child.
    from agent.db.introspect import Table, ForeignKey
    child = Table(name="child", foreign_keys=[ForeignKey("parent_ref", "parent", "id")])
    parent = Table(name="parent")
    assert join_paths([child, parent], ["child", "parent"]) == [
        {"from": "child", "to": "parent", "on": "parent_ref", "ref_on": "id"}]


def test_table_relation_renders_a_complete_join_condition():
    from agent.db.introspect import Table, ForeignKey
    from agent.graph import _table_relation
    child = Table(name="child", foreign_keys=[ForeignKey("parent_ref", "parent", "id")])
    out = _table_relation({"tables": [child, Table(name="parent")],
                           "retrieved_tables": ["child", "parent"], "schema": "S"})
    assert "child.parent_ref = parent.id" in out["schema"]   # from.on = to.ref_on
