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
