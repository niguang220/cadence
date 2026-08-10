import tempfile
from pathlib import Path

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.semantic_layer import MetricRegistry


def _catalog():
    with tempfile.TemporaryDirectory() as d:
        tables = introspect(build(Path(d) / "s.db"))
        names = {t.name for t in tables}
        cols = {f"{t.name}.{c.name}" for t in tables for c in t.columns}
        return names, cols


def test_every_metric_declares_intrinsic_tables_and_qualified_columns():
    for m in MetricRegistry.load().metrics:
        assert m.required_tables, f"{m.name}: required_tables must be non-empty"
        assert m.required_columns, f"{m.name}: required_columns must be non-empty"
        assert all("." in c for c in m.required_columns), \
            f"{m.name}: required_columns must be fully-qualified table.column"


def test_declared_metric_tables_and_columns_exist_in_catalog():
    names, cols = _catalog()
    for m in MetricRegistry.load().metrics:
        assert set(m.required_tables) <= names, f"{m.name}: unknown required table(s)"
        assert set(m.required_columns) <= cols, f"{m.name}: unknown required column(s)"
        # every required_column's table is also in required_tables (consistency)
        assert {c.split('.')[0] for c in m.required_columns} <= set(m.required_tables), \
            f"{m.name}: a required_column references a table not in required_tables"
