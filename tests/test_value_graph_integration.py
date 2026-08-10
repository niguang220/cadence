"""The graph stays ES-blind: it passes only a RetrievalConfig; the pipeline owns value-backend
construction. A value hit reaches the rendered schema when a fake backend is injected at the
pipeline boundary the graph calls."""
import inspect

import agent.graph as graph
from agent.db.introspect import Column, Table
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.pipeline import run_retrieval as _real_run_retrieval
from agent.retrieval.serde import serialize_config
from agent.retrieval.value_backend import FakeValueBackend, ValueDoc


def _vtables():
    return [
        Table(name="product", columns=[
            Column(name="product_id", type="INTEGER", pk=True, notnull=True),
            Column(name="name", type="TEXT", pk=False, notnull=False, policy="searchable")]),
        Table(name="orders", columns=[
            Column(name="order_id", type="INTEGER", pk=True, notnull=True),
            Column(name="qty", type="INTEGER", pk=False, notnull=False)]),
    ]


def test_graph_routes_value_hits_via_injected_backend(monkeypatch):
    b = FakeValueBackend()
    b.ensure_index()
    b.upsert([ValueDoc("d1", "product", "name", "Widget")])
    monkeypatch.setattr(graph, "run_retrieval",
                        lambda q, t, c, *, k, metric_hits=None, value_backend=None:
                        _real_run_retrieval(q, t, c, k=k, metric_hits=metric_hits, value_backend=b))
    state = {"question": "do we sell Widget", "tables": _vtables(), "k": 5,
             "retrieval_config_serialized": serialize_config(RetrievalConfig.value_ablation())}
    s = graph._schema_recall(state)
    assert "product" in s["retrieved_tables"]
    assert "TABLE product" in s["schema"]


def test_graph_module_is_elasticsearch_blind():
    src = inspect.getsource(graph)
    assert "elasticsearch" not in src.lower()
    assert "ElasticsearchValueBackend" not in src
