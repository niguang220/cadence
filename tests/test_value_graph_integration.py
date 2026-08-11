"""The graph stays ES-blind: it passes only a RetrievalConfig; the pipeline owns value-backend
construction. It also OWNS the query split: lexical/dense read the enhanced rewrite, value linking
reads the ORIGINAL user question, so a model-invented entity in the rewrite can never drive a value
admission. A value hit reaches the rendered schema when a fake backend is injected at the runtime
seam the graph reads."""
import inspect

import agent.graph as graph
from agent.db.build_value_db import build as build_value_db
from agent.db.introspect import Column, Table, introspect
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.pipeline import run_retrieval as _real_run_retrieval
from agent.retrieval.serde import deserialize_result, serialize_config
from agent.retrieval.value_backend import FakeValueBackend, ValueDoc
from agent.retrieval.value_index import build_value_index

# the two REAL bad rewrites query_enhance produced against DeepSeek (space after the CJK entity gone)
_BAD_REWRITES = {
    "上海云图信息技术 有几个未解决的工单?": "上海云图信息技术有限公司有多少个未解决的工单？",
    "广州天河数据服务 有几份合同?": "广州天河数据服务公司共有多少份合同？",
}


def _vtables():
    return [
        Table(name="product", columns=[
            Column(name="product_id", type="INTEGER", pk=True, notnull=True),
            Column(name="name", type="TEXT", pk=False, notnull=False, policy="searchable")]),
        Table(name="orders", columns=[
            Column(name="order_id", type="INTEGER", pk=True, notnull=True),
            Column(name="qty", type="INTEGER", pk=False, notnull=False)]),
    ]


def _state(question, tables, backend, *, enhanced=None):
    st = {"question": question, "tables": tables, "k": 5, "value_backend": backend,
          "retrieval_config_serialized": serialize_config(RetrievalConfig.value_ablation())}
    if enhanced is not None:
        st["enhanced_question"] = enhanced
    return st


def test_graph_routes_value_hits_via_injected_backend():
    b = FakeValueBackend()
    b.ensure_index()
    b.upsert([ValueDoc("d1", "product", "name", "Widget")])
    s = graph._schema_recall(_state("do we sell Widget", _vtables(), b))
    assert "product" in s["retrieved_tables"]
    assert "TABLE product" in s["schema"]


def test_graph_splits_enhanced_for_lexdense_and_original_for_value(monkeypatch):
    captured = {}

    def spy(q, t, c, *, k, metric_hits=None, value_backend=None, value_query=None):
        captured["question"] = q          # lexical/dense input
        captured["value_query"] = value_query
        return _real_run_retrieval(q, t, c, k=k, metric_hits=metric_hits,
                                   value_backend=value_backend, value_query=value_query)

    b = FakeValueBackend()
    b.ensure_index()
    b.upsert([ValueDoc("d1", "product", "name", "Widget")])
    monkeypatch.setattr(graph, "run_retrieval", spy)
    graph._schema_recall(_state("do we sell Widget", _vtables(), b,
                                enhanced="please enumerate every product named Widget"))
    assert captured["question"] == "please enumerate every product named Widget"   # enhanced
    assert captured["value_query"] == "do we sell Widget"                          # original


def _value_db(tmp_path):
    db = build_value_db(tmp_path / "v.db")
    tables = introspect(db)
    b = FakeValueBackend()
    build_value_index(tables, db, b)
    return tables, b


def test_two_cjk_cases_recover_recall_with_original_question(tmp_path):
    tables, b = _value_db(tmp_path)
    for original, bad in _BAD_REWRITES.items():
        s = graph._schema_recall(_state(original, tables, b, enhanced=bad))
        assert "company" in s["retrieved_tables"], f"{original!r}: still empty recall"
        rr = deserialize_result(s["retrieval_result_serialized"])
        vsig = [sig for sig in rr.signals if sig.channel == "value" and sig.table == "company"]
        assert vsig, f"{original!r}: no company value signal"
        assert all(sig.query_term == original for sig in vsig)        # provenance = original, never the rewrite


def test_enhanced_only_entity_does_not_produce_value_signal_or_grounding(tmp_path):
    tables, b = _value_db(tmp_path)
    # original asks about one company; the enhanced rewrite fabricates a DIFFERENT indexed company.
    original = "上海云图信息技术 有几个未解决的工单?"
    enhanced = "北京数据科技有限公司有多少个未解决的工单？"     # a real indexed entity, but model-introduced
    s = graph._schema_recall(_state(original, tables, b, enhanced=enhanced))
    rr = deserialize_result(s["retrieval_result_serialized"])
    grounded = [sig.matched_value for sig in rr.signals if sig.channel == "value"]
    assert "北京数据科技有限公司" not in grounded              # the fabricated entity never grounds
    assert "北京数据科技有限公司" not in s.get("value_grounding", "")


def test_graph_module_is_elasticsearch_blind():
    src = inspect.getsource(graph)
    assert "elasticsearch" not in src.lower()
    assert "ElasticsearchValueBackend" not in src
