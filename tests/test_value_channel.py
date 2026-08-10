"""ValueChannel: turn value-backend hits into typed value RetrievalSignals, restricted to the
searchable allowlist, and let a backend error propagate (the pipeline turns it into an event)."""
import pytest

from agent.db.introspect import Column, Table
from agent.retrieval.aggregate import aggregate
from agent.retrieval.channels import ValueChannel
from agent.retrieval.value_backend import FakeValueBackend, ValueBackendError, ValueDoc


def _col(name, policy="public"):
    return Column(name=name, type="TEXT", pk=False, notnull=False, policy=policy)


def _tables():
    return [Table(name="product", columns=[_col("name", "searchable"), _col("price")]),
            Table(name="account", columns=[_col("company_name", "searchable")])]


def _backend():
    b = FakeValueBackend()
    b.ensure_index()
    b.upsert([ValueDoc("d1", "product", "name", "Widget"),
              ValueDoc("d2", "account", "company_name", "Acme Corp")])
    return b


def test_value_channel_emits_typed_value_signals():
    sigs = ValueChannel(_backend()).signals("do we sell Widget", _tables())
    s = next(s for s in sigs if s.table == "product")
    assert s.channel == "value" and s.target_type == "value" and s.column == "name"
    assert s.match_type == "exact_keyword" and s.document_id == "d1"
    assert s.query_term == "do we sell Widget" and s.raw_score == 1.0


def test_value_signals_aggregate_one_best_per_table():
    sigs = ValueChannel(_backend()).signals("Widget at Acme Corp", _tables())
    ctrs = aggregate("value", sigs)
    assert {c.table for c in ctrs} == {"product", "account"}
    assert all(len(c.signals) == 1 for c in ctrs)       # one best value evidence per table


def test_value_channel_propagates_backend_error():
    class Boom:
        def search(self, *a, **k):
            raise ValueBackendError("es down")

    with pytest.raises(ValueBackendError):
        ValueChannel(Boom()).signals("q", _tables())
