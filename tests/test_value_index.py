"""Value-index ingestion: indexes only searchable values, never PII; idempotent; prunes stale;
fingerprint-skips an unchanged rebuild. FakeValueBackend + a tiny sqlite fixture (no ES)."""
import sqlite3

from agent.db.introspect import Column, Table
from agent.retrieval.value_backend import FakeValueBackend
from agent.retrieval.value_index import build_value_index, document_id, index_name


def _col(name, policy="public"):
    return Column(name=name, type="TEXT", pk=False, notnull=False, policy=policy)


def _tables():
    return [Table(name="product", columns=[_col("name", "searchable"), _col("price")]),
            Table(name="account", columns=[_col("company_name", "searchable"),
                                           _col("contact_email", "pii")])]


def _db(tmp_path, *, gadget=True):
    p = tmp_path / "v.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        "CREATE TABLE product(name TEXT, price REAL);"
        "CREATE TABLE account(company_name TEXT, contact_email TEXT);"
        "INSERT INTO product VALUES ('Widget', 9.9);"
        + ("INSERT INTO product VALUES ('Gadget', 5.0);" if gadget else "")
        + "INSERT INTO account VALUES ('Acme Corp', 'ceo@acme.com');")
    conn.commit()
    conn.close()
    return str(p)


ALLOW = [("product", "name"), ("account", "company_name")]


def test_indexes_only_searchable_values(tmp_path):
    b = FakeValueBackend()
    r = build_value_index(_tables(), _db(tmp_path), b)
    assert r.doc_count == 3 and not r.skipped        # Widget, Gadget, Acme Corp — NOT price, NOT email


def test_pii_value_never_indexed_or_returned(tmp_path):
    b = FakeValueBackend()
    build_value_index(_tables(), _db(tmp_path), b)
    # even searching the pii column explicitly + the pii value: it was never indexed
    hits = b.search("ceo@acme.com Acme Corp", allowed=ALLOW + [("account", "contact_email")])
    assert all(h.column != "contact_email" for h in hits)
    assert all(h.value != "ceo@acme.com" for h in hits)


def test_idempotent_rebuild_same_fingerprint_and_docs(tmp_path):
    db = _db(tmp_path)
    b = FakeValueBackend()
    r1 = build_value_index(_tables(), db, b)
    r2 = build_value_index(_tables(), db, b)
    assert r1.fingerprint == r2.fingerprint
    hits = b.search("Widget Gadget Acme Corp", allowed=ALLOW)
    assert len(hits) == 3


def test_fingerprint_skip_writes_nothing(tmp_path):
    class SpyBackend(FakeValueBackend):
        def __init__(self):
            super().__init__()
            self.writes = 0

        def upsert(self, docs):
            self.writes += 1
            super().upsert(docs)

    db = _db(tmp_path)
    b = SpyBackend()
    r1 = build_value_index(_tables(), db, b)
    r2 = build_value_index(_tables(), db, b, known_fingerprint=r1.fingerprint)
    assert r2.skipped and b.writes == 1              # second build is a no-op


def test_prune_removes_stale_docs(tmp_path):
    db = _db(tmp_path, gadget=True)
    b = FakeValueBackend()
    build_value_index(_tables(), db, b)
    conn = sqlite3.connect(db)                       # remove Gadget from the SOURCE
    conn.execute("DELETE FROM product WHERE name='Gadget'")
    conn.commit()
    conn.close()
    build_value_index(_tables(), db, b)
    hits = b.search("Widget Gadget", allowed=[("product", "name")])
    assert {h.value for h in hits} == {"Widget"}     # Gadget pruned


def test_index_name_and_document_id_stability():
    assert index_name(_tables()).startswith("cadence-values-")
    assert document_id("product", "name", "Widget") == document_id("product", "name", "Widget")
    assert document_id("product", "name", "Widget") != document_id("product", "name", "Gadget")


def test_index_name_is_a_valid_elasticsearch_index_name():
    # ES rejects spaces / , / ( ) / uppercase etc.; the name must be a lowercase hex-suffixed slug.
    import re
    name = index_name(_tables())
    assert name == name.lower()
    assert re.fullmatch(r"cadence-values-[0-9a-f]+", name), name
