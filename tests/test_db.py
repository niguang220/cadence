"""Tests for the demo DB build + schema introspection.

Contract tests on the *shape* the rest of the project relies on — pinned
edge-case fixtures, FK integrity, the invoice-total invariant, schema-linking
material — not just "does it have rows".
"""
import sqlite3

import pytest

from agent.db.build_demo_db import build
from agent.db.introspect import introspect, render_schema


def _db(tmp_path):
    return build(tmp_path / "t.db")


def _conn(db):
    c = sqlite3.connect(db)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def test_14_tables_all_described(tmp_path):
    tables = introspect(_db(tmp_path))
    assert len(tables) == 14
    assert all(t.description for t in tables)


def test_foreign_key_integrity_is_clean(tmp_path):
    conn = _conn(_db(tmp_path))
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_invoice_total_matches_sum_of_lines(tmp_path):
    conn = _conn(_db(tmp_path))
    mismatches = conn.execute(
        """
        SELECT i.invoice_id FROM invoice i
        JOIN (SELECT invoice_id, ROUND(SUM(unit_price * quantity), 2) s
              FROM invoice_line GROUP BY invoice_id) il
          ON il.invoice_id = i.invoice_id
        WHERE ROUND(i.total, 2) <> il.s
        """
    ).fetchall()
    conn.close()
    assert mismatches == []


def test_named_edge_fixtures_present(tmp_path):
    q = _conn(_db(tmp_path)).execute
    # no-invoice customer (LEFT JOIN customer->invoice)
    cid = q("SELECT customer_id FROM customer WHERE first_name='Noinvoice'").fetchone()[0]
    assert q("SELECT count(*) FROM invoice WHERE customer_id=?", (cid,)).fetchone()[0] == 0
    # no-track album (LEFT JOIN album->track)
    aid = q("SELECT album_id FROM album WHERE title LIKE 'Empty Album%'").fetchone()[0]
    assert q("SELECT count(*) FROM track WHERE album_id=?", (aid,)).fetchone()[0] == 0
    # zero-sales track (LEFT JOIN track->invoice_line)
    tid = q("SELECT track_id FROM track WHERE name='Zero Sales Track'").fetchone()[0]
    assert q("SELECT count(*) FROM invoice_line WHERE track_id=?", (tid,)).fetchone()[0] == 0
    # null-album single
    assert q("SELECT album_id FROM track WHERE name='Null Album Single'").fetchone()[0] is None
    # distinct-trap playlist: many tracks, one artist -> 'how many artists' needs DISTINCT
    pid = q("SELECT playlist_id FROM playlist WHERE name='Distinct Trap Playlist'").fetchone()[0]
    n_tracks, n_artists = q(
        """SELECT COUNT(*), COUNT(DISTINCT al.artist_id)
           FROM playlist_track pt JOIN track t ON t.track_id = pt.track_id
           JOIN album al ON al.album_id = t.album_id
           WHERE pt.playlist_id=?""",
        (pid,),
    ).fetchone()
    assert n_tracks > 1 and n_artists == 1


def test_review_comments_are_mixed_null_and_nonnull(tmp_path):
    conn = _conn(_db(tmp_path))
    total = conn.execute("SELECT COUNT(*) FROM review").fetchone()[0]
    nonnull = conn.execute("SELECT COUNT(comment) FROM review").fetchone()[0]
    conn.close()
    assert 0 < nonnull < total  # COUNT(*) vs COUNT(comment) genuinely differ


def test_country_columns_distinct_and_not_confused(tmp_path):
    by_name = {t.name: t for t in introspect(_db(tmp_path))}
    cust = next(c for c in by_name["customer"].columns if c.name == "country")
    inv = next(c for c in by_name["invoice"].columns if c.name == "billing_country")
    assert "own country" in cust.description.lower()
    assert "billing" in inv.description.lower()


def test_low_cardinality_columns_carry_sample_values(tmp_path):
    by_name = {t.name: t for t in introspect(_db(tmp_path))}
    country = next(c for c in by_name["customer"].columns if c.name == "country")
    email = next(c for c in by_name["customer"].columns if c.name == "email")
    first_name = next(c for c in by_name["employee"].columns if c.name == "first_name")
    assert country.sample_values        # low cardinality -> sampled for schema linking
    assert email.sample_values == ()    # high cardinality -> not sampled
    assert first_name.policy == "pii" and first_name.sample_values == ()


def test_pii_columns_are_hidden_from_rendered_schema(tmp_path):
    tables = introspect(_db(tmp_path))
    rendered = render_schema(tables, only=["customer", "employee"])
    assert "email" not in rendered
    assert "first_name" not in rendered
    assert "last_name" not in rendered
    assert "customer_id" in rendered
    assert "country" in rendered


def test_render_fk_neighbors_includes_referenced_tables(tmp_path):
    tables = introspect(_db(tmp_path))
    rendered = render_schema(tables, only=["invoice_line"], include_fk_neighbors=True)
    # trailing " (" disambiguates 'invoice' from 'invoice_line'
    assert "TABLE invoice_line (" in rendered
    assert "TABLE invoice (" in rendered   # FK neighbor
    assert "TABLE track (" in rendered     # FK neighbor
    assert "TABLE invoice (" not in render_schema(tables, only=["invoice_line"])


def test_fk_neighbors_are_bidirectional(tmp_path):
    # picking a parent table should also pull its child tables (incoming FKs)
    tables = introspect(_db(tmp_path))
    rendered = render_schema(tables, only=["invoice"], include_fk_neighbors=True)
    assert "TABLE customer (" in rendered      # outgoing: invoice -> customer
    assert "TABLE invoice_line (" in rendered  # incoming: invoice_line -> invoice


def test_low_cardinality_samples_not_truncated(tmp_path):
    by_name = {t.name: t for t in introspect(_db(tmp_path))}
    country = next(c for c in by_name["customer"].columns if c.name == "country")
    # every distinct value of a low-cardinality column is shown, incl. common markets
    assert "USA" in country.sample_values and "UK" in country.sample_values


def test_country_mismatch_fixture(tmp_path):
    q = _conn(_db(tmp_path)).execute
    cid = q("SELECT customer_id FROM customer WHERE first_name='CountryMismatch'").fetchone()[0]
    home = q("SELECT country FROM customer WHERE customer_id=?", (cid,)).fetchone()[0]
    billed = q("SELECT billing_country FROM invoice WHERE customer_id=?", (cid,)).fetchone()[0]
    assert home == "Singapore" and billed == "China"  # same-name-different-meaning trap


def test_supplier_fanout_track_fixture(tmp_path):
    q = _conn(_db(tmp_path)).execute
    tid = q("SELECT track_id FROM track WHERE name='Supplier Fanout Track'").fetchone()[0]
    n_suppliers = q("SELECT count(*) FROM track_supplier WHERE track_id=?", (tid,)).fetchone()[0]
    n_sales = q("SELECT count(*) FROM invoice_line WHERE track_id=?", (tid,)).fetchone()[0]
    assert n_suppliers == 2 and n_sales == 1  # joining suppliers would inflate the sale count


def test_build_refuses_overwrite_when_disabled(tmp_path):
    p = build(tmp_path / "x.db")
    with pytest.raises(FileExistsError):
        build(p, overwrite=False)


def test_build_is_deterministic(tmp_path):
    a, b = build(tmp_path / "a.db"), build(tmp_path / "b.db")
    ca, cb = sqlite3.connect(a), sqlite3.connect(b)
    for query in (
        "SELECT COUNT(*) FROM track",
        "SELECT ROUND(SUM(total), 2) FROM invoice",
        "SELECT COUNT(*) FROM invoice_line",
    ):
        assert ca.execute(query).fetchone() == cb.execute(query).fetchone()
    ca.close()
    cb.close()
