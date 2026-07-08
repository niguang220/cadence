"""Tests for the lexical schema retriever (Phase 1, PR #2)."""
from agent.db.build_demo_db import build
from agent.db.introspect import introspect
from agent.lexical_retriever import retrieve_tables


def _tables(tmp_path):
    return introspect(build(tmp_path / "t.db"))


def test_question_about_tracks_and_genre(tmp_path):
    got = retrieve_tables("how many tracks are in each genre", _tables(tmp_path))
    assert "track" in got and "genre" in got


def test_sample_value_links_to_its_column(tmp_path):
    # "Singapore" is a value of customer.country -> customer is retrieved
    got = retrieve_tables("which customers are from Singapore", _tables(tmp_path))
    assert "customer" in got


def test_sales_by_country_links_invoice(tmp_path):
    # total + billing_country are invoice columns -> invoice ranks first
    got = retrieve_tables("total sales by billing country", _tables(tmp_path))
    assert got[0] == "invoice"


def test_plurals_match_singular_table(tmp_path):
    assert "album" in retrieve_tables("list the albums", _tables(tmp_path))


def test_off_topic_question_returns_nothing(tmp_path):
    assert retrieve_tables("what is the weather today", _tables(tmp_path)) == []


def test_k_limits_number_of_tables(tmp_path):
    got = retrieve_tables("tracks and invoices and customers and genres", _tables(tmp_path), k=2)
    assert len(got) <= 2


# --- adversarial: ranking pathologies, not just happy-path inclusion ---


def test_track_table_not_buried_by_fk_columns(tmp_path):
    # track_id appears in many tables; the actual 'track' table must still rank
    tables = _tables(tmp_path)
    assert "track" in retrieve_tables("tracks by artist", tables)
    assert "track" in retrieve_tables("which suppliers provide each track", tables)


def test_revenue_synonym_links_invoice(tmp_path):
    assert retrieve_tables("revenue by country", _tables(tmp_path))[0] == "invoice"


def test_song_synonym_links_track(tmp_path):
    assert "track" in retrieve_tables("songs by genre", _tables(tmp_path))


def test_unrelated_domain_returns_nothing(tmp_path):
    assert retrieve_tables("gas pipeline pressure", _tables(tmp_path)) == []
