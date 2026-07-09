"""Column-governance tests for PII blocking."""

from agent.db.build_demo_db import build
from agent.db.introspect import introspect
from agent.execution import run_query
from agent.governance import check_result_governance, check_sql_governance


def _tables(tmp_path):
    return introspect(build(tmp_path / "t.db"))


def test_governance_allows_public_customer_aggregates(tmp_path):
    tables = _tables(tmp_path)
    result = check_sql_governance(
        "SELECT country, COUNT(*) FROM customer GROUP BY country",
        tables,
    )
    assert result.ok


def test_governance_blocks_direct_pii_column(tmp_path):
    tables = _tables(tmp_path)
    result = check_sql_governance("SELECT email FROM customer", tables)
    assert not result.ok
    assert result.columns == ["customer.email"]


def test_governance_blocks_qualified_pii_column(tmp_path):
    tables = _tables(tmp_path)
    result = check_sql_governance("SELECT c.email FROM customer c", tables)
    assert not result.ok
    assert result.columns == ["customer.email"]


def test_governance_blocks_select_star_on_table_with_pii(tmp_path):
    tables = _tables(tmp_path)
    result = check_sql_governance("SELECT * FROM customer", tables)
    assert not result.ok
    assert "customer.*" in result.columns


def test_governance_allows_qualified_star_for_public_table_in_join(tmp_path):
    tables = _tables(tmp_path)
    result = check_sql_governance(
        "SELECT i.* FROM invoice i JOIN customer c ON c.customer_id = i.customer_id",
        tables,
    )
    assert result.ok


def test_governance_allows_count_star_but_blocks_pii_aggregate(tmp_path):
    tables = _tables(tmp_path)
    assert check_sql_governance("SELECT COUNT(*) FROM customer", tables).ok

    result = check_sql_governance("SELECT COUNT(email) FROM customer", tables)
    assert not result.ok
    assert result.columns == ["customer.email"]


def test_governance_blocks_pii_filter_even_when_not_selected(tmp_path):
    tables = _tables(tmp_path)
    result = check_sql_governance(
        "SELECT customer_id FROM customer WHERE email LIKE '%@example.com'",
        tables,
    )
    assert not result.ok
    assert result.columns == ["customer.email"]


def test_governance_blocks_alias_shadowing_pii_reference(tmp_path):
    tables = _tables(tmp_path)
    result = check_sql_governance(
        """
        SELECT COUNT(*)
        FROM customer c
        WHERE c.email IS NOT NULL
          AND EXISTS (SELECT 1 FROM invoice c)
        """,
        tables,
    )
    assert not result.ok
    assert result.columns == ["customer.email"]


def test_run_query_applies_governance_when_tables_are_supplied(tmp_path):
    db = build(tmp_path / "t.db")
    tables = introspect(db)
    result = run_query(db, "SELECT email FROM customer LIMIT 1", tables=tables)
    assert not result.ok
    assert "governance violation" in result.error


def test_run_query_assume_safe_does_not_skip_governance(tmp_path):
    db = build(tmp_path / "t.db")
    tables = introspect(db)
    result = run_query(db, "SELECT email FROM customer LIMIT 1", tables=tables, assume_safe=True)
    assert not result.ok
    assert "governance violation" in result.error


def test_answer_layer_fallback_blocks_pii_column_names(tmp_path):
    tables = _tables(tmp_path)
    result = check_result_governance(["email"], tables)
    assert not result.ok
    assert result.columns == ["email"]
