"""Tests for the SQL safety gate (Phase 1, PR #3)."""
import pytest

from agent.safety import check_sql_safety


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM track",
        "SELECT name, unit_price FROM track WHERE genre_id = 1",
        "WITH t AS (SELECT * FROM track) SELECT count(*) FROM t",
        "SELECT 1 UNION SELECT 2",
    ],
)
def test_read_only_queries_are_safe(sql):
    assert check_sql_safety(sql).safe


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO track(name) VALUES('x')",
        "UPDATE track SET name = 'x'",
        "DELETE FROM track",
        "DROP TABLE track",
        "CREATE TABLE x (a int)",
        "SELECT 1; DROP TABLE track",          # multiple statements
        "PRAGMA foreign_keys = ON",
        "ATTACH DATABASE 'x.db' AS y",
        "SELECT load_extension('evil.so')",
        "",                                    # empty
    ],
)
def test_non_read_only_or_invalid_is_rejected(sql):
    result = check_sql_safety(sql)
    assert not result.safe
    assert result.reason  # always explains why


def test_cte_hiding_dml_is_rejected():
    sql = "WITH x AS (DELETE FROM track RETURNING track_id) SELECT * FROM x"
    assert not check_sql_safety(sql).safe


def test_single_trailing_semicolon_is_tolerated():
    assert check_sql_safety("SELECT * FROM track;").safe
    # but a real second statement is still rejected
    assert not check_sql_safety("SELECT 1; DROP TABLE track").safe


def test_pragma_table_valued_functions_are_rejected():
    # these parse as ordinary SELECTs but expose PRAGMA introspection
    assert not check_sql_safety("SELECT * FROM pragma_table_info('track')").safe
    assert not check_sql_safety("SELECT * FROM pragma_database_list").safe
