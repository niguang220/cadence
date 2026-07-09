"""Tests for read-only SQL execution (Phase 1, PR #3)."""
from agent.db.build_demo_db import build
from agent.execution import run_query


def test_select_returns_columns_and_rows(tmp_path):
    db = build(tmp_path / "t.db")
    res = run_query(db, "SELECT name, unit_price FROM track LIMIT 3")
    assert res.ok
    assert res.columns == ["name", "unit_price"]
    assert len(res.rows) == 3


def test_query_error_is_returned_not_raised(tmp_path):
    db = build(tmp_path / "t.db")
    res = run_query(db, "SELECT no_such_column FROM track")
    assert not res.ok and res.error


def test_write_blocked_by_read_only_connection(tmp_path):
    # defence in depth: a write that bypassed the safety gate still fails read-only
    db = build(tmp_path / "t.db")
    res = run_query(db, "DELETE FROM track")
    assert not res.ok
    assert "readonly" in res.error.lower() or "read-only" in res.error.lower()


def test_max_rows_truncates(tmp_path):
    db = build(tmp_path / "t.db")
    res = run_query(db, "SELECT track_id FROM track", max_rows=5)
    assert res.ok
    assert len(res.rows) == 5
    assert res.truncated


def test_attach_rejected_by_default_no_file_side_effect(tmp_path):
    db = build(tmp_path / "t.db")
    target = tmp_path / "attached.db"
    res = run_query(db, f"ATTACH DATABASE '{target}' AS evil")
    assert not res.ok
    assert not target.exists()  # the safety gate stops it before any side effect


def test_pragma_rejected_by_default(tmp_path):
    db = build(tmp_path / "t.db")
    assert not run_query(db, "PRAGMA foreign_keys = OFF").ok


def test_authorizer_blocks_attach_even_when_assumed_safe(tmp_path):
    # second line of defence: even bypassing the parser, the authorizer denies ATTACH
    db = build(tmp_path / "t.db")
    target = tmp_path / "attached2.db"
    res = run_query(db, f"ATTACH DATABASE '{target}' AS evil", assume_safe=True)
    assert not res.ok
    assert not target.exists()


def test_timeout_interrupts_long_running_query(tmp_path):
    db = build(tmp_path / "t.db")
    # COUNT over a big cross join forces full computation (~306^4 rows) -> the
    # progress-handler timeout must abort it (a lazy fetch wouldn't trigger this)
    sql = "SELECT COUNT(*) FROM track a, track b, track c, track d"
    res = run_query(db, sql, timeout_seconds=0.05)
    assert not res.ok
    assert "interrupt" in res.error.lower()


def test_invalid_limits_are_rejected(tmp_path):
    db = build(tmp_path / "t.db")
    assert not run_query(db, "SELECT 1", max_rows=0).ok
    assert not run_query(db, "SELECT 1", timeout_seconds=0).ok


def test_pragma_table_function_rejected_by_default(tmp_path):
    db = build(tmp_path / "t.db")
    assert not run_query(db, "SELECT * FROM pragma_table_info('track')").ok


def test_authorizer_denies_transactions_and_savepoints(tmp_path):
    # not data writes, but a read-only SELECT executor shouldn't run them either
    db = build(tmp_path / "t.db")
    for sql in ("BEGIN", "SAVEPOINT sp1", "ANALYZE"):
        res = run_query(db, sql, assume_safe=True)
        assert not res.ok, sql
