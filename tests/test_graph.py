"""Tests for the Phase 2 reliability loop (validate_result + self-correction).

Unit tests pin each validate rule; integration tests drive the whole graph with a
fake model that returns a *sequence* of replies, so we can prove it self-corrects
(bad SQL -> repair -> good SQL) and gives up (bounded) when it can't.
No API key, no tokens.
"""
from agent.db.build_demo_db import build
from agent.execution import ExecutionResult
from agent.graph import MAX_ATTEMPTS
from agent.pipeline import answer_question
from agent.validate import (has_aggregate, has_group_by, has_join, has_order_by,
                            validate_result, wants_aggregation, wants_ranking)


class SequenceModel:
    """Returns canned replies in order (last one repeats); records every prompt."""

    def __init__(self, *replies: str):
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.calls = 0

    def invoke(self, prompt):
        self.prompts.append(prompt)
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return type("R", (), {"content": reply})()


# --- validate predicates -------------------------------------------------------

def test_structural_predicates():
    assert has_join("SELECT * FROM a JOIN b ON a.id = b.id")
    assert not has_join("SELECT * FROM a")
    assert has_order_by("SELECT x FROM a ORDER BY x")
    assert has_group_by("SELECT x, COUNT(*) FROM a GROUP BY x")
    assert has_aggregate("SELECT COUNT(*) FROM a")
    assert not has_aggregate("SELECT x FROM a")
    assert wants_ranking("what are the 5 longest tracks?")
    assert wants_aggregation("how many tracks are there?")
    assert not wants_ranking("list all genres")


# --- validate rules ------------------------------------------------------------

def test_validate_flags_execution_error():
    v = validate_result("q", "SELECT 1", ExecutionResult(False, error="no such column"))
    assert not v.ok and v.repair_kind == "exec_error"


def test_validate_empty_join_suspicious_but_empty_single_table_ok():
    empty = ExecutionResult(True, columns=["x"], rows=[])
    assert not validate_result("q", "SELECT a.x FROM a JOIN b ON a.id = b.id", empty).ok
    # an empty single-table filter is usually the correct answer -> not flagged
    assert validate_result("q", "SELECT x FROM a WHERE x > 999", empty).ok


def test_validate_ranking_without_order_by():
    rows = ExecutionResult(True, columns=["name"], rows=[("a",), ("b",)])
    assert not validate_result("the 5 longest tracks", "SELECT name FROM track LIMIT 5", rows).ok
    assert validate_result("the 5 longest tracks",
                           "SELECT name FROM track ORDER BY milliseconds DESC LIMIT 5", rows).ok


def test_validate_aggregation_without_aggregate():
    many = ExecutionResult(True, columns=["name"], rows=[("a",), ("b",), ("c",)])
    assert not validate_result("how many tracks", "SELECT name FROM track", many).ok
    one = ExecutionResult(True, columns=["c"], rows=[(306,)])
    assert validate_result("how many tracks", "SELECT COUNT(*) FROM track", one).ok


# --- self-correction loop (end to end through the graph) -----------------------

def test_loop_self_corrects_on_execution_error(tmp_path):
    db = build(tmp_path / "t.db")
    model = SequenceModel(
        "SELECT no_such_col FROM track",     # attempt 1: fails to execute
        "SELECT COUNT(*) FROM track",        # attempt 2 (repair): works
    )
    res = answer_question(db, "how many tracks are there?", model=model)
    assert model.calls == 2                  # repaired after one failure
    assert res.execution.ok and res.execution.rows == [(306,)]
    assert "306" in res.answer
    # the repair attempt's prompt carried the failing SQL + the DB error
    assert "no_such_col" in model.prompts[1]


def test_loop_repairs_a_validate_flag_not_just_errors(tmp_path):
    db = build(tmp_path / "t.db")
    model = SequenceModel(
        "SELECT name, milliseconds FROM track LIMIT 5",                       # runs, but no ORDER BY
        "SELECT name, milliseconds FROM track ORDER BY milliseconds DESC LIMIT 5",
    )
    res = answer_question(db, "what are the 5 longest tracks, longest first?", model=model)
    assert model.calls == 2                  # validate flagged the missing ORDER BY -> repaired
    assert "ORDER BY" in res.sql and res.execution.ok


def test_loop_gives_up_after_max_attempts(tmp_path):
    db = build(tmp_path / "t.db")
    model = SequenceModel("SELECT no_such_col FROM track")   # always broken
    res = answer_question(db, "how many tracks are there?", model=model)
    assert model.calls == MAX_ATTEMPTS       # bounded
    assert not res.execution.ok
    assert "couldn't answer" in res.answer.lower()
    generates = [t for t in res.trace if t["node"] == "generate_sql"]
    assert len(generates) == MAX_ATTEMPTS
