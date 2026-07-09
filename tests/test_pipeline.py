"""Tests for the baseline NL->SQL pipeline (Phase 1, PR #4).

The LLM is faked (a model that returns a fixed string), so these run with no API
key and no tokens; the retrieval, safety and execution around it are real.
"""
from agent.execution import ExecutionResult
from agent.generation import _extract_sql, _format_answer
from agent.pipeline import answer_question


class FakeModel:
    """Stands in for the chat model: returns a fixed string, records the prompt."""

    def __init__(self, reply: str):
        self._reply = reply
        self.last_prompt = None
        self.calls = 0

    def invoke(self, prompt):
        self.last_prompt = prompt
        self.calls += 1
        return type("R", (), {"content": self._reply})()


def test_extract_sql_strips_fences_and_surrounding_prose():
    assert _extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert _extract_sql("Here is your query:\n```sql\nSELECT 1\n```\nHope it helps!") == "SELECT 1"
    assert _extract_sql("SELECT 1") == "SELECT 1"


def test_pipeline_runs_generated_sql_and_answers(tmp_path):
    from agent.db.build_demo_db import build
    db = build(tmp_path / "t.db")
    res = answer_question(db, "how many tracks are there", model=FakeModel("SELECT COUNT(*) FROM track"))
    assert res.sql == "SELECT COUNT(*) FROM track"
    assert res.execution.ok
    assert res.execution.rows == [(306,)]   # 300 seeded + 6 pinned fixtures
    assert "306" in res.answer


def test_pipeline_passes_only_relevant_schema_to_model(tmp_path):
    from agent.db.build_demo_db import build
    db = build(tmp_path / "t.db")
    model = FakeModel("SELECT 1")
    answer_question(db, "list tracks by genre", model=model)
    assert "TABLE track (" in model.last_prompt
    assert "TABLE genre (" in model.last_prompt
    # an unrelated table (not retrieved, not an FK neighbour) must NOT leak in
    # -> proves we didn't fall back to dumping the whole schema
    assert "TABLE employee (" not in model.last_prompt


def test_pipeline_does_not_put_pii_columns_in_prompt(tmp_path):
    from agent.db.build_demo_db import build
    db = build(tmp_path / "t.db")
    model = FakeModel("SELECT COUNT(*) FROM customer")
    answer_question(db, "how many customers are there", model=model)
    assert "TABLE customer (" in model.last_prompt
    assert "email" not in model.last_prompt
    assert "first_name" not in model.last_prompt
    assert "last_name" not in model.last_prompt


def test_pipeline_refuses_when_no_tables_match(tmp_path):
    from agent.db.build_demo_db import build
    db = build(tmp_path / "t.db")
    res = answer_question(db, "what is the meaning of life", model=FakeModel("SELECT 1"))
    assert res.retrieved_tables == [] and res.sql == ""
    assert "couldn't identify" in res.answer.lower()


def test_pipeline_handles_model_declining_with_cannot_answer(tmp_path):
    from agent.db.build_demo_db import build
    db = build(tmp_path / "t.db")
    res = answer_question(db, "how many tracks", model=FakeModel("CANNOT_ANSWER"))
    assert not res.execution.ok
    assert "couldn't write" in res.answer.lower()
    # the model often adds an explanation after the sentinel -> still a decline
    res2 = answer_question(db, "how many tracks", model=FakeModel("CANNOT_ANSWER - no weather table"))
    assert not res2.execution.ok and "couldn't write" in res2.answer.lower()


def test_pipeline_rejects_unsafe_generated_sql(tmp_path):
    from agent.db.build_demo_db import build
    db = build(tmp_path / "t.db")
    res = answer_question(db, "drop it", model=FakeModel("DROP TABLE track"))
    assert not res.execution.ok
    assert "couldn't answer" in res.answer.lower()


def test_pipeline_rejects_generated_pii_sql_without_repairing(tmp_path):
    from agent.db.build_demo_db import build
    db = build(tmp_path / "t.db")
    model = FakeModel("SELECT email FROM customer LIMIT 5")
    res = answer_question(db, "show customer emails", model=model)
    assert model.calls == 1
    assert not res.execution.ok
    assert "governance violation" in res.execution.error
    assert "couldn't answer" in res.answer.lower()
    execute = next(t for t in res.trace if t["node"] == "execute")
    assert execute["governance"] == "blocked"
    validate = next(t for t in res.trace if t["node"] == "validate")
    assert validate["kind"] == "governance_block"


def test_pipeline_result_layer_governance_block_is_traced(tmp_path):
    from agent.db.build_demo_db import build
    db = build(tmp_path / "t.db")
    model = FakeModel("SELECT 'redacted' AS email FROM customer LIMIT 1")
    res = answer_question(db, "show customer emails", model=model)
    assert model.calls == 1
    assert not res.execution.ok
    assert "governance violation" in res.execution.error
    execute = next(t for t in res.trace if t["node"] == "execute")
    assert execute["governance"] == "blocked"
    validate = next(t for t in res.trace if t["node"] == "validate")
    assert validate["kind"] == "governance_block"


def test_pipeline_surfaces_sql_errors(tmp_path):
    from agent.db.build_demo_db import build
    db = build(tmp_path / "t.db")
    res = answer_question(db, "bad query", model=FakeModel("SELECT no_such_col FROM track"))
    assert not res.execution.ok and res.execution.error


def test_format_answer_null_and_singular_row():
    assert _format_answer(ExecutionResult(True, columns=["total"], rows=[(None,)])) == "total: (null)"
    out = _format_answer(ExecutionResult(True, columns=["a", "b"], rows=[(1, 2)]))
    assert "(1 row)" in out and "rows" not in out


def test_format_answer_count_is_consistent_with_preview():
    rows = [(i,) for i in range(6)]
    out = _format_answer(ExecutionResult(True, columns=["x"], rows=rows))
    assert "showing 5 of 6" in out
