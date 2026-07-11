"""Tests for the get_schema tool and the tool-calling generation path.

The fake model here *has* ``bind_tools``, so it drives the tool-calling branch and
returns a scripted sequence: first a get_schema call, then the SQL that uses the
table it pulled in. No API key, no tokens.
"""
from langchain_core.messages import AIMessage

from agent.db.build_demo_db import build
from agent.db.introspect import introspect
from agent.pipeline import answer_question
from agent.tools import build_get_schema_tool


class FakeToolModel:
    """Has bind_tools (-> tool-calling path). Replays scripted steps:
    ("tool", ["t1", ...]) emits a get_schema call; ("sql", "SELECT ...") answers.

    Plan-aware: under the planner-driven graph the FIRST model call is the planner,
    which invokes with a string prompt (the tool path invokes with a message list). A
    planner prompt yields a single SQL-step plan and does NOT consume a scripted step;
    ``calls`` counts every invoke (planner included), while ``_step`` indexes the
    scripted tool/sql steps -- so each ``calls`` assertion is the old count + 1."""

    def __init__(self, *steps):
        self._steps = list(steps)
        self.calls = 0
        self._step = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        text = messages if isinstance(messages, str) else str(messages)
        if text.rstrip().endswith("JSON:") and "Output a JSON array of steps" in text:
            return AIMessage(content='[{"kind": "sql", "instruction": "answer the question"}]')
        kind, payload = self._steps[min(self._step, len(self._steps) - 1)]
        self._step += 1
        if kind == "tool":
            return AIMessage(content="", tool_calls=[
                {"name": "get_schema", "args": {"table_names": payload}, "id": f"c{self.calls}"}])
        return AIMessage(content=payload)


# --- the tool itself -----------------------------------------------------------

def test_get_schema_tool_renders_known_and_flags_unknown(tmp_path):
    tables = introspect(build(tmp_path / "t.db"))
    requested: list[str] = []
    tool = build_get_schema_tool(tables, requested)

    out = tool.invoke({"table_names": ["track_supplier"]})
    assert "track_supplier" in out and "TABLE" in out
    assert requested == ["track_supplier"]            # recorded for the trace

    bad = tool.invoke({"table_names": ["no_such_table"]})
    assert "Unknown table" in bad


# --- the tool-calling generation path -----------------------------------------

def test_model_calls_get_schema_then_writes_sql(tmp_path):
    db = build(tmp_path / "t.db")
    tables = introspect(db)
    # the retriever may not surface track_supplier for this wording; the model asks
    # for it via the tool, then writes SQL against it.
    model = FakeToolModel(
        ("tool", ["track_supplier"]),
        ("sql", "SELECT COUNT(*) FROM track_supplier"),
    )
    res = answer_question(db, "how many track-supplier links exist?", model=model, tables=tables)

    assert res.execution.ok
    assert model.calls == 3                            # planner, one tool round, then the SQL
    gen = next(t for t in res.trace if t["node"] == "generate_sql")
    assert "track_supplier" in gen.get("requested_tables", [])


def test_tool_path_still_self_corrects(tmp_path):
    # the tool path must keep the repair loop: bad SQL -> validate flags -> retry.
    db = build(tmp_path / "t.db")
    tables = introspect(db)
    model = FakeToolModel(
        ("sql", "SELECT no_such_col FROM track"),      # attempt 1: execution error
        ("sql", "SELECT COUNT(*) FROM track"),         # attempt 2 (repair): works
    )
    res = answer_question(db, "how many tracks are there?", model=model, tables=tables)
    assert res.execution.ok and res.execution.rows == [(306,)]
    assert model.calls == 3                             # planner + draft + one repair


def test_tool_call_then_self_correct(tmp_path):
    # the most production-like path: ask for a table via the tool, write SQL that
    # fails, then repair fixes it -- tool use AND self-correction in one run.
    db = build(tmp_path / "t.db")
    tables = introspect(db)
    model = FakeToolModel(
        ("tool", ["track_supplier"]),                        # attempt 1: pull the table
        ("sql", "SELECT no_such_col FROM track_supplier"),   # attempt 1: bad SQL -> exec error
        ("sql", "SELECT COUNT(*) FROM track_supplier"),      # attempt 2 (repair): works
    )
    res = answer_question(db, "how many track-supplier links exist?", model=model, tables=tables)
    assert res.execution.ok
    assert model.calls == 4                              # planner + tool round + bad SQL + repair
    gen = [t for t in res.trace if t["node"] == "generate_sql"]
    assert len(gen) == 2 and "track_supplier" in gen[0].get("requested_tables", [])


def test_tool_path_repair_keeps_the_planner_step_instruction(tmp_path):
    # teeth for the PRODUCTION (bind_tools) path: the repair round must carry the
    # planner's SQL-step instruction, not fall back to the raw question.
    from agent.graph import _generate_with_tools
    tables = introspect(build(tmp_path / "t.db"))

    class RecordingToolModel:
        def bind_tools(self, tools):
            return self
        def bind(self, **kw):
            return self
        def invoke(self, messages):
            self.messages = messages
            return AIMessage(content="SELECT 1", tool_calls=[])

    m = RecordingToolModel()
    state = {"tables": tables, "schema": "S", "question": "plot the mrr trend",
             "sql": "SELECT bad", "error": "no such column: bad",
             "plan": [{"kind": "sql", "instruction": "pull monthly mrr rows"}],
             "step_index": 0}
    _generate_with_tools(state, m, attempts=1, requested=[])
    assert any("pull monthly mrr rows" in getattr(msg, "content", "") for msg in m.messages)
