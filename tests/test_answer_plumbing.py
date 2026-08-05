"""_to_answer must surface the Python step's analysis (incl. any chart) on AnswerResult
so a caller (the demo) can render it -- previously it was computed then dropped."""
from agent.graph import _to_answer
from agent.usage import UsageCallback


def test_to_answer_surfaces_python_analysis():
    final = {"question": "q", "python_analysis": {"ok": True, "analysis": {"chart": "abc"}}}
    res = _to_answer(final, UsageCallback())
    assert res.python_analysis == {"ok": True, "analysis": {"chart": "abc"}}


def test_to_answer_python_analysis_none_for_sql_only():
    res = _to_answer({"question": "q"}, UsageCallback())
    assert res.python_analysis is None


def test_respond_keeps_chart_base64_out_of_the_answer_text():
    # a chart belongs in the structured python_analysis (for st.image), never dumped as a
    # giant base64 blob into the human answer text; the non-chart analysis still shows.
    from agent.graph import _respond
    from agent.execution import ExecutionResult
    state = {"result": ExecutionResult(True, ["m"], [(1,)]), "tables": [],
             "python_analysis": {"analysis": {"trend": [1, 2, 3], "chart": "BIGBASE64BLOB"}}}
    out = _respond(state)
    assert "BIGBASE64BLOB" not in out["answer"]
    assert "trend" in out["answer"]


def test_respond_omits_empty_analysis_line_for_a_chart_only_step():
    # when the Python step produced ONLY a chart, don't tack an empty "Analysis: {}" onto
    # the answer -- the chart renders separately from python_analysis.
    from agent.graph import _respond
    from agent.execution import ExecutionResult
    state = {"result": ExecutionResult(True, ["m"], [(1,)]), "tables": [],
             "python_analysis": {"analysis": {"chart": "BLOB"}}}
    out = _respond(state)
    assert "Analysis:" not in out["answer"] and "BLOB" not in out["answer"]
