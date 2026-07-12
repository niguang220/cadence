"""Tests for ambiguity clarification (PR #8).

Unit tests pin the heuristic; an end-to-end test proves the agent asks (and does
NOT call the model) when the question is ambiguous, and still answers a clear one.
"""
from agent.clarify import (
    build_clarification_options,
    detect_ambiguity,
    format_clarification_intent,
    normalize_clarification_response,
    parse_clarification_intent,
)
from agent.db.build_demo_db import build
from agent.db.introspect import introspect
from agent.pipeline import answer_question, resume_question_session, start_question_session
from agent.semantic_layer import load_metrics


class FakeModel:
    def __init__(self, reply: str):
        self._reply = reply
        self.last_prompt = None
        self.saw_consistency = False

    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        # semantic_consistency is the LAST model call on a validated SQL step; a pure
        # SIDE-CHANNEL (does NOT touch last_prompt) so it can't overwrite the generation
        # prompt those tests assert on. Returns a passthrough ok verdict.
        if "semantic-consistency judge" in text:
            self.saw_consistency = True
            return type("R", (), {"content": '{"ok": true}'})()
        self.last_prompt = prompt
        # query_enhance runs before the planner on the proceed path; a passthrough
        # (empty enhanced_question -> falls back to the original) keeps generation
        # byte-identical. On the ask/refuse paths enhance never runs (last_prompt stays
        # the pre-enhance value), so the "never called the model" assertions still hold.
        if "governed metric terms" in text:
            return type("R", (), {"content": '{"enhanced_question": ""}'})()
        # Plan-aware: the planner is the first model call under the step-loop graph;
        # a planner prompt yields a single SQL step so the configured SQL still drives
        # generation (last_prompt is thus the generation prompt, not the plan).
        if text.rstrip().endswith("JSON:") and "Output a JSON array of steps" in text:
            return type("R", (), {"content": '[{"kind": "sql", "instruction": "answer the question"}]'})()
        return type("R", (), {"content": self._reply})()


# --- the heuristic -------------------------------------------------------------

def test_flags_unqualified_superlatives():
    assert detect_ambiguity("who are the best customers?")
    assert detect_ambiguity("what are the most popular tracks?")
    assert detect_ambiguity("show me the most valuable customers")


def test_ignores_questions_that_name_a_metric():
    # a metric word ("selling", "spending", "rated", "sales") makes it specific
    assert detect_ambiguity("which tracks are best-selling?") is None
    assert detect_ambiguity("who are the top 3 customers by total spending?") is None
    assert detect_ambiguity("what is the highest-rated track?") is None


def test_ignores_plain_questions():
    assert detect_ambiguity("how many tracks are there?") is None
    assert detect_ambiguity("what are the 5 longest tracks?") is None
    assert detect_ambiguity("list all genres") is None


def test_flags_top_without_a_dimension():
    # "top tracks" / "top customers" -- by what? as ambiguous as "best"
    assert detect_ambiguity("what are the top tracks?")
    assert detect_ambiguity("who are the top customers?")


def test_top_with_a_dimension_is_specific():
    # "top" + a concrete ranking dimension names what to rank by -> not ambiguous
    assert detect_ambiguity("what are the top 5 longest tracks?") is None
    assert detect_ambiguity("the top 3 highest-priced tracks") is None


def test_normalizes_short_clarification_replies():
    assert normalize_clarification_response("sales") == (
        "rank the requested entities by total sales or revenue, descending"
    )
    assert normalize_clarification_response("total") == (
        "rank the requested entities by total sales or revenue, descending"
    )
    assert normalize_clarification_response("count") == (
        "rank the requested entities by count, descending"
    )
    assert normalize_clarification_response("rating") == (
        "rank the requested entities by rating or score, descending"
    )
    assert normalize_clarification_response("by average order value") == (
        "rank the requested entities by average order value, descending"
    )


def test_parses_clarification_intent_from_options(tmp_path):
    db = build(tmp_path / "t.db")
    tables = introspect(db)
    options = build_clarification_options("who are the best customers?", tables=tables)

    intent = parse_clarification_intent("total sales", options)

    assert intent == {
        "metric": "total",
        "aggregation": "sum",
        "measure": "invoice.total",
        "sort": "desc",
        "source": "schema",
        "raw_response": "total sales",
    }
    rendered = format_clarification_intent(intent)
    assert "metric: total" in rendered
    assert "measure: invoice.total" in rendered


def test_parses_short_freeform_clarification_intent():
    assert parse_clarification_intent("sales") == {
        "metric": "total_sales",
        "aggregation": "sum",
        "measure": "sales, revenue, amount, or total column",
        "sort": "desc",
        "source": "freeform",
        "raw_response": "sales",
    }


def test_schema_options_cover_numeric_columns(tmp_path):
    db = build(tmp_path / "t.db")
    tables = introspect(db)

    options = build_clarification_options("who are the best customers?", tables=tables)
    labels = {o["label"] for o in options}
    assert "Total" in labels        # invoice.total is a numeric sum candidate
    assert "Count" in labels
    assert "Rating" in labels       # review.rating is a numeric avg candidate
    assert {o["confidence"] for o in options} == {"fallback"}
    # All schema-derived options carry the actual table.column as measure
    measures = {o["intent"]["measure"] for o in options if o.get("intent")}
    assert any("invoice.total" in m or "." in m or m == "*" for m in measures)


def test_semantic_options_preserve_metric_alias_labels():
    metrics = [m for m in load_metrics() if m.name == "mrr"]
    options = build_clarification_options("best accounts", metrics=metrics)
    assert options[0]["label"] == "MRR"
    assert options[0]["source"] == "semantic_layer"
    assert options[0]["confidence"] == "governed"


def test_semantic_options_suppress_schema_fallbacks(tmp_path):
    db = build(tmp_path / "t.db")
    tables = introspect(db)
    metrics = [m for m in load_metrics() if m.name == "mrr"]

    options = build_clarification_options("best accounts", tables=tables, metrics=metrics)

    assert [o["label"] for o in options] == ["MRR"]
    assert options[0]["source"] == "semantic_layer"
    assert options[0]["confidence"] == "governed"


# --- end to end ----------------------------------------------------------------

def test_ambiguous_question_asks_instead_of_answering(tmp_path):
    db = build(tmp_path / "t.db")
    model = FakeModel("SELECT 1")
    res = answer_question(db, "who are the best customers?", model=model)
    assert res.clarification and not res.sql           # asked, didn't write SQL
    assert "metric" in res.answer.lower()
    assert "Total" in res.answer                       # the sum option label
    assert model.last_prompt is None                   # never called the model


def test_clear_question_is_unaffected(tmp_path):
    db = build(tmp_path / "t.db")
    res = answer_question(db, "how many tracks are there?",
                          model=FakeModel("SELECT COUNT(*) FROM track"))
    assert res.clarification is None
    assert res.execution.ok and "306" in res.answer


def test_hitl_clarification_resumes_with_user_metric(tmp_path):
    db = build(tmp_path / "t.db")
    model = FakeModel(
        "SELECT customer_id, SUM(total) AS total_spend "
        "FROM invoice GROUP BY customer_id ORDER BY total_spend DESC LIMIT 5"
    )

    thread_id, first = start_question_session(db, "who are the best customers?", model=model)

    assert first["clarification"]
    assert "Total" in {o["label"] for o in first["options"]}
    assert "Count" in {o["label"] for o in first["options"]}
    assert first["question"] == "who are the best customers?"
    assert model.last_prompt is None

    _, mid = resume_question_session(thread_id, "sales")
    # HITL now interrupts a SECOND time for plan approval after the clarification is
    # resolved; approve the (unchanged) plan to run it to completion.
    assert isinstance(mid, dict) and mid.get("plan")
    _, result = resume_question_session(thread_id, {"decision": "approve"})

    assert result.clarification is None
    assert result.execution.ok
    assert result.retrieved_tables
    assert "Use this resolved clarification intent" in model.last_prompt
    assert "metric: total" in model.last_prompt
    assert "rank the requested entities by total sales or revenue" in model.last_prompt
    resumed = next(t for t in result.trace if t["node"] == "clarify_check" and t.get("resumed"))
    assert resumed["clarification_response"] == "sales"
    assert resumed["intent_verdict"] == "ok"
    # "sales" is a freeform reply (doesn't hit the option label "Total" exactly);
    # freeform matching still resolves to a total_sales intent with a generic measure.
    assert resumed["clarification_intent"]["metric"] == "total_sales"
    generated = next(t for t in result.trace if t["node"] == "generate_sql")
    assert generated["clarification_intent"]["measure"]  # some measure is present


def test_invalid_hitl_clarification_refuses_without_guessing(tmp_path):
    db = build(tmp_path / "t.db")
    model = FakeModel("SELECT 1")

    thread_id, first = start_question_session(db, "who are the best customers?", model=model)
    assert isinstance(first, dict)

    _, result = resume_question_session(thread_id, "profit")

    assert not result.execution.ok
    assert "couldn't map that clarification" in result.answer.lower()
    assert result.sql == ""
    assert model.last_prompt is None
    resumed = next(t for t in result.trace if t["node"] == "clarify_check" and t.get("resumed"))
    assert resumed["intent_verdict"] == "refuse"
    assert resumed["clarification_intent"] is None
