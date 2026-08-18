"""A prose refusal ending in a standalone CANNOT_ANSWER must decline cleanly.

The model sometimes explains why it cannot answer and puts the sentinel on its own line at the
end. Before this, ``_extract_sql`` found no fence and no SQL keyword and returned the whole
explanation verbatim, so prose was handed to the governance parser: sqlglot failed, and the
resulting parse error -- ANSI highlight bytes included -- was reported to the user as a
"governance violation". These tests pin the decline as a normal generation-stage decline.

The sentinel must be a STANDALONE line. A question or an explanation that merely contains the
token must not be turned into a refusal.
"""
from __future__ import annotations

import agent.graph as graph
from agent.generation import _extract_sql
from agent.graph import run_agent

_PROSE_DECLINE = (
    "The user table has no email column, and neither does the account table.\n"
    "There is no email data anywhere in the schema.\n"
    "\n"
    "CANNOT_ANSWER"
)


# --- _extract_sql: normalisation ------------------------------------------------------------

def test_prose_ending_in_a_standalone_sentinel_normalises_to_the_sentinel():
    assert _extract_sql(_PROSE_DECLINE) == "CANNOT_ANSWER"


def test_standalone_sentinel_is_case_insensitive_and_tolerates_indentation():
    for raw in ("no email column here\ncannot_answer",
                "no email column here\n  Cannot_Answer  ",
                "cannot_answer",
                "explanation\r\nCANNOT_ANSWER\r\n"):
        assert _extract_sql(raw) == "CANNOT_ANSWER", raw


def test_a_bare_sentinel_with_a_trailing_explanation_still_declines():
    # pre-existing behaviour: the sentinel leads, so the decline check already caught it
    assert _extract_sql("CANNOT_ANSWER - no weather table").upper().startswith("CANNOT_ANSWER")


# --- negative: a substring must never trigger -----------------------------------------------

def test_the_token_inside_a_sentence_does_not_normalise():
    text = "I CANNOT_ANSWER that without a date column, sorry."
    assert _extract_sql(text) == text


def test_a_question_about_the_token_does_not_normalise():
    text = "Which tickets contain the phrase CANNOT_ANSWER in their body?"
    assert _extract_sql(text) == text


def test_the_token_glued_to_other_words_does_not_normalise():
    text = "XCANNOT_ANSWERX"
    assert _extract_sql(text) == text


# --- SQL keeps priority ---------------------------------------------------------------------

def test_real_sql_wins_over_a_standalone_sentinel_line():
    text = "I am not fully sure.\nCANNOT_ANSWER\nSELECT COUNT(*) FROM account"
    assert _extract_sql(text) == "SELECT COUNT(*) FROM account"


def test_fenced_sql_wins_over_a_standalone_sentinel_line():
    text = "CANNOT_ANSWER\n```sql\nSELECT 1\n```"
    assert _extract_sql(text) == "SELECT 1"


def test_a_cte_still_wins_over_a_standalone_sentinel_line():
    text = "CANNOT_ANSWER\nWITH m AS (SELECT 1) SELECT * FROM m"
    assert _extract_sql(text).startswith("WITH m AS")


# --- graph level: a normal decline, never a governance/parser failure -----------------------

class _ProseDeclineModel:
    """Plans one SQL step, then refuses in prose with a trailing standalone sentinel."""

    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        if "governed metric terms" in text:
            return type("R", (), {"content": '{"enhanced_question": ""}'})()
        if text.rstrip().endswith("JSON:") and "Output a JSON array of steps" in text:
            return type("R", (), {
                "content": '[{"kind": "sql", "instruction": "answer the question"}]'})()
        return type("R", (), {"content": _PROSE_DECLINE})()


def _decline_run(saas_db):
    return run_agent(saas_db, "List our users' email addresses", model=_ProseDeclineModel())


def test_prose_decline_is_a_generation_stage_decline(saas_db):
    res = _decline_run(saas_db)
    generate = next(t for t in res.trace if t.get("node") == "generate_sql")
    assert generate.get("declined") is True
    assert res.sql == "CANNOT_ANSWER"


def test_prose_decline_never_reaches_execution_or_governance(saas_db):
    res = _decline_run(saas_db)
    nodes = [t.get("node") for t in res.trace]
    assert "execute" not in nodes
    assert "validate" not in nodes


def test_prose_decline_leaks_no_parser_error_to_the_user(saas_db):
    res = _decline_run(saas_db)
    surfaced = f"{res.answer}\n{res.execution.error}"
    assert "governance violation" not in surfaced
    assert "could not parse SQL" not in surfaced
    assert "\x1b" not in surfaced, "raw ANSI parser highlighting must not reach the user"
    assert "couldn't write" in res.answer.lower()


def test_a_model_that_returns_real_sql_is_unaffected(saas_db):
    from tests.conftest import PlanningFakeModel

    res = run_agent(saas_db, "how many accounts are there",
                    model=PlanningFakeModel("SELECT COUNT(*) FROM account"))
    assert res.execution.ok and res.sql.startswith("SELECT")
    assert any(t.get("node") == "execute" for t in res.trace)
