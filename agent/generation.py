"""SQL generation + answer formatting helpers, and the agent's result type.

These are the leaf steps the graph wires together: turn a model response into SQL
(``generate_sql`` / ``_extract_sql``), render an execution result as prose
(``_format_answer``), and the ``AnswerResult`` the agent returns. Kept separate from
``pipeline`` (the public ``answer_question`` entry) and ``graph`` (the state machine)
so neither becomes a grab-bag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.execution import ExecutionResult
from agent.prompts import CANNOT_ANSWER, SQL_SYSTEM_PROMPT
from agent.retrieval.contracts import RetrievalResult

_NO_QUERY = "(no query run)"

# Pull the contents out of the first ```...``` block; the model often wraps SQL
# in a fence with prose around it ("Here's a query: ```sql ... ```").
_FENCE_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

# A refusal the model wrote as an explanation with the sentinel alone on its own line. Matching
# is deliberately anchored to a WHOLE line (horizontal whitespace only either side): the token
# appearing inside a sentence -- an explanation that mentions it, or a question asking about it
# -- must never be turned into a refusal.
_CANNOT_ANSWER_LINE = re.compile(r"(?im)^[^\S\r\n]*CANNOT_ANSWER[^\S\r\n]*$")


def _extract_sql(text: str) -> str:
    text = (text or "").strip()
    m = _FENCE_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    # No code fence: the model sometimes prefixes reasoning prose ("Let me check the latest
    # dates...\nSELECT ..."). Strip to the first SQL statement keyword so governance and
    # execution see SQL, not prose (unstripped prose fails to parse and is misreported as a
    # governance violation). Prefer a keyword that starts a line; else the first anywhere.
    km = re.search(r"(?im)^\s*(?:WITH|SELECT)\b", text) or re.search(r"(?is)\b(?:WITH|SELECT)\b", text)
    if km:
        return text[km.start():].strip()
    # No SQL anywhere, but the model signed its explanation off with the sentinel on its own
    # line. Normalise to the bare sentinel so the generation step recognises this as the decline
    # it is. Without this the explanation itself was returned as "SQL" and reached the governance
    # parser, whose failure was then reported to the user as a governance violation.
    # SQL keeps priority: this runs only after both checks above found none.
    if _CANNOT_ANSWER_LINE.search(text):
        return CANNOT_ANSWER
    return text


def generate_sql(question: str, schema: str, model, *, semantic_block: str = "") -> str:
    """Ask the model for a SQL query given the question and (top-k) schema."""
    prompt = SQL_SYSTEM_PROMPT.format(schema=schema, question=question,
                                     semantic_block=semantic_block)
    response = model.invoke(prompt)
    return _extract_sql(getattr(response, "content", response))


def _fmt(value) -> str:
    return "(null)" if value is None else str(value)


def _format_answer(execution: ExecutionResult) -> str:
    if not execution.ok:
        return f"I couldn't answer that: {execution.error}"
    if not execution.rows:
        return "No matching results."
    if len(execution.rows) == 1 and len(execution.columns) == 1:
        return f"{execution.columns[0]}: {_fmt(execution.rows[0][0])}"
    header = ", ".join(execution.columns)
    n = len(execution.rows)
    preview = "; ".join("(" + ", ".join(_fmt(c) for c in r) + ")" for r in execution.rows[:5])
    if execution.truncated:
        suffix = " (showing 5, more available)"
    elif n > 5:
        suffix = f" (showing 5 of {n})"
    else:
        suffix = f" ({n} {'row' if n == 1 else 'rows'})"
    return f"[{header}]{suffix}: {preview}"


@dataclass
class AnswerResult:
    question: str
    retrieved_tables: list[str]
    sql: str
    execution: ExecutionResult
    answer: str
    assumptions: list[str] = field(default_factory=list)
    clarification: str | None = None    # set when the agent asked to clarify instead of answering
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)   # token + latency totals for the run
    python_analysis: dict | None = None   # a Python step's parsed output (analysis + any chart)
    retrieval_result: "RetrievalResult | None" = None   # reconstructed typed retrieval trace
