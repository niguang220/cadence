"""A bounded semantic-consistency check on a SQL step -- the 2nd (and last) LLM node.

After a SQL step validates OK (structurally + governance-clean), this asks the model a
single narrow question: does the SQL and the rows it returned actually answer the
QUESTION's intent -- the measure, entity, and grain? A confident ``not ok`` verdict is
fed back into the SAME generate/repair loop (its ``repair_hint`` becomes the repair
problem), bounded by the shared ``attempts`` budget; on exhaustion the graph REFUSES
rather than answering a subtly-wrong query.

Fail OPEN: on any parse failure (or a malformed judge reply) the verdict defaults to
``ok=True``. A broken judge must not block an otherwise-valid query -- the repair budget
already bounds any loop, so the safe default is to let the query through.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from agent.execution import ExecutionResult
from agent.prompts import SEMANTIC_CONSISTENCY_PROMPT

_RESULT_PREVIEW_ROWS = 5


@dataclass
class ConsistencyVerdict:
    ok: bool
    mismatch_kind: str = ""
    expected: str = ""
    observed: str = ""
    evidence: str = ""
    repair_hint: str = ""


def _format_result(result: ExecutionResult) -> str:
    """A compact preview (columns + first few rows) so the judge sees the shape/grain of
    what came back, not the whole result set."""
    columns = ", ".join(result.columns) or "(no columns)"
    if not result.rows:
        return f"columns: {columns}\n(no rows)"
    preview = "\n".join(str(list(r)) for r in result.rows[:_RESULT_PREVIEW_ROWS])
    more = "" if len(result.rows) <= _RESULT_PREVIEW_ROWS else f"\n... (+{len(result.rows) - _RESULT_PREVIEW_ROWS} more rows)"
    return f"columns: {columns}\n{preview}{more}"


def check_semantic_consistency(question: str, sql: str, result: ExecutionResult,
                               model) -> ConsistencyVerdict:
    """Judge whether ``sql``/``result`` are semantically consistent with ``question``.

    Parses the model's JSON verdict into a ``ConsistencyVerdict``; on parse failure or a
    non-string reply defaults to ``ok=True`` (fail-open -- a broken judge must not block a
    query)."""
    prompt = SEMANTIC_CONSISTENCY_PROMPT.format(
        question=question, sql=sql, result=_format_result(result))
    text = getattr(model.invoke(prompt), "content", "")
    if not isinstance(text, str):
        return ConsistencyVerdict(ok=True)   # non-string content (None/list) -> broken judge, fail-open
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return ConsistencyVerdict(ok=True)                    # fail-open on a broken judge
    if not isinstance(data, dict):
        return ConsistencyVerdict(ok=True)
    ok = data.get("ok", True)
    if not isinstance(ok, bool):
        return ConsistencyVerdict(ok=True)   # non-boolean verdict -> broken judge, fail-open
    # Only a clear, explicit ``"ok": false`` (a real JSON boolean) is a mismatch.
    return ConsistencyVerdict(
        ok=ok,
        mismatch_kind=str(data.get("mismatch_kind", "")),
        expected=str(data.get("expected", "")),
        observed=str(data.get("observed", "")),
        evidence=str(data.get("evidence", "")),
        repair_hint=str(data.get("repair_hint", "")),
    )
