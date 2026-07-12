"""PlannerNode: ask the model to decompose a question into a 1-2 step plan
(one SQL step + optional Python step). Returns a Plan; unparseable output yields an
empty Plan so validate_plan rejects it and the graph replans (bounded)."""
from __future__ import annotations

import json
import re

from agent.plan import Plan, deserialize_plan
from agent.prompts import PLANNER_PROMPT

_FENCE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)


def _json_arrays(text: str):
    """Yield each JSON array in text: a ```json fence first, then every '[' raw_decoded
    in place. Yielding candidates (rather than returning the first) lets the caller skip
    an earlier non-plan list -- e.g. an echoed ["sql", "python"] -- and keep looking for
    one that actually holds steps."""
    fence = _FENCE.search(text)
    if fence:
        try:
            data = json.loads(fence.group(1))
            if isinstance(data, list):
                yield data
        except ValueError:
            pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            data, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(data, list):
            yield data


def _parse_steps(text: str) -> list[dict]:
    for data in _json_arrays((text or "").strip()):
        out = [{"kind": str(i["kind"]), "instruction": str(i["instruction"])}
               for i in data if isinstance(i, dict) and "kind" in i and "instruction" in i]
        if out:                          # first array that actually yields steps wins
            return out
    return []


def plan_query(question: str, schema: str, model, *, semantic_block: str = "",
               feedback: str = "") -> Plan:
    prompt = PLANNER_PROMPT.format(schema=schema, question=question,
                                   semantic_block=semantic_block)
    if feedback:
        # Prepend so the prompt still ends with "JSON:" (the fake-model recognition
        # marker). This is the plan-repair analogue of feeding a failed SQL's error back.
        prompt = f"Your previous plan was rejected: {feedback}. Correct it.\n\n{prompt}"
    response = model.invoke(prompt)
    text = getattr(response, "content", response)
    return deserialize_plan(_parse_steps(text))
