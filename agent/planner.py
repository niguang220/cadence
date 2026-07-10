"""PlannerNode: ask the model to decompose a question into a 1-2 step plan
(one SQL step + optional Python step). Returns a Plan; unparseable output yields an
empty Plan so validate_plan rejects it and the graph replans (bounded)."""
from __future__ import annotations

import json
import re

from agent.plan import Plan, deserialize_plan
from agent.prompts import PLANNER_PROMPT

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def _parse_steps(text: str) -> list[dict]:
    text = (text or "").strip()
    m = _JSON_ARRAY.search(text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and "kind" in item and "instruction" in item:
            out.append({"kind": str(item["kind"]), "instruction": str(item["instruction"])})
    return out


def plan_query(question: str, schema: str, model, *, semantic_block: str = "") -> Plan:
    prompt = PLANNER_PROMPT.format(schema=schema, question=question,
                                   semantic_block=semantic_block)
    response = model.invoke(prompt)
    text = getattr(response, "content", response)
    return deserialize_plan(_parse_steps(text))
