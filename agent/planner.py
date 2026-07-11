"""PlannerNode: ask the model to decompose a question into a 1-2 step plan
(one SQL step + optional Python step). Returns a Plan; unparseable output yields an
empty Plan so validate_plan rejects it and the graph replans (bounded)."""
from __future__ import annotations

import json
import re

from agent.plan import Plan, deserialize_plan
from agent.prompts import PLANNER_PROMPT

_FENCE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)


def _extract_json_array(text: str) -> list:
    """Return the first JSON array in text, or []. Prefer a ```json fence; otherwise
    scan each '[' and raw_decode from there -- so a valid plan followed by prose or a
    later '[...]' isn't swallowed into an unparseable greedy match."""
    fence = _FENCE.search(text)
    if fence:
        try:
            data = json.loads(fence.group(1))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            data, _ = decoder.raw_decode(text[i:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, list):
            return data
    return []


def _parse_steps(text: str) -> list[dict]:
    data = _extract_json_array((text or "").strip())
    out = []
    for item in data:
        if isinstance(item, dict) and "kind" in item and "instruction" in item:
            out.append({"kind": str(item["kind"]), "instruction": str(item["instruction"])})
    return out


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
