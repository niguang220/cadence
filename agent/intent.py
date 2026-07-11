"""A cheap, deterministic routing guard -- NOT an intelligent intent classifier.
It only rejects obvious non-data inputs (greetings, meta-questions); everything else
defaults to 'data' (a data agent is mostly given data questions, and a wrong refusal is
worse than passing an odd question through to the schema/feasibility gates)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_GREETING = re.compile(r"^\W*(hi|hello|hey|thanks|thank you|good (morning|evening))\b",
                       re.IGNORECASE)
_META = re.compile(r"\b(who are you|what can you do|are you (a )?(bot|ai)|help me use)\b",
                   re.IGNORECASE)


@dataclass
class IntentVerdict:
    kind: str        # "data" | "out_of_scope"
    reason: str = ""


def classify_intent(question: str) -> IntentVerdict:
    q = (question or "").strip()
    if not q:
        return IntentVerdict("out_of_scope", "empty question")
    if _GREETING.search(q) or _META.search(q):
        return IntentVerdict("out_of_scope", "not a data question")
    return IntentVerdict("data")
