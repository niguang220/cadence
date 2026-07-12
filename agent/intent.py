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
# social / greeting filler: an input made of ONLY these words is not a data question.
_FILLER = {"hi", "hello", "hey", "thanks", "thank", "you", "there", "how", "are",
           "good", "morning", "evening", "afternoon", "please", "doing", "today"}


@dataclass
class IntentVerdict:
    kind: str        # "data" | "out_of_scope"
    reason: str = ""


def classify_intent(question: str) -> IntentVerdict:
    q = (question or "").strip()
    if not q:
        return IntentVerdict("out_of_scope", "empty question")
    if _META.search(q):
        return IntentVerdict("out_of_scope", "not a data question")
    # A greeting/politeness input is out-of-scope ONLY when nothing substantive remains
    # after removing social filler -- so a greeting PREFIX on a real data question
    # ("hi, how many accounts?") stays in scope. Conservative: any non-filler word -> data.
    if _GREETING.search(q):
        tokens = re.findall(r"[a-z0-9']+", q.lower())
        if not any(t not in _FILLER for t in tokens):
            return IntentVerdict("out_of_scope", "not a data question")
    return IntentVerdict("data")
