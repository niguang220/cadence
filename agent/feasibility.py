"""Deterministic feasibility gate. The ONLY reliable deterministic signal is "did
retrieval find any relevant tables" -- the semantic retriever already judges "is this
about our data" better than lexical word-matching could (which false-refuses on plurals/
synonyms and contradicts a successful recall). So this gate refuses only on empty recall
and traces other signals as risk; it never does brittle catalog word-matching. No LLM."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeasibilityVerdict:
    feasible: bool
    reason_code: str = ""
    message: str = ""
    risks: list[str] = field(default_factory=list)


def assess_feasibility(question, tables, recalled, metrics, paths) -> FeasibilityVerdict:
    if not recalled:                        # the one reliable deterministic refusal
        return FeasibilityVerdict(False, "no_recalled_tables",
                                  "No tables look relevant to this question.")
    risks = []
    if len(recalled) > 1 and not paths:
        risks.append("possible_missing_join")   # direct-edge hint missing; NOT a refusal
    return FeasibilityVerdict(True, risks=risks)
