"""LLM query rewrite with a governed-metric guardrail. The rewrite adds time/entity
context only; it must not redefine governed metrics, and the original question is always
preserved (kept in state for the answer/trace). This is the ONLY pre-step LLM node."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from agent.prompts import QUERY_ENHANCE_PROMPT
from agent.semantic_layer import MetricDef


@dataclass
class EnhanceResult:
    enhanced_question: str
    rewrite_diff: str = ""
    preserved_terms: list[str] = field(default_factory=list)
    governed_terms: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def enhance_query(question: str, metrics: list[MetricDef], model) -> EnhanceResult:
    governed = [m.name for m in metrics]
    prompt = QUERY_ENHANCE_PROMPT.format(question=question,
                                         governed_terms=", ".join(governed) or "(none)")
    text = getattr(model.invoke(prompt), "content", "")
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return EnhanceResult(question, governed_terms=governed)   # safe fallback
    enhanced = str(data.get("enhanced_question") or question)
    warnings = [str(w) for w in data.get("warnings", []) if w]
    # guardrail: if the rewrite DROPPED a governed term, do not use the lossy rewrite --
    # fall back to the original question so the governed definition still applies.
    dropped = [t for t in governed
               if t.lower() in question.lower() and t.lower() not in enhanced.lower()]
    if dropped:
        warnings.append(f"rewrite dropped governed term(s) {dropped}; kept original")
        enhanced = question
    return EnhanceResult(enhanced, str(data.get("rewrite_diff", "")),
                         preserved_terms=governed, governed_terms=governed,
                         warnings=warnings)
