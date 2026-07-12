"""LLM query rewrite with a governed-metric guardrail. The rewrite adds time/entity
context only; it must not redefine governed metrics, and the original question is always
preserved (kept in state for the answer/trace). This is the ONLY pre-step LLM node."""
from __future__ import annotations

import json
import re
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
    # guardrail: a metric is "dropped" if ANY of its surface forms (canonical name OR an
    # alias) was in the question but NONE survives in the rewrite. Aliases matter -- real
    # configs rely on them, and a question often hits an alias, not the canonical name.
    def _present(text: str, term: str) -> bool:
        # word-boundary match (like metric retrieval), so a short alias "arr" does not
        # match inside "array" and wrongly disable the enhancement.
        return re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE) is not None

    dropped = []
    for met in metrics:
        forms = [met.name, *met.aliases]
        if (any(_present(question, f) for f in forms)
                and not any(_present(enhanced, f) for f in forms)):
            dropped.append(met.name)
    if dropped:
        warnings.append(f"rewrite dropped governed term(s) {dropped}; kept original")
        enhanced = question
    return EnhanceResult(enhanced, str(data.get("rewrite_diff", "")),
                         preserved_terms=governed, governed_terms=governed,
                         warnings=warnings)
