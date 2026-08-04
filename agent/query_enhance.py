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


# Deterministic backstop for the invented-time failure class: the rewrite must not add a
# time frame the question did not contain. An invented window gets hardened by the planner
# into date logic against a static DB, and the step never yields SQL (the MRR demo bug). The
# prompt forbids this too; this guard catches the residual the prompt still lets through.
# It errs toward catching: the fallback is a SAFE revert to the honest original question, so
# a false positive only forfeits the rewrite's optimization, never a correct answer (the
# per-gate rule -- a soft-revert fallback licenses a high-recall bias). Known limits: it
# targets relative/explicit date markers (relative phrases, numeric windows, explicit years),
# not every paraphrase of time, and NOT a new constraint that reuses a marker already in the
# question (that belongs to the broader no-unentailed-constraint check). Downstream validate
# and the execution oracle catch what slips past.
_TIME_MARKERS = re.compile(
    r"\b(?:"
    r"today|yesterday|tomorrow|"
    r"(?:this|last|past|next|current|previous|trailing|recent)\s+"
    r"(?:month|quarter|year|week|day|days|weeks|months|quarters|years)|"
    r"(?:last|past|previous|trailing|next)\s+\d+\s+(?:day|week|month|quarter|year)s?|"
    r"year[-\s]?to[-\s]?date|ytd|mtd|qtd|"
    r"so\s+far|as\s+of|"
    r"since\s+\d+|during\s+\d{4}|in\s+\d{4}|q[1-4]\s+\d{4}|\d{4}-\d{2}-\d{2}"
    r")\b",
    re.IGNORECASE,
)


def _introduced_time(original: str, rewrite: str) -> bool:
    """True if the rewrite has a time marker the original did not (case-insensitive)."""
    orig = {m.group(0).lower() for m in _TIME_MARKERS.finditer(original)}
    new = {m.group(0).lower() for m in _TIME_MARKERS.finditer(rewrite)}
    return bool(new - orig)


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
    # Evaluate both guards against the model's rewrite BEFORE reverting, so the trace records
    # which failure(s) occurred even when they co-occur -- a dropped-term revert must not hide
    # the invented-time signal. Either firing reverts to the original question.
    invented_time = _introduced_time(question, enhanced)
    if dropped:
        warnings.append(f"rewrite dropped governed term(s) {dropped}; kept original")
    if invented_time:
        warnings.append("rewrite introduced a time frame absent from the question; kept original")
    if dropped or invented_time:
        enhanced = question
    return EnhanceResult(enhanced, str(data.get("rewrite_diff", "")),
                         preserved_terms=governed, governed_terms=governed,
                         warnings=warnings)
