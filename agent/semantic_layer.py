"""Structured metric semantic layer: governed business-term definitions, retrieved
and injected into SQL generation. Each metric gives atomic building blocks (a measure
aggregation + required filters), NOT a full SELECT -- the LLM still assembles the query.
"""
from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

_SPEC_PATH = Path(__file__).resolve().parent / "semantic_layer_metrics.json"

@dataclass
class MetricDef:
    name: str
    aliases: list[str]
    definition: str
    measure: str
    grain: str
    required_filters: list[str]
    common_mistake: str


class MetricRegistry:
    """Single access point for governed metric definitions.

    The registry owns metric loading, matching, and prompt formatting so graph
    nodes don't grow separate semantic-layer logic over time.
    """

    def __init__(self, metrics: list[MetricDef], *, embed=None):
        self.metrics = metrics
        self._embed = embed

    @classmethod
    def load(cls, path: str | Path = _SPEC_PATH, *, embed=None) -> "MetricRegistry":
        return cls(load_metrics(path), embed=embed)

    def retrieve(self, question: str, *, threshold: float = 0.5, top_k: int = 3) -> list[MetricDef]:
        return _retrieve_metrics(question, self.metrics, threshold=threshold,
                                 top_k=top_k, embed=self._embed)

    def format(self, metrics: list[MetricDef]) -> str:
        return format_metrics(metrics)


def load_metrics(path: str | Path = _SPEC_PATH) -> list[MetricDef]:
    data = json.loads(Path(path).read_text())
    return [MetricDef(**d) for d in data]

def format_metrics(metrics: list[MetricDef]) -> str:
    if not metrics:
        return ""
    blocks = []
    for m in metrics:
        filters = "\n".join(f"    • {f}" for f in m.required_filters)
        blocks.append(
            f"- {m.name} — {m.definition}\n"
            f"  measure: {m.measure}   grain: {m.grain}\n"
            f"  filters you MUST apply:\n{filters}\n"
            f"  Common mistake: {m.common_mistake}")
    body = "\n".join(blocks)
    return ("Use these governed metric definitions (company conventions). "
            "Apply them; assemble the full query yourself:\n" + body + "\n\n")

def _cos(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def _default_embed(texts):
    from agent.hybrid_retriever import _embed   # reuse the project's fastembed model
    return _embed(texts)

def _retrieve_metrics(question, metrics, *, threshold=0.5, top_k=3, embed=None):
    embed = embed or _default_embed
    q = question.lower()
    chosen, seen = [], set()
    # 1) alias exact -- always included (precision), not capped
    for m in metrics:
        if any(re.search(rf"\b{re.escape(a.lower())}\b", q) for a in m.aliases):
            if m.name not in seen:
                chosen.append(m); seen.add(m.name)
    # 2) dense recall -- fill up to top_k by descending similarity, threshold-gated
    rest = [m for m in metrics if m.name not in seen]
    if rest and len(chosen) < top_k:
        try:
            vecs = embed([question] + [f"{m.name} {' '.join(m.aliases)} {m.definition}" for m in rest])
            qv, mvs = vecs[0], vecs[1:]
            scored = sorted(((_cos(qv, mv), m) for m, mv in zip(rest, mvs)),
                            key=lambda x: x[0], reverse=True)
            for score, m in scored:
                if len(chosen) >= top_k:
                    break
                if score >= threshold and m.name not in seen:
                    chosen.append(m); seen.add(m.name)
        except Exception:
            pass   # alias hits already collected; dense recall is best-effort
    return chosen


def retrieve_metrics(question, metrics, *, threshold=0.5, top_k=3, embed=None):
    """Compatibility wrapper for callers/tests that still pass metric lists."""
    return MetricRegistry(metrics, embed=embed).retrieve(question, threshold=threshold, top_k=top_k)
