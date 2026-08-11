"""Stage 3C follow-up — retrieval-only, LLM-free CHANNEL attribution on the 20-table fixture.

Instead of only the fused result, it records each gold table's rank in EACH layer — lexical channel,
dense channel, fused RRF, then the deterministic Top-5 selector and the relation context — under a
deterministic semantic OFF and ON view. Semantic ON's protected anchors come from the REAL
MetricRegistry (alias-bound hits on the question), never reverse-injected from gold. The value channel
is inert on this fixture (no searchable columns) and is asserted inert and excluded from attribution.
No BM25, no selector, no RRF-weight or TopK change, no real LLM, no paid API.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.retrieval.aggregate import aggregate
from agent.retrieval.backends import InMemoryDenseBackend
from agent.retrieval.channels import DenseChannel, LexicalChannel, ValueChannel
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.metric_match import MetricMatchProvider
from agent.retrieval.pipeline import run_retrieval
from agent.retrieval.selector import protected_anchors
from agent.retrieval.value_backend import FakeValueBackend
from agent.semantic_layer import MetricRegistry
from evalharness.channel_attribution import channel_recall_precision, classify_gold_table
from evalharness.golden import load_saas_metrics

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"


def _ordered(ranks: dict) -> list[str]:
    return sorted(ranks, key=lambda t: ranks[t])


def _alias_hits(registry, question, threshold=0.5):
    # alias-only binding mirrors preflight: dense metric hits are discovery, never request-bound.
    return [h for h in registry.retrieve_matches(question, threshold=threshold) if h.match_type == "alias"]


def per_case(tables, case, value_backend, config, registry, *, semantic_layer: bool) -> dict:
    q, gold = case.question, list(case.required_tables)
    bound = _alias_hits(registry, q) if semantic_layer else []
    protected = set(protected_anchors(MetricMatchProvider(tables).from_hits(bound)))

    lex_ranks = {r.table: r.channel_rank for r in aggregate("lexical", LexicalChannel().signals(q, tables))}
    dense_ranks = {r.table: r.channel_rank
                   for r in aggregate("dense", DenseChannel(InMemoryDenseBackend()).signals(q, tables))}
    value_inert = ValueChannel(value_backend).signals(q, tables) == [] if config.value_backend == "es" else True

    vb = value_backend if config.value_backend == "es" else None
    rr = run_retrieval(q, tables, config, k=5, metric_hits=bound, value_backend=vb)
    fused_ranks = {c.table: c.fusion_rank for c in rr.candidates}
    selection, context = set(rr.selection.anchor_tables), set(rr.relation_plan.context_tables)
    lex_ord, dense_ord, fused_ord = _ordered(lex_ranks), _ordered(dense_ranks), _ordered(fused_ranks)

    golds = []
    for g in gold:
        golds.append({
            "table": g, "lex_rank": lex_ranks.get(g), "dense_rank": dense_ranks.get(g),
            "fused_rank": fused_ranks.get(g), "selector_kept": g in selection,
            "context_recovered": g in context, "governance_protected": g in protected,
            "tags": classify_gold_table(lex_rank=lex_ranks.get(g), dense_rank=dense_ranks.get(g),
                                        fused_rank=fused_ranks.get(g), selector_kept=g in selection,
                                        context_recovered=g in context, governance_protected=g in protected),
        })
    return {
        "id": case.id, "category": case.category, "semantic_layer": semantic_layer,
        "required_tables": gold, "protected_anchors": sorted(protected), "value_inert": value_inert,
        "lex": channel_recall_precision(lex_ord, gold), "dense": channel_recall_precision(dense_ord, gold),
        "rrf": channel_recall_precision(fused_ord, gold),
        "lex_top5": lex_ord[:5], "dense_top5": dense_ord[:5], "fused_top5": fused_ord[:5],
        "selection": sorted(selection), "context": sorted(context), "gold_tables": golds,
    }


def run_channel_attribution(tables, db, cases, value_backend, config, registry) -> list[dict]:
    out = []
    for sem in (False, True):
        for case in cases:
            out.append(per_case(tables, case, value_backend, config, registry, semantic_layer=sem))
    return out


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(records: list[dict]) -> dict:
    out = {"value_inert_all": all(r["value_inert"] for r in records)}
    for sem, label in ((False, "off"), (True, "on")):
        rows = [r for r in records if r["semantic_layer"] == sem and r["category"] != "control"]

        def chan(key, kind, k):
            return _mean(r[key][kind][k] for r in rows)

        # per-gold-table aggregation
        golds = [(r, g) for r in rows for g in r["gold_tables"]]
        tag_counts = Counter(t for _, g in golds for t in g["tags"])
        tag_ids = {tag: sorted({r["id"] for r, g in golds if tag in g["tags"]}) for tag in tag_counts}
        # confounder frequency: non-gold tables appearing in each channel's Top-5
        def confounders(top_key):
            c = Counter()
            for r in rows:
                gset = set(r["required_tables"])
                c.update(t for t in r[top_key] if t not in gset)
            return c.most_common(6)
        # governance vs explicit-clue split for gold tables lost by the channels
        gov_only = sorted({r["id"] for r, g in golds if g["governance_protected"]
                           and (g["lex_rank"] is None or g["lex_rank"] > 5)})
        lexically_retrievable = sorted({(r["id"], g["table"]) for r, g in golds
                                        if g["lex_rank"] is not None and g["lex_rank"] <= 5})
        out[label] = {
            "n_metric_cases": len({r["id"] for r in rows}),
            "recall": {ch: {k: chan(ch, "recall", k) for k in (5, 10, 15)} for ch in ("lex", "dense", "rrf")},
            "precision": {ch: {k: chan(ch, "precision", k) for k in (5, 10, 15)} for ch in ("lex", "dense", "rrf")},
            "selection_recall": _mean(1.0 if g["selector_kept"] else 0.0 for _, g in golds),
            "context_recall": _mean(1.0 if g["context_recovered"] else 0.0 for _, g in golds),
            "case_categories": {t: tag_ids[t] for t in tag_counts},
            "top_confounders": {"lexical": confounders("lex_top5"), "dense": confounders("dense_top5"),
                                "rrf": confounders("fused_top5")},
            "governance_only_required_ids": gov_only,
            "lexically_retrievable_gold": lexically_retrievable[:20],
        }
    return out


def main() -> None:  # pragma: no cover - real fastembed locally (free; no paid API)
    import tempfile

    from agent.retrieval.value_index import build_value_index
    with tempfile.TemporaryDirectory() as workdir:
        db = str(build(Path(workdir) / "saas_expanded.db", confounders=True))
        tables = introspect(db)
        backend = FakeValueBackend()
        build_value_index(tables, db, backend)                  # inert on saas
        records = run_channel_attribution(tables, db, load_saas_metrics(), backend,
                                          RetrievalConfig.dense_value(), MetricRegistry.load())
    report = {"kind": "channel_attribution", "measured": True, "llm": False,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_tables": len(tables), "config": "dense_value",
              "summary": summarize(records), "records": records}
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"channel_attribution_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    s = report["summary"]
    print(f"wrote {out} · value_inert={s['value_inert_all']}")
    for mode in ("off", "on"):
        m = s[mode]
        print(f"[{mode}] recall lex/dense/rrf @15 = "
              f"{m['recall']['lex'][15]}/{m['recall']['dense'][15]}/{m['recall']['rrf'][15]} · "
              f"selection_recall={m['selection_recall']} context_recall={m['context_recall']}")
        print(f"      categories: {{ {', '.join(f'{k}:{len(v)}' for k,v in m['case_categories'].items())} }}")


if __name__ == "__main__":  # pragma: no cover
    main()
