"""Stage 3C follow-up — eval-only, LLM-free RRF lexical-weight sweep.

Tests whether equal-weight RRF fusion should change. Fixes dense weight = 1.0 and sweeps the lexical
weight over {0, 0.1, 0.25, 0.5, 0.75, 1.0} on the frozen 24 metric + 6 control cases over the 20-table
fixture, reconstructing the EXISTING fusion -> selector -> relation offline (weighted_rrf accepts a
per-channel weight, so NO RetrievalConfig field is added and NO production preset / weight / TopK /
semantic default is touched). The value channel is excluded by construction (only lexical + dense enter
channel_results). Semantic ON protected anchors come from the real MetricRegistry alias hits, never
from gold. Deterministic (fastembed dense); no paid API. w=1.0 is the current equal-weight baseline.
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
from agent.retrieval.channels import DenseChannel, LexicalChannel
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.fusion import weighted_rrf
from agent.retrieval.metric_match import MetricMatchProvider
from agent.retrieval.relation import plan_relations
from agent.retrieval.selector import NoOpSelector, TopKSelector, protected_anchors
from agent.retrieval.value_backend import FakeValueBackend
from agent.semantic_layer import MetricRegistry
from evalharness.channel_attribution import channel_recall_precision
from evalharness.golden import load_saas_metrics

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"
_WEIGHTS = ("0", "0.1", "0.25", "0.5", "0.75", "1.0")   # lexical weight; "1.0" == current equal weight
_BASELINE = "1.0"


def _alias_hits(registry, question, threshold=0.5):
    return [h for h in registry.retrieve_matches(question, threshold=threshold) if h.match_type == "alias"]


def _select_and_relate(candidates, protected, tables, cfg):
    selector = NoOpSelector() if len(candidates) <= cfg.context_anchor_k else TopKSelector()
    selection = selector.select(candidates, protected, context_anchor_k=cfg.context_anchor_k)
    plan = plan_relations(tables, selection.anchor_tables, max_hops=cfg.max_bridge_hops)  # shortest_path
    return list(selection.anchor_tables), list(plan.context_tables)


def sweep_case(tables, case, registry, cfg, *, semantic_layer: bool) -> dict:
    q, gold = case.question, list(case.required_tables)
    goldset = set(gold)
    protected = protected_anchors(MetricMatchProvider(tables).from_hits(
        _alias_hits(registry, q) if semantic_layer else []))
    lex = aggregate("lexical", LexicalChannel().signals(q, tables))
    dense = aggregate("dense", DenseChannel(InMemoryDenseBackend()).signals(q, tables))
    channel_results = {"lexical": lex, "dense": dense}           # value excluded by construction
    lex_ranks = {r.table: r.channel_rank for r in lex}
    dense_ranks = {r.table: r.channel_rank for r in dense}

    by_weight = {}
    for label in _WEIGHTS:
        cands = weighted_rrf(channel_results, rrf_constant=cfg.rrf_constant,
                             weights={"lexical": float(label), "dense": 1.0}, candidate_k=cfg.candidate_k)
        ordered = [c.table for c in cands]
        sel, ctx = _select_and_relate(cands, protected, tables, cfg)
        rp = channel_recall_precision(ordered, gold)
        by_weight[label] = {
            "recall": rp["recall"], "precision": rp["precision"],
            "selection_recall": len(goldset & set(sel)) / len(goldset) if goldset else None,
            "context_recall": len(goldset & set(ctx)) / len(goldset) if goldset else None,
            "gold_fused_ranks": {g: next((c.fusion_rank for c in cands if c.table == g), None) for g in gold},
        }
    return {"id": case.id, "category": case.category, "semantic_layer": semantic_layer,
            "required_tables": gold, "lex_ranks": lex_ranks, "dense_ranks": dense_ranks,
            "by_weight": by_weight}


def run_sweep(tables, cases, registry, cfg) -> list[dict]:
    return [sweep_case(tables, case, registry, cfg, semantic_layer=sem)
            for sem in (False, True) for case in cases]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(records: list[dict]) -> dict:
    out = {}
    for sem, mode in ((False, "off"), (True, "on")):
        metric = [r for r in records if r["semantic_layer"] == sem and r["category"] != "control"]
        controls = [r for r in records if r["semantic_layer"] == sem and r["category"] == "control"]

        def full_top15(r, w):                                    # all gold in candidate Top15
            return r["by_weight"][w]["recall"][15] == 1.0

        def full_selection(r, w):
            return r["by_weight"][w]["selection_recall"] == 1.0

        weights_out = {}
        for w in _WEIGHTS:
            def m(key, sub, k=None):
                return _mean((r["by_weight"][w][key][k] if k else r["by_weight"][w][key]) for r in metric)
            # gained/lost vs equal-weight baseline
            def flips(rows, pred):
                gained = sorted(r["id"] for r in rows if pred(r, w) and not pred(r, _BASELINE))
                lost = sorted(r["id"] for r in rows if pred(r, _BASELINE) and not pred(r, w))
                return gained, lost
            t15_g, t15_l = flips(metric, full_top15)
            sel_g, sel_l = flips(metric, full_selection)
            ctl_g, ctl_l = flips(controls, full_selection)
            # a gold in dense Top15 that equal-weight demoted out of Top15, restored at this w
            dense_restored = sorted({
                r["id"] for r in metric for g in r["required_tables"]
                if r["dense_ranks"].get(g) is not None and r["dense_ranks"][g] <= 15
                and (r["by_weight"][_BASELINE]["gold_fused_ranks"][g] is None
                     or r["by_weight"][_BASELINE]["gold_fused_ranks"][g] > 15)
                and r["by_weight"][w]["gold_fused_ranks"][g] is not None
                and r["by_weight"][w]["gold_fused_ranks"][g] <= 15})
            # a gold in lexical Top5 that was in fused Top5 at equal weight, lost from Top5 at this w
            lex_top5_lost = sorted({
                r["id"] for r in metric for g in r["required_tables"]
                if r["lex_ranks"].get(g) is not None and r["lex_ranks"][g] <= 5
                and (r["by_weight"][_BASELINE]["gold_fused_ranks"][g] or 99) <= 5
                and (r["by_weight"][w]["gold_fused_ranks"][g] or 99) > 5})
            weights_out[w] = {
                "candidate_recall": {k: m("recall", "recall", k) for k in (5, 10, 15)},
                "precision": {k: m("precision", "precision", k) for k in (5, 10, 15)},
                "selection_recall": m("selection_recall", "selection_recall"),
                "context_recall": m("context_recall", "context_recall"),
                "controls_selection_recall": _mean(r["by_weight"][w]["selection_recall"] for r in controls),
                "vs_equal_weight": {
                    "candidate_top15_gained": t15_g, "candidate_top15_lost": t15_l,
                    "selection_gained": sel_g, "selection_lost": sel_l,
                    "controls_selection_gained": ctl_g, "controls_selection_lost": ctl_l,
                    "dense_restored": dense_restored, "lexical_top5_lost": lex_top5_lost,
                },
            }
        out[mode] = weights_out
    return out


def main() -> None:  # pragma: no cover - real fastembed locally (free; no paid API)
    import tempfile

    from agent.retrieval.value_index import build_value_index
    cfg = RetrievalConfig.dense_value()
    with tempfile.TemporaryDirectory() as workdir:
        db = str(build(Path(workdir) / "saas_expanded.db", confounders=True))
        tables = introspect(db)
        build_value_index(tables, db, FakeValueBackend())       # inert on saas
        records = run_sweep(tables, load_saas_metrics(), MetricRegistry.load(), cfg)
    report = {"kind": "rrf_weight_sweep", "measured": True, "llm": False,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_tables": len(tables), "dense_weight": 1.0, "lexical_weights": list(_WEIGHTS),
              "baseline": _BASELINE, "summary": summarize(records), "records": records}
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"rrf_weight_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    s = report["summary"]
    print(f"wrote {out}")
    for mode in ("off", "on"):
        print(f"\n[{mode}] lexical-weight sweep (dense=1.0):")
        for w in _WEIGHTS:
            m = s[mode][w]
            v = m["vs_equal_weight"]
            tag = " <== current" if w == _BASELINE else ""
            print(f"  w={w:4} R@5/10/15={m['candidate_recall'][5]}/{m['candidate_recall'][10]}/"
                  f"{m['candidate_recall'][15]} P@5={m['precision'][5]} sel={m['selection_recall']} "
                  f"ctx={m['context_recall']} ctl_sel={m['controls_selection_recall']} "
                  f"dense_restored={len(v['dense_restored'])} ctl_lost={len(v['controls_selection_lost'])}{tag}")


if __name__ == "__main__":  # pragma: no cover
    main()
