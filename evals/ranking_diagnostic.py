"""Stage 3C follow-up — retrieval-only, LLM-free ranking diagnostic on a NON-saturated schema.

Runs the existing dense_value RRF config over the saas_metrics golden on the confounder-expanded
20-table fixture (candidate_k=15, context_anchor_k=5 unchanged) and records, per case, whether each
gold table enters the fused Top-k and, if so, whether the Top-5 selector drops it. Deterministic: raw
golden question (no query_enhance / no LLM), fastembed dense (fixed model), inert value channel
(saas has no searchable columns). No paid API; no production preset changes; Top6 is an OFFLINE probe.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.pipeline import run_retrieval
from agent.retrieval.value_backend import FakeValueBackend
from evalharness.golden import load_saas_metrics
from evalharness.ranking_diagnostic import ranking_diagnostic

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"


def run_ranking_diagnostic(tables, db, cases, value_backend, config: RetrievalConfig, *, k: int = 5):
    vb = value_backend if config.value_backend == "es" else None
    out = []
    for case in cases:
        rr = run_retrieval(case.question, tables, config, k=k, value_backend=vb)     # RAW question, no LLM
        out.append({"id": case.id, "category": case.category,
                    "required_tables": list(case.required_tables),
                    **ranking_diagnostic(rr, case.required_tables)})
    return out


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(records: list[dict]) -> dict:
    metric = [r for r in records if r["category"] != "control"]
    controls = [r for r in records if r["category"] == "control"]

    def block(rows):
        return {"n": len(rows),
                **{f"recall_at_{k}": _mean(r["recall_at"][k] for r in rows) for k in (5, 6, 10, 15)},
                **{f"precision_at_{k}": _mean(r["precision_at"][k] for r in rows) for k in (5, 10, 15)},
                "mean_selection_recall": _mean(r["selection_recall"] for r in rows),
                "mean_context_recall": _mean(r["context_recall"] for r in rows)}

    # ARPU-specific: is `plan` (a gold endpoint) in Top15, and is it only the Top-5 cutoff that loses it?
    arpu = []
    for r in (r for r in records if r["category"] == "arpu"):
        prank = r["gold_fusion_ranks"].get("plan")
        arpu.append({
            "id": r["id"], "plan_fusion_rank": prank,
            "plan_in_top15": prank is not None and prank <= 15,
            "plan_in_top5": prank is not None and prank <= 5,
            "plan_in_top6": prank is not None and prank <= 6,
            "plan_dropped_by_selector": "plan" in r["gold_dropped_by_selector"],
            "plan_role": ("endpoint" if "plan" in r["required_tables"]
                          else "bridge" if "plan" in r["bridges_added"] else "absent"),
            "context_recall": r["context_recall"], "selection_recall": r["selection_recall"],
        })

    ctx_counts = [r["context_table_count"] for r in records]
    dropped = {r["id"]: r["gold_dropped_by_selector"] for r in records if r["gold_dropped_by_selector"]}
    bridges = {r["id"]: r["bridges_added"] for r in records if r["bridges_added"]}
    return {
        "n_cases": len(records),
        "metric_cases": block(metric),
        "controls": block(controls),
        "gold_dropped_by_selector": dropped,
        "bridges_added": bridges,
        "arpu": arpu,
        "context_tables": {"avg": round(sum(ctx_counts) / len(ctx_counts), 2), "max": max(ctx_counts)},
    }


def main() -> None:  # pragma: no cover - runs real fastembed locally (free; no paid API)
    import tempfile

    from agent.retrieval.value_index import build_value_index
    with tempfile.TemporaryDirectory() as workdir:
        db = str(build(Path(workdir) / "saas_expanded.db", confounders=True))
        tables = introspect(db)
        backend = FakeValueBackend()
        build_value_index(tables, db, backend)                  # inert on saas (no searchable columns)
        records = run_ranking_diagnostic(tables, db, load_saas_metrics(), backend,
                                         RetrievalConfig.dense_value())
    report = {"kind": "ranking_diagnostic", "measured": True, "llm": False,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_tables": len(tables), "config": "dense_value",
              "summary": summarize(records), "records": records}
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"ranking_diagnostic_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    s = report["summary"]
    print(f"wrote {out} ({report['n_tables']} tables, {report['n_cases'] if 'n_cases' in report else s['n_cases']} cases)")
    print("metric cases:", {k: s["metric_cases"][k] for k in
                            ("recall_at_5", "recall_at_6", "recall_at_10", "recall_at_15",
                             "mean_selection_recall", "mean_context_recall")})
    print("ARPU:", s["arpu"])


if __name__ == "__main__":  # pragma: no cover
    main()
