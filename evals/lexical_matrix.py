"""Deterministic, service-free lexical-backend x RRF-weight selection matrix.

Runs the six frozen cells (hand_weighted|bm25 x lexical weight 0.25|0.5|1.0, dense weight fixed
at 1.0) over two frozen surfaces and applies the locked selection rule. No LLM, no API key, no
Docker, no Elasticsearch: retrieval only, raw golden questions, fixed embedding model.

    .venv/bin/python -m evals.lexical_matrix --spider-dir /path/to/spider_data

Surfaces:
  * explicit_clue -- the 6 control cases of the saas_metrics golden on the 20-table confounder
    fixture, semantic OFF. Governed metric cases are excluded from the OFF gate by policy.
  * spider        -- the frozen 30-case external slice, semantic OFF, schema retrieval only.
  * governed      -- the 24 metric cases with governance ON, to verify the protected-anchor
    invariant. Not a selection criterion.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.metric_match import validate_all_metrics
from agent.retrieval.pipeline import run_retrieval
from agent.retrieval.serde import serialize_config
from agent.semantic_layer import MetricRegistry
from evalharness.golden import load_saas_metrics
from evalharness.lexical_matrix import cell_id, cells, score_case, select, summarize_surface
from evalharness.spider import load_spider_slice, sql_characteristics

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"


def _config(cell: dict) -> RetrievalConfig:
    base = RetrievalConfig.default()
    return RetrievalConfig(**{**base.__dict__, "lexical_backend": cell["lexical_backend"]}) \
        .with_weights(lexical=cell["lexical_weight"], dense=1.0)


def _resolve_gold(gold: list[str], tables) -> list[str]:
    """Map gold table names onto the schema's actual casing.

    ``sql_characteristics`` lowercases table names, but external Spider schemas use mixed case
    (``Pets``, ``Has_Pet``). Recall is a set comparison, so without this the two sides never meet
    and every configuration scores identically -- which is a measurement bug, not a finding.
    Mirrors the case-insensitive comparison the Spider record scorer already performs.
    """
    by_lower = {t.name.lower(): t.name for t in tables}
    return [by_lower.get(name.lower(), name) for name in gold]


def _metric_hits(registry, question: str, tables):
    validate_all_metrics(registry, tables)
    return [h for h in registry.retrieve_matches(question) if h.match_type == "alias"]


def run_cell(cell: dict, *, saas_tables, controls, metric_cases, spider, registry) -> dict:
    cfg = _config(cell)
    explicit = [score_case(run_retrieval(c.question, saas_tables, cfg, k=5), c.required_tables)
                for c in controls]
    governed = [score_case(
        run_retrieval(c.question, saas_tables, cfg, k=5,
                      metric_hits=_metric_hits(registry, c.question, saas_tables)),
        c.required_tables) for c in metric_cases]
    spider_rows = []
    for case, tables, gold in spider:
        spider_rows.append(score_case(run_retrieval(case.question, tables, cfg, k=5), gold))
    return {
        **cell,
        "config": serialize_config(cfg),
        "explicit_clue": summarize_surface(explicit),
        "governed": summarize_surface(governed),
        "spider": summarize_surface(spider_rows),
    }


def _determinism_probe(cell: dict, saas_tables, controls) -> bool:
    cfg = _config(cell)
    def ranks():
        return [[c.table for c in sorted(run_retrieval(x.question, saas_tables, cfg, k=5).candidates,
                                         key=lambda c: c.fusion_rank)] for x in controls]
    return ranks() == ranks()


def build_report(spider_dir: Path) -> dict:
    saas_tables = introspect(build(Path(_tmpdir()) / "confounders.db", confounders=True))
    golden = load_saas_metrics()
    controls = [c for c in golden if c.category == "control"]
    metric_cases = [c for c in golden if c.category != "control"]
    registry = MetricRegistry.load()

    spider_cases = load_spider_slice(spider_dir)
    spider = []
    for case in spider_cases:
        gold, _ = sql_characteristics(case.gold_sql)
        tables = introspect(case.db_path)
        spider.append((case, tables, _resolve_gold(gold, tables)))

    summaries = []
    for cell in cells():
        summary = run_cell(cell, saas_tables=saas_tables, controls=controls,
                           metric_cases=metric_cases, spider=spider, registry=registry)
        summary["deterministic"] = _determinism_probe(cell, saas_tables, controls)
        summary["cell"] = cell_id(cell)
        summaries.append(summary)

    baseline = next(s["explicit_clue"] for s in summaries
                    if s["lexical_backend"] == "hand_weighted" and s["lexical_weight"] == 1.0)
    decision = select(summaries, baseline_explicit_clue=baseline)
    return {
        "kind": "lexical_backend_weight_matrix",
        "measured": True,
        "llm": False,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_internal_tables": len(saas_tables),
        "surfaces": {"explicit_clue": len(controls), "governed": len(metric_cases),
                     "spider": len(spider)},
        "dense_weight": 1.0,
        "decision": decision,
    }


def _tmpdir() -> str:
    import tempfile
    return tempfile.mkdtemp()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Deterministic lexical backend/weight matrix.")
    parser.add_argument("--spider-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = build_report(args.spider_dir)
    output = args.output or _REPORT_DIR / f"lexical_matrix_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for cell in report["decision"]["cells"]:
        print(f"{cell['cell']:>22}  spider R@5={cell['spider']['recall_at_5']}  "
              f"R@15={cell['spider']['recall_at_15']}  ctx={cell['spider']['context_recall']}  "
              f"explicit sel={cell['explicit_clue']['selection_recall']}  "
              f"governed sel={cell['governed']['selection_recall']}  "
              f"eligible={cell['eligible']} {cell['eliminated_for']}")
    print("\nselected:", json.dumps(report["decision"]["selected"]))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
