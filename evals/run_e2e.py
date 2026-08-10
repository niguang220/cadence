"""Manual real-API driver for the end-to-end capability baseline (roadmap Step 1).

Runs every saas_metrics case through the FULL agent, semantic layer ON and OFF, N times
each, and reports the frozen Step-1 shape (traps ON/OFF, controls, first-try->repaired SQL
validity, latency percentiles, avg tokens). This is EXPENSIVE (30 cases x 5 repeats x 2
configs = 300 agent runs) and needs DEEPSEEK_API_KEY -- it is deliberately standalone, not
folded into the cheap default scorecard (see DECISIONS.md #11). Output is measured=True
with provenance (golden SHA-256 + model + UTC timestamp) and lands in docs/reliability/
(evals/results/ is gitignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.serde import serialize_config
from evalharness.e2e_eval import run_case, summarize
from evalharness.golden import SAAS_METRICS_PATH, load_saas_metrics

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"

# One retrieval config per invocation (a clean single-factor ablation). Only Stage-1 presets are
# selectable here -- es/qdrant/llm-selector presets are deliberately out of scope this stage.
_CONFIGS = {
    "lexical_baseline": RetrievalConfig.lexical_baseline,
    "current_hybrid": RetrievalConfig.current_hybrid,
    "rrf_hybrid": RetrievalConfig.rrf_hybrid,
}


def config_by_name(name: str) -> RetrievalConfig:
    try:
        return _CONFIGS[name]()
    except KeyError:
        raise ValueError(f"unknown retrieval config {name!r}; choices: {sorted(_CONFIGS)}")


def report_path(config_name: str, stamp: str, *, report_dir: Path = _REPORT_DIR) -> Path:
    """Config name lives in the filename so the three ablation runs never overwrite or get
    confused with one another. Still matches the docs/reliability/e2e_baseline_*.json exclude glob."""
    return report_dir / f"e2e_baseline_{config_name}_{stamp}.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-API e2e capability baseline; one retrieval config per invocation.")
    p.add_argument("--retrieval-config", choices=sorted(_CONFIGS), default="lexical_baseline",
                   help="Stage-1 retrieval preset to hold fixed for this run.")
    p.add_argument("--repeats", type=int, default=5, help="runs per (case, semantic-layer) pairing.")
    p.add_argument("--k", type=int, default=5, help="retrieval top-k passed to the agent.")
    return p.parse_args(argv)


def run_e2e(db_path, tables, cases, model, *, config: RetrievalConfig, k: int = 5, repeats: int = 5,
            semantic_layers=(False, True)) -> dict:
    records = []
    for repeat_index in range(repeats):
        for semantic_layer in semantic_layers:           # OFF then ON, paired within a repeat
            for case in cases:
                records.append(run_case(db_path, tables, case, model, semantic_layer=semantic_layer,
                                        config=config, k=k, repeat_index=repeat_index))
    return {"records": [asdict(r) for r in records], "summary": summarize(records),
            "repeats": repeats, "semantic_layer_configs": list(semantic_layers),
            "retrieval_config": serialize_config(config), "retrieval_k": k}


def build_report(db_path, tables, cases, model, *, model_name: str, config: RetrievalConfig, k: int = 5,
                 repeats: int = 5) -> dict:
    out = run_e2e(db_path, tables, cases, model, config=config, k=k, repeats=repeats)
    return {"measured": True, "model": model_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "golden_sha256": hashlib.sha256(Path(SAAS_METRICS_PATH).read_bytes()).hexdigest(), **out}


def main(argv=None) -> None:
    args = parse_args(argv)
    import os

    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("run_e2e requires DEEPSEEK_API_KEY (real-API tier); refusing to run",
              file=sys.stderr)
        raise SystemExit(1)
    from agent.llm import create_sql_model
    model = create_sql_model()
    model_name = getattr(model, "model_name", getattr(model, "model", "unknown"))

    config = config_by_name(args.retrieval_config)
    with tempfile.TemporaryDirectory() as workdir:
        db = str(build(Path(workdir) / "saas.db"))
        tables = introspect(db)
        report = build_report(db, tables, load_saas_metrics(), model, model_name=model_name,
                              config=config, k=args.k, repeats=args.repeats)

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = report_path(config.name, time.strftime("%Y%m%d_%H%M%S"))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
