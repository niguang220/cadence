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

import hashlib
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from evalharness.e2e_eval import run_case, summarize
from evalharness.golden import SAAS_METRICS_PATH, load_saas_metrics

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"


def run_e2e(db_path, tables, cases, model, *, repeats: int = 5,
            configs=(False, True)) -> dict:
    records = []
    for semantic_layer in configs:
        for _ in range(repeats):
            for case in cases:
                records.append(run_case(db_path, tables, case, model,
                                        semantic_layer=semantic_layer))
    return {"records": [asdict(r) for r in records], "summary": summarize(records),
            "repeats": repeats, "configs": list(configs)}


def build_report(db_path, tables, cases, model, *, model_name: str,
                 repeats: int = 5) -> dict:
    out = run_e2e(db_path, tables, cases, model, repeats=repeats)
    return {
        "measured": True,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "golden_sha256": hashlib.sha256(Path(SAAS_METRICS_PATH).read_bytes()).hexdigest(),
        **out,
    }


def main() -> None:
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

    with tempfile.TemporaryDirectory() as workdir:
        db = str(build(Path(workdir) / "saas.db"))
        tables = introspect(db)
        report = build_report(db, tables, load_saas_metrics(), model,
                              model_name=model_name, repeats=5)

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"e2e_baseline_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
