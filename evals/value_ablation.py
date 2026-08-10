"""Stage 2 value-retrieval ablation driver (deterministic, no LLM / no API).

Runs the value-linking golden through the retrieval pipeline under four configs and records
rank-sensitive metrics per (config, case): lexical / value / dense / dense+value. Uses the
FakeValueBackend (proven parity with the ES backend on match tiers) so the ablation is reproducible
without Docker. This measures RETRIEVAL quality (ordered candidates, Fusion@5, precision,
selection/context) -- exec_match is not part of this ablation."""
from __future__ import annotations

import json
import time
from pathlib import Path

from agent.db.build_value_db import build
from agent.db.introspect import introspect
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.pipeline import run_retrieval
from agent.retrieval.value_backend import FakeValueBackend
from agent.retrieval.value_index import build_value_index
from evalharness.golden import load_value_linking
from evalharness.value_metrics import rank_sensitive_metrics

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"

# The 2x2 factorial over {dense off/on} x {value off/on}, with lexical as the admission floor.
_ABLATION = {
    "lexical": RetrievalConfig.lexical_baseline,     # lexical only
    "value": RetrievalConfig.value_ablation,         # lexical + value
    "dense": RetrievalConfig.rrf_hybrid,             # lexical + dense
    "dense_value": RetrievalConfig.dense_value,      # lexical + dense + value
}


def run_value_ablation(tables, db_path, cases, *, k: int = 5) -> list[dict]:
    value_backend = FakeValueBackend()
    build_value_index(tables, db_path, value_backend)      # one ingestion, reused across configs
    records: list[dict] = []
    for cfg_factory in _ABLATION.values():
        cfg = cfg_factory()
        vb = value_backend if cfg.value_backend == "es" else None
        for case in cases:
            rr = run_retrieval(case.question, tables, cfg, k=k, value_backend=vb)
            records.append({
                "config": cfg.name, "case": case.id, "category": case.category,
                "expect_value_hit": case.expect_value_hit,
                "value_hit": any(s.channel == "value" for s in rr.signals),
                "admission_rejected": any(e.event == "admission_rejected" for e in rr.stage_events),
                "value_degraded": any(e.event == "value_degraded" for e in rr.stage_events),
                **rank_sensitive_metrics(rr, case.required_tables),
            })
    return records


def build_report(tables, db_path, cases, *, k: int = 5) -> dict:
    return {"measured": True, "kind": "value_linking_ablation",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "configs": [f().name for f in _ABLATION.values()],
            "n_tables": len(tables), "retrieval_k": k,
            "records": run_value_ablation(tables, db_path, cases, k=k)}


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as workdir:
        db = build(Path(workdir) / "value.db")
        tables = introspect(db)
        report = build_report(tables, db, load_value_linking())
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"value_ablation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(report['records'])} records)")


if __name__ == "__main__":
    main()
