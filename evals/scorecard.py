"""Two-tier reliability scorecard.

deterministic (default): gate routing report + teeth/contract validations. Zero-API,
zero-Docker. Structurally carries no catch-rate/match-rate field. real-api / all:
explicit opt-in; a missing model hard-fails (SystemExit) rather than emitting a
complete-looking report; the sandbox half also needs local Docker. Real-API scores are
single-run point estimates tagged measured=true, with provenance (golden SHA-256 +
per-case outcomes + model + UTC timestamp) so two runs are diffable.

The deterministic teeth RAISE on failure (never bare ``assert``: a stripped assert under
``python -O`` would let this runner report a fake "teeth passed" count) and the pass
counter increments only after a case fully passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from evalharness.consistency_eval import diverges, execute_case, governance_clean, score_consistency
from evalharness.gate_eval import evaluate_gate
from evalharness.golden import (
    CONSISTENCY_PATH, SANDBOX_PATH, load_consistency, load_gate, load_sandbox,
)
from evalharness.sandbox_eval import score_sandbox, validate_wrong_program

_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _build_saas_db(dir_path: str) -> tuple[str, list]:
    db = str(build(Path(dir_path) / "saas.db"))
    return db, introspect(db)


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deterministic_report() -> dict:
    gate = evaluate_gate(load_gate())

    with tempfile.TemporaryDirectory() as workdir:
        db, tables = _build_saas_db(workdir)

        consistency_passed = 0
        for case in load_consistency():
            cand, gold = execute_case(case, db, tables)
            if not (cand.ok and gold.ok):
                raise RuntimeError(f"{case.id}: candidate or gold SQL failed to execute")
            if not (governance_clean(cand, tables) and governance_clean(gold, tables)):
                raise RuntimeError(f"{case.id}: a result is governance-blocked")
            if diverges(cand, gold) != case.expected_caught:
                raise RuntimeError(f"{case.id}: divergence != expected_caught (fixture has no teeth)")
            consistency_passed += 1        # only after the case fully passes

        sandbox_passed = 0
        for case in load_sandbox():
            if case.wrong_program:
                validate_wrong_program(case)   # raises on failure
                sandbox_passed += 1

    return {
        "gate": {
            "by_construction": True,       # deterministic gates -> 1.0 is a spec check, not a measured accuracy
            "routing_accuracy": gate.routing_accuracy,
            "intent": vars(gate.intent),
            "feasibility": vars(gate.feasibility),
            "n": gate.n,
        },
        "teeth": {"consistency_passed": consistency_passed, "sandbox_passed": sandbox_passed},
    }


def real_api_report(model) -> dict:
    from evals.run_consistency import run_consistency
    from evals.run_sandbox import run_sandbox

    with tempfile.TemporaryDirectory() as workdir:
        db, tables = _build_saas_db(workdir)
        consistency_outcomes = run_consistency(db, tables, load_consistency(), model)
        sandbox_outcomes = run_sandbox(load_sandbox(), model)   # driver measures computation cases only

    return {
        "measured": True,
        "consistency": {
            **score_consistency(consistency_outcomes),
            "outcomes": [vars(o) for o in consistency_outcomes],
        },
        "sandbox": {
            **score_sandbox(sandbox_outcomes),
            "outcomes": [vars(o) for o in sandbox_outcomes],
        },
        "golden_sha256": {
            "consistency": _sha256(CONSISTENCY_PATH),
            "sandbox": _sha256(SANDBOX_PATH),
        },
    }


def run(tier: str, *, model_factory=None) -> dict:
    report = {"tier": tier, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if tier in ("deterministic", "all"):
        report.update(deterministic_report())
    if tier in ("real-api", "all"):
        if model_factory is None:
            print("real-api tier requires a model (set DEEPSEEK_API_KEY); refusing to emit a "
                  "partial score", file=sys.stderr)
            raise SystemExit(1)
        model = model_factory()
        report["model"] = getattr(model, "model_name", getattr(model, "model", "unknown"))
        report["real_api"] = real_api_report(model)
    return report


def _default_model_factory():
    import os
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return None
    from agent.llm import create_sql_model
    return create_sql_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Cadence reliability scorecard")
    parser.add_argument("--tier", choices=["deterministic", "real-api", "all"], default="deterministic")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()  # a local .env is convenient; the real-api tier reads DEEPSEEK_API_KEY from env
    report = run(args.tier, model_factory=_default_model_factory())
    _RESULTS_DIR.mkdir(exist_ok=True)
    out = _RESULTS_DIR / f"scorecard_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
