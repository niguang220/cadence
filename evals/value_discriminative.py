"""Stage 3A — deterministic value-linking discriminative eval (no LLM).

Runs the expanded value-linking set through the retrieval pipeline under four configs
(lexical / value / dense / dense+value) on the non-saturated 16-table fixture, and reports
rank-sensitive retrieval quality per category/role: candidate recall, Fusion@5, the value match
tier and the expected table's fused rank. Also runs the negatives/safety cases for zero-leak /
zero-wrong-admission checks. Deterministic (FakeValueBackend, parity-proven with ES); the fixture
is high-cardinality so recall is not saturated. No fabricated numbers."""
from __future__ import annotations

import json
import time
from collections import defaultdict
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
_BUCKET = {"exact_keyword": 4, "exact_phrase": 3, "token_match": 2, "fuzzy": 1}
_HIGH_CONF = {"exact_keyword", "exact_phrase"}
_PII_COLUMNS = {"email", "full_name", "phone"}
_CONFIGS = {
    "lexical": RetrievalConfig.lexical_baseline,
    "value": RetrievalConfig.value_ablation,
    "dense": RetrievalConfig.rrf_hybrid,
    "dense_value": RetrievalConfig.dense_value,
}


def _linking_record(tables, case, cfg, value_backend):
    vb = value_backend if cfg.value_backend == "es" else None
    rr = run_retrieval(case.question, tables, cfg, k=5, value_backend=vb)
    m = rank_sensitive_metrics(rr, case.required_tables)
    vsig = [s for s in rr.signals if s.channel == "value" and s.table == case.expected_table]
    best = max(vsig, key=lambda s: _BUCKET.get(s.match_type, 0), default=None)
    ordered = m["candidate_tables_ordered"]
    return {"config": cfg.name, "case": case.id, "category": case.category, "role": case.role,
            "expected_table": case.expected_table,
            "value_match_type": best.match_type if best else None,
            "expected_table_rank": (ordered.index(case.expected_table) + 1
                                    if case.expected_table in ordered else None),
            "value_degraded": any(e.event == "value_degraded" for e in rr.stage_events),
            "admission_rejected": any(e.event == "admission_rejected" for e in rr.stage_events),
            "candidate_recall": m["candidate_recall"], "fusion_at_5_recall": m["fusion_at_5_recall"],
            "candidate_precision": m["candidate_precision"]}


def _safety_checks(tables, cases, value_backend):
    cfg = RetrievalConfig.value_ablation()
    out = []
    for c in cases:
        if c.role not in ("negative", "safety"):
            continue
        rr = run_retrieval(c.question, tables, cfg, k=5, value_backend=value_backend)
        vsig = [s for s in rr.signals if s.channel == "value"]
        pii = any(s.column in _PII_COLUMNS for s in vsig)
        admitting = any(s.match_type in _HIGH_CONF for s in vsig)
        rec = {"case": c.id, "category": c.category, "role": c.role, "pii_touched": pii,
               "admitting_value_hit": admitting, "value_tables": sorted({s.table for s in vsig}),
               "value_degraded": any(e.event == "value_degraded" for e in rr.stage_events)}
        if c.role == "negative":
            rec["safe"] = not admitting and not pii
        else:                                    # safety: it DOES hit, but never PII, and no degrade
            rec["safe"] = not pii and not rec["value_degraded"]
        out.append(rec)
    return out


def run_value_discriminative(tables, db, cases) -> dict:
    vb = FakeValueBackend()
    build_value_index(tables, db, vb)
    linking = [c for c in cases if c.role in ("primary", "diagnostic")]
    records = [_linking_record(tables, c, f(), vb) for f in _CONFIGS.values() for c in linking]
    return {"records": records, "safety": _safety_checks(tables, cases, vb),
            "configs": [f().name for f in _CONFIGS.values()], "n_tables": len(tables)}


def summarize(result: dict) -> dict:
    recs = result["records"]
    by = {(r["config"], r["case"]): r for r in recs}
    primaries = sorted({r["case"] for r in recs if r["role"] == "primary"})
    cat = {r["case"]: r["category"] for r in recs if r["role"] == "primary"}

    def cr(config, case):
        return by.get((config, case), {}).get("candidate_recall")

    improved = [c for c in primaries
                if cr("value_ablation", c) is not None and cr("lexical_baseline", c) is not None
                and cr("value_ablation", c) > cr("lexical_baseline", c)]
    # per config x category means (primaries only)
    grp = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["role"] == "primary" and r["candidate_recall"] is not None:
            grp[r["config"]][r["category"]].append((r["candidate_recall"], r["fusion_at_5_recall"]))
    per = {cfg: {c: {"candidate_recall": round(sum(x[0] for x in v) / len(v), 3),
                     "fusion_at_5": round(sum(x[1] for x in v) / len(v), 3), "n": len(v)}
                 for c, v in cats.items()} for cfg, cats in grp.items()}
    safety = result["safety"]
    return {
        "n_primary": len(primaries),
        "value_improves": {"cases": improved, "n_cases": len(improved),
                           "categories": sorted({cat[c] for c in improved}),
                           "n_categories": len({cat[c] for c in improved})},
        "paired_value_vs_lexical": {c: round(cr("value_ablation", c) - cr("lexical_baseline", c), 3)
                                    for c in primaries
                                    if cr("value_ablation", c) is not None
                                    and cr("lexical_baseline", c) is not None},
        "paired_dense_value_vs_dense": {c: round(cr("dense_value", c) - cr("rrf_hybrid", c), 3)
                                        for c in primaries
                                        if cr("dense_value", c) is not None
                                        and cr("rrf_hybrid", c) is not None},
        "per_config_category": per,
        "safety": {"all_negatives_safe": all(s["safe"] for s in safety if s["role"] == "negative"),
                   "all_safety_safe": all(s["safe"] for s in safety if s["role"] == "safety"),
                   "any_pii_leak": any(s["pii_touched"] for s in safety),
                   "any_value_degraded": any(s["value_degraded"] for s in safety)},
    }


def main() -> None:  # pragma: no cover
    import tempfile
    with tempfile.TemporaryDirectory() as workdir:
        db = build(Path(workdir) / "value.db")
        tables = introspect(db)
        result = run_value_discriminative(tables, db, load_value_linking())
    report = {"kind": "value_discriminative", "measured": True,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "summary": summarize(result), **result}
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"value_discriminative_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
