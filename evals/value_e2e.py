"""Stage 2 full-agent value E2E driver.

Runs the value-linking POSITIVES through the FULL agent (real ES value backend + real model) under
four retrieval configs, and scores exec_match against each case's gold_sql. The three NEGATIVES get
a deterministic safety acceptance (no LLM): they must never produce an admitting value hit or touch
a PII column. semantic_layer=False and clarify=False isolate the value-retrieval variable; the ES
index is built ONCE before the round and reused; all configs share one concurrency.

Official numbers require a REAL Elasticsearch backend (do NOT use the fake for official runs) and
DEEPSEEK_API_KEY. The record layer stores tables / tiers / ranks / booleans only — never raw values.
"""
from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path

from agent.db.build_value_db import build
from agent.db.introspect import introspect
from agent.graph import run_agent
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.pipeline import run_retrieval
from agent.retrieval.value_index import build_value_index
from evalharness.golden import load_value_linking
from evalharness.oracle import execution_match
from evalharness.value_metrics import rank_sensitive_metrics
from agent.execution import run_query

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"
_MAX_CONCURRENCY = 16

_CONFIGS = {
    "lexical": RetrievalConfig.lexical_baseline,
    "value": RetrievalConfig.value_ablation,
    "dense": RetrievalConfig.rrf_hybrid,
    "dense_value": RetrievalConfig.dense_value,
}
_HIGH_CONF = {"exact_keyword", "exact_phrase"}
_PII_COLUMNS = {"email", "full_name", "phone"}


def _positive_record(db, tables, case, model, cfg, value_backend, repeat):
    vb = value_backend if cfg.value_backend == "es" else None
    gold = run_query(db, case.gold_sql, tables=tables)
    t0 = time.perf_counter()
    r = run_agent(db, case.question, model=model, tables=tables, semantic_layer=False,
                  clarify=False, retrieval_config=cfg, value_backend=vb)
    agent_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()                                 # ES query cost, measured on its own
    run_retrieval(case.question, tables, cfg, k=5, value_backend=vb)
    retrieval_ms = (time.perf_counter() - t1) * 1000
    ok = r.execution.ok
    match = execution_match(r.execution.rows, gold.rows, ordered=False) if ok else False
    events = r.retrieval_result.stage_events if r.retrieval_result else []
    usage = r.usage or {}
    rec = {"config": cfg.name, "case": case.id, "category": case.category, "repeat": repeat,
           "exec_match": match, "no_sql": not r.sql, "answer_mismatch": ok and not match,
           "prompt_tokens": int(usage.get("input_tokens", 0)),
           "completion_tokens": int(usage.get("output_tokens", 0)),
           "agent_latency_ms": agent_ms, "retrieval_latency_ms": retrieval_ms,
           "value_degraded": any(e.event == "value_degraded" for e in events),
           "admission_rejected": any(e.event == "admission_rejected" for e in events)}
    rr = r.retrieval_result
    rec.update(rank_sensitive_metrics(rr, case.required_tables) if rr else {})
    return rec


def negative_safety(tables, negatives, value_backend, *, config=None) -> list[dict]:
    """Deterministic safety acceptance (no LLM): a negative must never yield an admitting value hit
    or a PII column, under the value config."""
    cfg = config or RetrievalConfig.value_ablation()
    out = []
    for case in negatives:
        rr = run_retrieval(case.question, tables, cfg, k=5, value_backend=value_backend)
        value_sigs = [s for s in rr.signals if s.channel == "value"]
        out.append({"case": case.id, "category": case.category,
                    "admitting_value_hit": any(s.match_type in _HIGH_CONF for s in value_sigs),
                    "pii_touched": any(s.column in _PII_COLUMNS for s in value_sigs),
                    "safe": not any(s.match_type in _HIGH_CONF for s in value_sigs)
                            and not any(s.column in _PII_COLUMNS for s in value_sigs)})
    return out


def run_value_e2e(db, tables, positives, negatives, model, value_backend, *,
                  repeats: int = 5, concurrency: int = 4) -> dict:
    build_value_index(tables, db, value_backend)             # ONE ingestion, reused across the round
    jobs = [(cfg_factory(), case, rep)
            for cfg_factory in _CONFIGS.values()
            for rep in range(repeats)
            for case in positives]

    def work(job):
        cfg, case, rep = job
        return _positive_record(db, tables, case, model, cfg, value_backend, rep)

    workers = max(1, min(concurrency, _MAX_CONCURRENCY))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_i = {ex.submit(work, j): i for i, j in enumerate(jobs)}
        records = [None] * len(jobs)
        for fut, i in fut_to_i.items():
            records[i] = fut.result()                        # propagate worker errors, no silent drop
    return {"positive_records": records,
            "negative_safety": negative_safety(tables, negatives, value_backend),
            "configs": [f().name for f in _CONFIGS.values()], "repeats": repeats,
            "concurrency": workers, "n_positive_runs": len(records)}


def build_report(db, tables, cases, model, value_backend, *, model_name: str,
                 repeats: int = 5, concurrency: int = 4) -> dict:
    positives = [c for c in cases if c.role == "primary"]     # gold-scored (diagnostic/safety excluded)
    negatives = [c for c in cases if c.role == "negative"]     # deterministic safety acceptance
    out = run_value_e2e(db, tables, positives, negatives, model, value_backend,
                        repeats=repeats, concurrency=concurrency)
    return {"measured": True, "kind": "value_full_agent_e2e", "model": model_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_tables": len(tables), **out}


def main() -> None:  # pragma: no cover - requires real ES + DEEPSEEK_API_KEY
    import os
    import tempfile

    from dotenv import load_dotenv
    load_dotenv()
    es_url = os.environ.get("CADENCE_ES_URL")
    if not es_url:
        raise SystemExit("value_e2e needs CADENCE_ES_URL (a REAL Elasticsearch; the fake is not for "
                         "official numbers)")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("value_e2e needs DEEPSEEK_API_KEY (real-API tier)")
    from agent.llm import create_sql_model
    from agent.retrieval.value_backend import ElasticsearchValueBackend
    from agent.retrieval.value_index import index_name

    model = create_sql_model()
    model_name = getattr(model, "model_name", getattr(model, "model", "unknown"))
    with tempfile.TemporaryDirectory() as workdir:
        db = build(Path(workdir) / "value.db")
        tables = introspect(db)
        backend = ElasticsearchValueBackend.from_url(es_url, index_name(tables))
        report = build_report(db, tables, load_value_linking(), model, backend,
                              model_name=model_name)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"value_e2e_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out} ({report['n_positive_runs']} positive runs)")


if __name__ == "__main__":  # pragma: no cover
    main()
