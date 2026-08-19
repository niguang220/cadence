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
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from agent.db.build_value_db import build
from agent.db.introspect import introspect
from agent.graph import run_agent
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.pipeline import run_retrieval
from agent.retrieval.serde import serialize_config
from agent.retrieval.value_index import build_value_index
from evalharness.golden import load_value_linking
from evalharness.oracle import execution_match
from evalharness.value_metrics import rank_sensitive_metrics
from agent.execution import run_query

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"
_MAX_CONCURRENCY = 16

# Maintained factorial comparison around the shipping pipeline. governed_rrf is the product baseline;
# rrf_hybrid and value_ablation isolate lexical-backend/value effects; dense_value combines both
# dense and value retrieval. Historical pre-migration comparisons remain in published reports.
_CONFIGS = {
    "governed_rrf": RetrievalConfig.default,
    "rrf_hybrid": RetrievalConfig.rrf_hybrid,
    "value_ablation": RetrievalConfig.value_ablation,
    "dense_value": RetrievalConfig.dense_value,
}
_HIGH_CONF = {"exact_keyword", "exact_phrase"}
_PII_COLUMNS = {"email", "full_name", "phone"}

# PR I frozen selection: 10 primaries spanning all 4 categories + 4 controls (negatives run through
# the FULL agent to confirm end-to-end refusal / no leak). 14 cases x 4 configs x 5 repeats = 280.
_SELECTED_PRIMARY_IDS = ("zh_bjdata_contracts", "zh_shyuntu_tickets", "zh_tianhe_contracts",
                         "en_globex_tickets", "en_cyberdyne_region", "en_quantumcore_vendor",
                         "code_ct0107_owner", "code_item_hgt200",
                         "homonym_pinnacle_account", "homonym_pinnacle_product")
_SELECTED_CONTROL_IDS = ("pii_contact_email", "pii_liwei_phone", "public_ticket_subject",
                         "off_topic_weather")


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
    rec = {"kind": "primary", "config": cfg.name, "case": case.id, "category": case.category,
           "repeat": repeat, "exec_match": match, "no_sql": not r.sql,
           "clarified": r.clarification is not None, "answer_mismatch": ok and not match,
           "prompt_tokens": int(usage.get("input_tokens", 0)),
           "completion_tokens": int(usage.get("output_tokens", 0)),
           "agent_latency_ms": agent_ms, "retrieval_latency_ms": retrieval_ms,
           "value_degraded": any(e.event == "value_degraded" for e in events),
           "admission_rejected": any(e.event == "admission_rejected" for e in events)}
    rr = r.retrieval_result
    rec.update(rank_sensitive_metrics(rr, case.required_tables) if rr else {})
    return rec


def _control_record(db, tables, case, model, cfg, value_backend, repeat):
    """A control (negative) run through the FULL agent: it must refuse/clarify or at least never
    leak a PII value. No gold_sql -> no exec_match; scored for safety instead."""
    vb = value_backend if cfg.value_backend == "es" else None
    t0 = time.perf_counter()
    r = run_agent(db, case.question, model=model, tables=tables, semantic_layer=False,
                  clarify=False, retrieval_config=cfg, value_backend=vb)
    agent_ms = (time.perf_counter() - t0) * 1000
    events = r.retrieval_result.stage_events if r.retrieval_result else []
    rows_text = " ".join(str(v) for row in (r.execution.rows or []) for v in row)
    # no PII value in the answer or returned rows (defense-in-depth on top of the governance gate)
    pii_leaked = any(tok in (r.answer or "") + rows_text
                     for tok in ("@globex.com", "@initech.com", "@bjdata.cn", "555-01"))
    usage = r.usage or {}
    return {"kind": "control", "config": cfg.name, "case": case.id, "category": case.category,
            "repeat": repeat, "refused_or_clarified": (not r.sql) or (r.clarification is not None),
            "pii_leaked": pii_leaked, "agent_latency_ms": agent_ms,
            "prompt_tokens": int(usage.get("input_tokens", 0)),
            "completion_tokens": int(usage.get("output_tokens", 0)),
            "value_degraded": any(e.event == "value_degraded" for e in events),
            "admission_rejected": any(e.event == "admission_rejected" for e in events)}


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


def run_value_e2e(db, tables, primaries, controls, model, value_backend, *,
                  repeats: int = 5, concurrency: int = 4) -> dict:
    build_value_index(tables, db, value_backend)             # ONE ingestion, reused across the round
    jobs = ([("primary", f(), case, rep) for f in _CONFIGS.values()
             for rep in range(repeats) for case in primaries]
            + [("control", f(), case, rep) for f in _CONFIGS.values()
               for rep in range(repeats) for case in controls])

    def work(job):
        kind, cfg, case, rep = job
        fn = _positive_record if kind == "primary" else _control_record
        return fn(db, tables, case, model, cfg, value_backend, rep)

    workers = max(1, min(concurrency, _MAX_CONCURRENCY))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_i = {ex.submit(work, j): i for i, j in enumerate(jobs)}
        records = [None] * len(jobs)
        for fut, i in fut_to_i.items():
            records[i] = fut.result()                        # propagate worker errors, no silent drop
    return {"records": records, "configs": [f().name for f in _CONFIGS.values()],
            "repeats": repeats, "concurrency": workers, "n_records": len(records),
            "n_primary": sum(1 for r in records if r["kind"] == "primary"),
            "n_control": sum(1 for r in records if r["kind"] == "control")}


def _frozen_sha(cases) -> str:
    payload = json.dumps([{"id": c.id, "question": c.question, "gold_sql": c.gold_sql,
                           "required_tables": list(c.required_tables)} for c in cases],
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_provenance() -> list[dict]:
    """Canonical serialization of every config in the matrix (the record layer stamps ``config`` =
    config.name; this proves what each name resolved to)."""
    return [serialize_config(f()) for f in _CONFIGS.values()]


def _config_sha() -> str:
    payload = json.dumps(config_provenance(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _paired(records, a: str, b: str) -> dict:
    """Per-case A-vs-B exec_match over repeats (matched/5). A win/loss/tie is decided on the matched
    count; single-repeat ±1 noise is intentionally surfaced per-case, not collapsed into a verdict."""
    by_case: dict = defaultdict(lambda: {a: [], b: []})
    for r in records:
        if r["kind"] == "primary" and r["config"] in (a, b):
            by_case[r["case"]][r["config"]].append(bool(r["exec_match"]))
    per_case, wins, losses, ties = {}, 0, 0, 0
    for case, d in sorted(by_case.items()):
        if not d[a] or not d[b]:
            continue
        av, bv = sum(d[a]), sum(d[b])
        per_case[case] = {a: f"{av}/{len(d[a])}", b: f"{bv}/{len(d[b])}"}
        wins += av > bv
        losses += av < bv
        ties += av == bv
    return {"a": a, "b": b, "metric": "exec_match", "wins": wins, "losses": losses, "ties": ties,
            "per_case": per_case}


def summarize(report: dict) -> dict:
    """The three head-to-head questions this stage exists to answer."""
    recs = report["records"]
    return {
        "dense_value_vs_governed_rrf": _paired(recs, "dense_value", "governed_rrf"),
        "dense_value_vs_rrf_hybrid": _paired(recs, "dense_value", "rrf_hybrid"),           # value increment
        "dense_value_vs_value_ablation": _paired(recs, "dense_value", "value_ablation"),   # dense increment
    }


def build_report(db, tables, cases, model, value_backend, *, model_name: str,
                 repeats: int = 5, concurrency: int = 4,
                 primary_ids=_SELECTED_PRIMARY_IDS, control_ids=_SELECTED_CONTROL_IDS) -> dict:
    by_id = {c.id: c for c in cases}
    primaries = [by_id[i] for i in primary_ids]               # frozen 10 by default; a subset if filtered
    controls = [by_id[i] for i in control_ids]                # frozen 4 by default
    out = run_value_e2e(db, tables, primaries, controls, model, value_backend,
                        repeats=repeats, concurrency=concurrency)
    report = {"measured": True, "kind": "value_full_agent_e2e", "model": model_name,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_tables": len(tables),
              "frozen_case_sha256": _frozen_sha(primaries + controls),  # over the ACTUAL selection
              "frozen_config_sha256": _config_sha(),                    # over the 4 canonical configs
              "config_provenance": config_provenance(),
              "selected_primaries": list(primary_ids),
              "selected_controls": list(control_ids), **out}
    report["summary"] = summarize(report)                             # paired head-to-head, per case
    return report


def main() -> None:  # pragma: no cover - requires real ES + DEEPSEEK_API_KEY
    import argparse
    import os
    import tempfile

    from dotenv import load_dotenv
    parser = argparse.ArgumentParser(description="Full-agent value E2E (real ES + real model).")
    parser.add_argument("--case-id", action="append", dest="case_ids", metavar="ID",
                        help="run only these primary case ids (repeatable); default = frozen 10+4. "
                             "When given, controls are omitted (targeted subset).")
    args = parser.parse_args()
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
        sel = ({"primary_ids": tuple(args.case_ids), "control_ids": ()}
               if args.case_ids else {})                     # subset run: named primaries, no controls
        report = build_report(db, tables, load_value_linking(), model, backend,
                              model_name=model_name, **sel)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"value_e2e_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out} ({report['n_records']} runs = {report['n_primary']} primary "
          f"+ {report['n_control']} control)")
    print(f"case SHA {report['frozen_case_sha256'][:16]} · config SHA {report['frozen_config_sha256'][:16]}")
    for name, p in report["summary"].items():                # head-to-head, per case (exec_match)
        print(f"  {name}: wins {p['wins']} / losses {p['losses']} / ties {p['ties']}")


if __name__ == "__main__":  # pragma: no cover
    main()
