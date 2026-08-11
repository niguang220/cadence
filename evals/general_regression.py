"""Stage 3C — general-mix regression: production default vs the value candidate on a NON-value-biased set.

Runs the FULL agent over the saas_metrics 30-case golden (24 traps + 6 controls) under two configs
(current_hybrid = shipping default; dense_value = the existing RRF + shortest_path + value candidate)
and both semantic-layer modes, 5 repeats each -> 600 records, one shared batch/concurrency. It answers
ONE question: does adopting dense_value make GENERAL queries regress?

saas_metrics has no searchable columns, so the value channel is inert BY CONSTRUCTION here — that is
the point: this isolates the candidate/relation machinery (RRF + shortest_path vs legacy_minmax +
one_hop) on general queries, and records value-hit counts to prove value never fired. Official numbers
need real ES 8.19 + real DeepSeek (see main); the driver/summary are unit-tested with fakes.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from agent.db.build_saas_db import build
from agent.db.introspect import introspect
from agent.retrieval.contracts import RetrievalConfig
from agent.retrieval.serde import serialize_config
from agent.retrieval.value_index import build_value_index
from evalharness.e2e_eval import EvalRecord, _percentile, run_case
from evalharness.golden import SAAS_METRICS_PATH, load_saas_metrics

_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reliability"
_MAX_CONCURRENCY = 16

# Production default vs the existing candidate. No new fusion, no current_hybrid+value, no name routing.
_CONFIGS = {"current_hybrid": RetrievalConfig.current_hybrid, "dense_value": RetrievalConfig.dense_value}
_SEMANTIC_MODES = (False, True)
_CLEAN_LOSS_MARGIN = 3                 # dense_value <= current_hybrid - 3 (of 5) is a clean loss


# --- run -----------------------------------------------------------------------------------------

def run_general_regression(db_path, tables, cases, model, value_backend, *,
                           repeats: int = 5, concurrency: int = 4) -> list[EvalRecord]:
    build_value_index(tables, db_path, value_backend)          # real ingestion (0 docs on saas; real ES path)
    # Deterministic order: config -> semantic (OFF, ON) -> repeat -> case; reassembled to match serial.
    plan = [(cfg_name, sem, rep, case)
            for cfg_name in _CONFIGS
            for sem in _SEMANTIC_MODES
            for rep in range(repeats)
            for case in cases]

    def work(item):
        cfg_name, sem, rep, case = item
        cfg = _CONFIGS[cfg_name]()
        vb = value_backend if cfg.value_backend == "es" else None
        return run_case(db_path, tables, case, model, semantic_layer=sem, config=cfg, k=5,
                        repeat_index=rep, value_backend=vb)

    workers = max(1, min(concurrency, _MAX_CONCURRENCY))
    if workers == 1:
        return [work(item) for item in plan]
    records: list = [None] * len(plan)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_i = {ex.submit(work, item): i for i, item in enumerate(plan)}
        for fut, i in fut_to_i.items():
            records[i] = fut.result()                          # propagate worker errors, no silent drop
    return records


# --- summary -------------------------------------------------------------------------------------

def _matched(records, config, sem, cid):
    rr = [r for r in records if r.retrieval_config["name"] == config and r.semantic_layer == sem
          and r.id == cid]
    return sum(r.exec_match for r in rr), len(rr)


def _paired(records, case_ids, sem) -> dict:
    """dense_value vs current_hybrid per case (matched/repeats) for one semantic mode."""
    per_case, wins, losses, ties, clean_loss, regressions = {}, 0, 0, 0, [], []
    for cid in case_ids:
        ch, nch = _matched(records, "current_hybrid", sem, cid)
        dv, ndv = _matched(records, "dense_value", sem, cid)
        if nch == 0 or ndv == 0:
            continue
        per_case[cid] = {"current_hybrid": f"{ch}/{nch}", "dense_value": f"{dv}/{ndv}"}
        wins += dv > ch
        losses += dv < ch
        ties += dv == ch
        if dv <= ch - _CLEAN_LOSS_MARGIN:
            clean_loss.append(cid)
        if dv < ch:
            regressions.append(cid)
    return {"wins": wins, "losses": losses, "ties": ties, "clean_loss_ids": sorted(clean_loss),
            "regression_ids": sorted(regressions),
            "per_case_losses": {c: v for c, v in per_case.items()
                                if int(v["dense_value"].split("/")[0]) < int(v["current_hybrid"].split("/")[0])}}


def _lat_tok(records, config) -> dict:
    """Latency (p50/p95) and mean tokens under three denominators: all runs / generation-reached /
    answered. current_hybrid refuses fast, so all-runs p50 misstates cost — the other two are the
    apples-to-apples views."""
    rr = [r for r in records if r.retrieval_config["name"] == config]
    buckets = {
        "all": rr,
        "generation_reached": [r for r in rr if r.failure_stage not in ("no_sql", "clarified")],
        "answered": [r for r in rr if r.sql_valid_final],
    }
    out = {}
    for name, rows in buckets.items():
        lat = [r.latency_ms for r in rows]
        out[name] = {"n": len(rows),
                     "latency_p50": _percentile(lat, 50), "latency_p95": _percentile(lat, 95),
                     "avg_prompt_tokens": (sum(r.prompt_tokens for r in rows) / len(rows)) if rows else 0.0,
                     "avg_completion_tokens": (sum(r.completion_tokens for r in rows) / len(rows)) if rows else 0.0}
    return out


def _recall(records, config, sem) -> dict:
    rr = [r for r in records if r.retrieval_config["name"] == config and r.semantic_layer == sem
          and r.candidate_recall is not None]
    def mean(key):
        vals = [getattr(r, key) for r in rr]
        return round(sum(vals) / len(vals), 3) if vals else None
    return {"candidate_recall": mean("candidate_recall"), "selection_recall": mean("selection_recall"),
            "context_recall": mean("context_recall"), "n": len(rr)}


def _events(records) -> dict:
    by = Counter()
    for r in records:
        for e in r.retrieval_stage_events:
            by[(e.get("stage"), e.get("event"))] += 1
    return {
        "value_degraded": sum(v for (s, ev), v in by.items() if ev == "value_degraded"),
        "admission_rejected": sum(v for (s, ev), v in by.items() if ev == "admission_rejected"),
        "dense_degraded": sum(v for (s, ev), v in by.items() if ev == "dense_degraded"),
        "relation_unconnected": sum(v for (s, ev), v in by.items() if ev == "unconnected_anchor"),
        # not a discrete event: a selector that dropped a GOLD table (selection_recall < candidate_recall)
        "selector_dropped_gold": sum(1 for r in records if r.candidate_recall is not None
                                     and r.selection_recall < r.candidate_recall),
        "by_stage_event": {f"{s}:{ev}": v for (s, ev), v in sorted(by.items())},
    }


def summarize(records: list[EvalRecord]) -> dict:
    trap_ids = sorted({r.id for r in records if r.category != "control"})
    control_ids = sorted({r.id for r in records if r.category == "control"})

    def rate(config, sem, ids):
        rr = [r for r in records if r.retrieval_config["name"] == config and r.semantic_layer == sem
              and r.id in ids]
        return {"matched": sum(r.exec_match for r in rr), "n": len(rr),
                "rate": round(sum(r.exec_match for r in rr) / len(rr), 3) if rr else None}

    def refusal(config, sem, ids):
        rr = [r for r in records if r.retrieval_config["name"] == config and r.semantic_layer == sem
              and r.id in ids]
        return sum(1 for r in rr if r.failure_stage == "no_sql")

    by_mode = {}
    for sem, label in ((False, "off"), (True, "on")):
        by_mode[label] = {
            "traps": {"current_hybrid": rate("current_hybrid", sem, trap_ids),
                      "dense_value": rate("dense_value", sem, trap_ids),
                      "paired": _paired(records, trap_ids, sem)},
            "controls": {"current_hybrid": rate("current_hybrid", sem, control_ids),
                         "dense_value": rate("dense_value", sem, control_ids),
                         "paired": _paired(records, control_ids, sem)},
        }
        # control diverging_ids: controls where dense_value regresses vs current_hybrid in this mode
        by_mode[label]["controls"]["diverging_ids"] = by_mode[label]["controls"]["paired"]["regression_ids"]

    value_hit_records = [r for r in records if r.value_hit]
    return {
        "n_records": len(records),
        "domain_note": ("saas_metrics has no searchable columns; the value channel is inert by "
                        "construction, so this isolates RRF+shortest_path vs legacy_minmax+one_hop on "
                        "general queries"),
        "value_hits": {"records_with_value_hit": len(value_hit_records),
                       "by_tier": dict(Counter(r.value_hit_tier for r in value_hit_records))},
        "by_mode": by_mode,
        "refusal_no_sql": {c: {"off": refusal(c, False, trap_ids + control_ids),
                               "on": refusal(c, True, trap_ids + control_ids)} for c in _CONFIGS},
        "retrieval": {c: {"off": _recall(records, c, False), "on": _recall(records, c, True)}
                      for c in _CONFIGS},
        "latency_tokens": {c: _lat_tok(records, c) for c in _CONFIGS},
        "events": {c: _events([r for r in records if r.retrieval_config["name"] == c]) for c in _CONFIGS},
    }


# --- report --------------------------------------------------------------------------------------

def _compact(rec: EvalRecord) -> dict:
    """Metrics-only projection for the artifact — no predicted_sql / gold_sql / question, so the file
    stays a metrics layer (the saas domain carries no entity values, but this keeps the redaction
    discipline and the file small)."""
    return {"id": rec.id, "category": rec.category, "config": rec.retrieval_config["name"],
            "semantic_layer": rec.semantic_layer, "repeat_index": rec.repeat_index,
            "exec_match": rec.exec_match, "failure_stage": rec.failure_stage,
            "sql_valid_first_try": rec.sql_valid_first_try, "sql_valid_final": rec.sql_valid_final,
            "repair_attempts": rec.repair_attempts, "latency_ms": rec.latency_ms,
            "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
            "candidate_recall": rec.candidate_recall, "selection_recall": rec.selection_recall,
            "context_recall": rec.context_recall, "value_hit": rec.value_hit,
            "value_hit_tier": rec.value_hit_tier, "gold_tables": rec.gold_tables,
            "retrieved_tables": rec.retrieved_tables, "retrieval_stage_events": rec.retrieval_stage_events}


def config_provenance() -> list[dict]:
    return [serialize_config(f()) for f in _CONFIGS.values()]


def _config_sha() -> str:
    return hashlib.sha256(json.dumps(config_provenance(), sort_keys=True).encode("utf-8")).hexdigest()


def build_report(db_path, tables, cases, model, value_backend, *, model_name: str,
                 repeats: int = 5, concurrency: int = 4) -> dict:
    records = run_general_regression(db_path, tables, cases, model, value_backend,
                                     repeats=repeats, concurrency=concurrency)
    return {
        "measured": True, "kind": "general_mix_regression", "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_tables": len(tables), "repeats": repeats,
        "golden_sha256": hashlib.sha256(Path(SAAS_METRICS_PATH).read_bytes()).hexdigest(),
        "frozen_config_sha256": _config_sha(),
        "config_provenance": config_provenance(),
        "configs": list(_CONFIGS), "semantic_modes": ["off", "on"],
        "n_records": len(records),
        "summary": summarize(records),
        "records": [_compact(r) for r in records],
    }


def main() -> None:  # pragma: no cover - requires real ES + DEEPSEEK_API_KEY
    import os
    import tempfile

    from dotenv import load_dotenv
    load_dotenv()
    es_url = os.environ.get("CADENCE_ES_URL")
    if not es_url:
        raise SystemExit("general_regression needs CADENCE_ES_URL (a REAL Elasticsearch)")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("general_regression needs DEEPSEEK_API_KEY (real-API tier)")
    from agent.llm import create_sql_model
    from agent.retrieval.value_backend import ElasticsearchValueBackend
    from agent.retrieval.value_index import index_name

    model = create_sql_model()
    model_name = getattr(model, "model_name", getattr(model, "model", "unknown"))
    with tempfile.TemporaryDirectory() as workdir:
        db = str(build(Path(workdir) / "saas.db"))
        tables = introspect(db)
        backend = ElasticsearchValueBackend.from_url(es_url, index_name(tables))
        report = build_report(db, tables, load_saas_metrics(), model, backend, model_name=model_name)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORT_DIR / f"general_regression_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    s = report["summary"]
    print(f"wrote {out} ({report['n_records']} records) · golden {report['golden_sha256'][:12]} · "
          f"config {report['frozen_config_sha256'][:12]}")
    for mode in ("off", "on"):
        t = s["by_mode"][mode]["traps"]["paired"]
        c = s["by_mode"][mode]["controls"]
        print(f"  {mode}: traps dv-vs-ch W/L/T {t['wins']}/{t['losses']}/{t['ties']} "
              f"clean_loss={t['clean_loss_ids']} | control diverging={c['diverging_ids']}")


if __name__ == "__main__":  # pragma: no cover
    main()
