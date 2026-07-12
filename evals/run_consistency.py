"""Manual real-API driver for the consistency surface (needs DEEPSEEK_API_KEY).

Runs BOTH classes through the real judge: for each case, execute the candidate via the
production path (real ``tables`` so column governance runs), assert it is
governance-clean, then ask ``check_semantic_consistency``; caught <=> verdict.ok is
False. Adversarial outcomes feed catch-rate, clean outcomes feed false-positive-rate.
Covering only adversarial cases would leave the FP-rate undefined -- so both halves run.
"""
from __future__ import annotations

from agent.semantic_consistency import check_semantic_consistency
from evalharness.consistency_eval import ConsistencyOutcome, execute_case, governance_clean


def run_consistency(db_path, tables, cases, model) -> list[ConsistencyOutcome]:
    outcomes = []
    for case in cases:
        cand, _gold = execute_case(case, db_path, tables)
        if not cand.ok:
            raise RuntimeError(f"{case.id}: candidate SQL failed to execute: {cand.error}")
        if not governance_clean(cand, tables):
            raise RuntimeError(f"{case.id}: candidate result is governance-blocked")
        verdict = check_semantic_consistency(case.question, case.candidate_sql, cand, model)
        outcomes.append(ConsistencyOutcome(case.id, case.expected_caught, caught=not verdict.ok))
    return outcomes
