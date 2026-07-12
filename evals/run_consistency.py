"""Manual real-API driver for the consistency surface (needs DEEPSEEK_API_KEY).

Runs BOTH classes through the real judge: for each case, execute candidate and gold via
the production path (real ``tables`` so column governance runs), self-verify the fixture
invariants (both execute, both governance-clean, and divergence matches expected_caught)
so a broken/mislabeled fixture can never reach the judge even when ``--tier real-api`` is
run without the deterministic teeth, then ask ``check_semantic_consistency``; caught <=>
verdict.ok is False. Adversarial outcomes feed catch-rate, clean outcomes feed
false-positive-rate -- covering only adversarial cases would leave the FP-rate undefined.
"""
from __future__ import annotations

from agent.semantic_consistency import check_semantic_consistency
from evalharness.consistency_eval import (
    ConsistencyOutcome, diverges, execute_case, governance_clean,
)


def run_consistency(db_path, tables, cases, model) -> list[ConsistencyOutcome]:
    outcomes = []
    for case in cases:
        cand, gold = execute_case(case, db_path, tables)
        if not (cand.ok and gold.ok):
            raise RuntimeError(f"{case.id}: candidate or gold SQL failed to execute")
        if not (governance_clean(cand, tables) and governance_clean(gold, tables)):
            raise RuntimeError(f"{case.id}: a result is governance-blocked")
        if diverges(cand, gold) != case.expected_caught:
            raise RuntimeError(f"{case.id}: divergence != expected_caught -- fixture invariant broken")
        verdict = check_semantic_consistency(case.question, case.candidate_sql, cand, model)
        outcomes.append(ConsistencyOutcome(case.id, case.expected_caught, caught=not verdict.ok))
    return outcomes
