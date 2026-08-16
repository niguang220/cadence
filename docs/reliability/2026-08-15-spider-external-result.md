# Spider external-validity screening result

**Run completed:** 2026-08-15 07:44:06 UTC  
**Decision:** keep `current_hybrid`; `rrf_hybrid` failed the preregistered gate  
**Protocol:** [preregistration](2026-08-15-spider-external-preregistration.md)

## Result

Cadence completed all 180 planned full-agent records: 30 frozen Spider dev examples,
15 databases, two retrieval configurations, and three repeats. All 30 gold queries passed the
local preflight; no record was skipped.

| Measure | `current_hybrid` | `rrf_hybrid` |
| --- | ---: | ---: |
| Execution match | 55/90 (61.1%) | **56/90 (62.2%)** |
| Candidate recall | 96.1% | **100.0%** |
| Selection recall | 96.1% | **98.3%** |
| Context recall | **100.0%** | 98.3% |
| `no_sql` | **6** | 9 |
| Answer mismatch | 29 | **25** |
| Mean latency | **4,206 ms** | 4,370 ms |
| Mean prompt tokens | **2,415** | 2,505 |

The case-level comparison was 3 RRF wins, 2 losses, and 25 ties, based on each case's
matched count out of three. The win indices were 138, 629, and 897; the loss indices were 68
and 576. Indices refer to the frozen manifest, not copied benchmark content.

This is a custom execution-match evaluation, not an official Spider leaderboard submission.

## Preregistered gate

| Check | Result |
| --- | --- |
| No more than two execution matches below the default | Pass (+1) |
| Paired wins at least paired losses | Pass (3 vs 2) |
| Candidate recall at least the default | Pass (100.0% vs 96.1%) |
| Context recall at least the default | **Fail (98.3% vs 100.0%)** |
| `no_sql` no more than two above the default | **Fail (9 vs 6)** |

The gate is conjunctive, so the candidate fails despite its one-record aggregate execution
advantage. The production default remains `current_hybrid`; this screen does not license a
larger confirmation run.

## What the failures suggest

Retrieval is not the main source of the absolute error rate on this slice. Both configurations
usually supplied the gold tables, while SQL generation produced 25-29 executable answer
mismatches. Several divergent cases had the same complete table context in both configurations,
and the three repeats still produced different SQL at temperature zero. The one-record aggregate
lead is therefore too small to separate retrieval quality from generation variance.

The two gate failures are concrete:

- RRF dropped one of two required tables from selection and context in all three repeats of
  manifest index 584, even though that table was present in its candidate set. This is a
  top-k selection/relationship-closure failure, not a dense-recall failure.
- The extra `no_sql` outcomes appeared with full context on several cases. They are concentrated
  in downstream planning/generation behavior rather than missing candidate tables and require
  trace-level diagnosis before another retrieval experiment.

RRF also cost about 3.9% more latency and 3.7% more prompt tokens on average. These are descriptive
measurements from one concurrent batch, not standalone performance benchmarks.

## Provenance and audit checks

- Model: `deepseek-chat`, temperature 0.
- Runtime: semantic layer OFF, clarification OFF, `k=5`, concurrency 4.
- Spider `dev.json` SHA-256:
  `30d64a3fccde493226df79687aed9e4a1c0129525baf44f29c0573d914d758a4`.
- Frozen manifest SHA-256:
  `fcc74736d50942a1b5974d73ba7c5d08ff6f370628dc4459cb4ccd51bce723cf`.
- The raw report contains exactly 180 unique `(configuration, index, repeat)` keys and no
  `question` or `gold_sql` fields.
- Raw artifact: `docs/reliability/spider_external_20260815_154406.json` (intentionally ignored
  by Git; reviewed conclusions are versioned here).

## Next action

Keep the current retrieval default. Before testing another retrieval candidate, retain the
failing cases as a diagnostic set and instrument why full-context runs return no SQL. Separately,
test a bounded selection/closure change against index 584 and the existing controls; do not
increase context globally without measuring its effect on token use and execution accuracy.
