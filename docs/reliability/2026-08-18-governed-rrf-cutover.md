# Governed typed RRF cutover — selection matrix and release gate

**Date:** 2026-08-18 · **Model:** `deepseek-chat` · **Decision:** cut over to `governed_rrf`

This records the two runs that supported making the typed RRF path the shipping default. It is a
measured-result record, not a re-analysis of the earlier preregistered screen, whose configuration
names and numbers are unchanged.

## 1. Deterministic backend/weight matrix (service-free)

Six frozen cells — `{hand_weighted, bm25}` × lexical weight `{0.25, 0.5, 1.0}`, dense weight fixed
at 1.0. No LLM, no API key, no Docker, no Elasticsearch. Weight 0 was excluded by policy: it is
dense-only ranking, not hybrid fusion.

Surfaces: the 6 explicit-clue control cases on the 20-table confounder fixture (semantic OFF); the
frozen 30-case Spider slice (semantic OFF, retrieval only); and the 24 governed metric cases
(semantic ON) used solely to check the protected-anchor invariant.

| cell | Spider R@5 | R@10 | R@15 | Spider ctx | explicit-clue sel / ctx | governed sel / ctx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hand_weighted @ 0.25 | 0.9833 | 1.000 | 1.000 | 0.9833 | 1.000 / 1.000 | 1.000 / 1.000 |
| hand_weighted @ 0.5 | 0.9833 | 1.000 | 1.000 | 0.9833 | 1.000 / 1.000 | 1.000 / 1.000 |
| hand_weighted @ 1.0 | 0.9833 | 1.000 | 1.000 | 0.9833 | 1.000 / 1.000 | 1.000 / 1.000 |
| bm25 @ 0.25 | 0.9833 | 1.000 | 1.000 | 0.9833 | 1.000 / 1.000 | 1.000 / 1.000 |
| bm25 @ 0.5 | 0.9833 | 1.000 | 1.000 | 0.9833 | 1.000 / 1.000 | 1.000 / 1.000 |
| **bm25 @ 1.0 (selected)** | 0.9833 | 1.000 | 1.000 | 0.9833 | 1.000 / 1.000 | 1.000 / 1.000 |

All six cells were eligible and **indistinguishable on both selection surfaces**. The only
differences were in mean rendered context tables, at the 0.03-table level — one case moving, which
the rule treats as noise rather than a measured advantage.

**Selected: `bm25` at lexical weight 1.0, dense weight 1.0**, by the policy of preferring a
standard maintained implementation absent a clear measured advantage, and then the neutral
untuned weight. Provenance records `surfaces_discriminated: false`: the weight is the untuned
equal weighting, **not** a value fitted to these surfaces. Reading it as a tuned optimum would
overstate what was measured.

Raw artifact: `docs/reliability/lexical_matrix_20260818_090021.json` (git-ignored).

## 2. Frozen full-agent gate run

180 records: 30 frozen Spider dev cases × 15 databases × 2 configurations × 3 repeats, `k=5`,
semantic OFF, clarification off, Cadence's own execution-match oracle. One run, no iterative tuning.

| Measure | `legacy_minmax` | `governed_rrf` |
| --- | ---: | ---: |
| Execution match | 58/90 (64.4%) | 58/90 (64.4%) |
| Candidate recall | 96.1% | **100.0%** |
| Selection recall | 96.1% | **98.3%** |
| Context recall | **100.0%** | 98.3% |
| `no_sql` | **4** | 10 |
| `answer_mismatch` | 28 | **22** |
| Avg prompt tokens | **2449** | 2616 |

Paired: 4 wins / 4 losses / 22 ties. Execution match is identical; the composition of the failures
moved, with six fewer wrong answers and six more refusals.

Latency is not compared: the two arms were not warm-equal, because the first configuration in the
plan pays the embedding index build.

### Additional refusals, attributed case by case

`no_sql` rose from 4 to 10. Seven records refused under `governed_rrf` where the comparator did
not, across five cases:

| index | database | repeats | gold context under `governed_rrf` | comparator outcome |
| --- | --- | ---: | --- | --- |
| 138 | `car_1` | 2 | candidate/selection/context recall all 1.000 | answer_mismatch |
| 161 | `car_1` | 2 | all 1.000 | answer_mismatch |
| 547 | `student_transcripts_tracking` | 1 | all 1.000 | answer_mismatch |
| 576 | `student_transcripts_tracking` | 1 | all 1.000 | **matched** |
| 783 | `world_1` | 1 | all 1.000 | answer_mismatch |

**Every additional refusal had the complete gold table set in context.** None is a retrieval
failure; all are generation-side declines. In six of the seven records the comparator produced a
wrong answer rather than a correct one, so most of this movement is a wrong answer becoming a
refusal. Index 576 repeat 1 is the exception and a genuine loss: the comparator answered correctly
and the default declined.

### Release gate

| Check | Result |
| --- | --- |
| Full service-free suite green | 739 passed, 8 skipped |
| Default path emits typed signals and real fusion scores | pass (asserted in the public-path test) |
| Governed protected-anchor selection/context recall = 1.0 | pass (1.000 / 1.000, all six cells) |
| Controls: no new regression | pass (selection and context recall 1.000, 6/6 admitted) |
| Spider candidate recall ≥ 0.95 | pass (1.000) |
| Spider context recall ≥ 0.95 | pass (0.983) |
| Spider execution match ≥ 0.60 | pass (0.644) |
| Every additional refusal has case-level attribution | pass (7 records, all attributed above) |
| PII, read-only execution, off-topic refusal, determinism | pass |

The earlier preregistered screen's own gate object still evaluates to `passed: false`, because two
of its checks are relative to the comparator (`context_recall_gte_baseline`, `no_sql_ceiling`).
That gate was deliberately left unchanged and un-inverted; it is reported here as diagnostic
evidence, and the absolute release gate above is what this cutover was judged against.

## 3. What this does not establish

- 30 cases scored by a custom oracle is a screen, not a benchmark result.
- The matrix could not discriminate between the six cells, so it justifies *a* defensible default,
  not that BM25 or equal weighting is optimal.
- The governed surface was measured only on repository-owned fixtures; Spider has no governed
  metrics, so protected anchors are structurally inert there.
- Whether the six-fewer-wrong-answers / six-more-refusals trade is desirable is a product
  judgement this run does not settle.
