# Stage 3B — value retrieval vs the shipping default: targeted head-to-head (280 runs)

**Date:** 2026-08-11 · **Status:** measured public report · **Model:** deepseek-chat · **Backend:** real Elasticsearch 8.19.0

Stage 3B put the shipping default (`current_hybrid`) in the **same real E2E batch** as the RRF
candidate-production line, on the Stage-3A frozen value-linking set, to ask one question: on
**value-sensitive** queries, does the full candidate scheme beat the default? The candidate is always
the existing `RetrievalConfig.dense_value()` (RRF + shortest_path + value) — no new fusion, no
`current_hybrid + value` construction, no name-based routing, no default change.

**Two gates, kept distinct:**
- **Targeted value-sensitive gate: PASS** (this report).
- **General / default-cutover gate: NOT EVALUATED here** — this set is deliberately value-biased, so it
  cannot speak for a default flip. That is Stage 3C's job (a non-value-biased general set).

**Source:** GitHub Actions `value-e2e` (manual, main-only) run **31464161544**. Raw JSON retained
outside Git. Frozen **case SHA `f9c4d8cb…`** (identical to the Stage-3A run) · **config SHA `559b66d7…`**
over the four canonical configs. 4 configs × (10 primary + 4 control) × 5 repeats = **280 records**.

---

## 1. Aggregate (200 primary; 50 runs/config)

| config | exec_match | no_sql | ans_mis | cand_rec | sel_rec | ctx_rec | Fusion@5 | value_degraded |
|---|---|---|---|---|---|---|---|---|
| **current_hybrid** (default) | 13/50 (26%) | 33 | 1 | 0.30 | 0.30 | 0.40 | 0.30 | 0 |
| rrf_hybrid | 9/50 (18%) | 41 | 0 | 0.40 | 0.30 | 0.35 | 0.30 | 0 |
| value_ablation | 28/50 (56%) | 12 | 6 | 0.70 | 0.70 | 0.75 | 0.70 | 0 |
| **dense_value** | **32/50 (64%)** | 13 | 5 | **1.00** | **0.90** | **0.90** | **0.90** | 0 |

Fusion@5 is each config's own candidate ordering (RRF fused rank for RRF configs; min-max hybrid
position for `current_hybrid`).

## 2. Per-case exec_match (matched / 5)

```
case                     current_hybrid  rrf_hybrid  value_ablation  dense_value  category
zh_bjdata_contracts           0/5           0/5           2/5            5/5        zh
zh_shyuntu_tickets            0/5           0/5           2/5            2/5        zh
zh_tianhe_contracts           0/5           0/5           5/5            0/5        zh
en_globex_tickets             5/5           5/5           5/5            5/5        en
en_cyberdyne_region           0/5           0/5           0/5            5/5        en
en_quantumcore_vendor         5/5           4/5           4/5            5/5        en
code_ct0107_owner             0/5           0/5           0/5            0/5        code
code_item_hgt200              0/5           0/5           0/5            0/5        code
homonym_pinnacle_account      3/5           0/5           5/5            5/5        homonym
homonym_pinnacle_product      0/5           0/5           5/5            5/5        homonym
```

## 3. Paired comparisons (exec_match, per case)

- **dense_value vs current_hybrid (the cutover question): 5 wins / 0 losses / 5 ties.**
  Wins: en_cyberdyne_region, homonym_pinnacle_account, homonym_pinnacle_product, zh_bjdata_contracts,
  zh_shyuntu_tickets. Zero losses.
- dense_value vs rrf_hybrid (value increment): **6 wins / 0 losses / 4 ties**.
- dense_value vs value_ablation (dense increment): **3 wins / 1 loss / 6 ties**. The one loss is
  `zh_tianhe_contracts` (dense_value 0/5 vs value_ablation 5/5) — and dense_value's *retrieval* there
  was **better** (cand_recall 1.00 vs 0.50); the extra dense candidates confused generation into wrong
  SQL. Honest signal: the dense channel can hurt generation even where it helps recall.

## 4. Safety (80 controls)

Zero PII leak (0/80). off_topic refuses 20/20. pii/public controls answer but never surface a sentinel
PII token (governance gate holds). `value_degraded` 0/280. Whole-artifact scan clean of raw entity
values / PII / codes.

## 5. Cost — by denominator (the honest version)

`current_hybrid`'s low all-runs p50 is an artifact of its 33 fast refusals; comparing raw p50 across
configs would misstate cost. On **generation-reached** and **answered** runs (apples-to-apples),
dense_value is **not** slower:

| config | all p50/p95 (n) | generation-reached p50/p95 (n) | answered p50/p95 (n) |
|---|---|---|---|
| current_hybrid | 1274/5880 (50) | 5230/7568 (17) | 4922/7745 (14) |
| dense_value | 4606/5555 (50) | 4288/5574 (37) | 4288/5574 (37) |

Tokens (prompt/completion avg): current_hybrid gen-reached 3945/275, dense_value gen-reached 2432/198.
Retrieval adds ~19 ms p50 (ES) for dense_value vs ~0.5 ms lexical. Net: on queries that actually reach
generation, dense_value's latency and tokens are comparable to (slightly below) current_hybrid — the
"3.6× slower" impression from all-runs p50 is an artifact of current_hybrid refusing more.

## 6. Pre-registered gate — verdict

| # | criterion | verdict |
|---|---|---|
| 1 | dense_value vs current_hybrid exec_match non-inferior + multi-case net win | PASS (5–0–5, 64% vs 26%) |
| 2 | candidate/context recall ≥ current_hybrid | PASS (1.00/0.90 vs 0.30/0.40) |
| 3 | controls no new failures | PASS |
| 4 | PII/public/off-topic/injection zero regression | PASS (0/80 leak, clean scan) |
| 5 | ES fail-closed + value_degraded, no wrong grounding | **test-backed PASS** — `value_degraded=0`; this live run did **not** trigger ES degradation, so fail-closed was not exercised here. It is covered by the bulk-error + degrade unit / es_integration tests. |
| 6 | latency/token increment reported honestly | PASS (§5, by denominator) |
| 7 | single-repeat ±1 not treated as deterministic | PASS (the one non-tie loss reported per-case) |

**Targeted value-sensitive gate: PASS. General/default-cutover gate: NOT EVALUATED.**

## 7. What this does and does not license

This licenses `dense_value` as a strong candidate **on value-sensitive queries**. It does **not**
license a default flip: the set is value-biased, current_hybrid's 26% here is expected precisely
because these are the queries value retrieval exists for, and a default affects all queries. The next
step is Stage 3C — the same paired comparison on a non-value-biased general set — not a canary and not
a default change. The production default stays `current_hybrid`.
