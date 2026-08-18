# Cadence project status

**Last reviewed:** 2026-08-15  
**Current release stage:** engineering testbed / pre-release 0.1  
**Shipping retrieval preset:** `governed_rrf` (BM25 lexical + in-memory dense, Weighted RRF, shortest-path relations; semantic governance on by default)

This page is the canonical summary of what is complete, what the measurements support, and
what remains before Cadence can be presented as more than a repository-owned prototype.

## Capability status

| Area | Status | Current boundary |
| --- | --- | --- |
| Core NL-to-SQL workflow | Complete for the demo schema | DeepSeek-specific default model; no provider configuration surface |
| SQL safety and read-only execution | Implemented and tested | SQLite-focused; not an external security audit |
| PII column/result governance | Implemented and tested | Metadata is repository-owned; no tenant or identity model |
| Clarification and plan approval | Implemented | In-memory checkpoint and runtime registries |
| Python analysis sandbox | Implemented | Local Docker one-shot execution |
| Governed metric registry | Implemented | Metrics only; entity and relationship contracts remain partial |
| Typed retrieval pipeline | Implemented and shipping as the default | `legacy_minmax` retained one cycle as a historical comparator |
| Elasticsearch value channel | Implemented, integration-tested, and exposed in the CLI | Requires an external ES service and explicit `index-values` ingestion |
| Reliability harness | Implemented | Most fixtures are small and repository-owned |
| Public benchmark | 30-case Spider screen completed | Custom execution oracle; screening slice, not a leaderboard result |
| Production service | Out of current scope | No multi-tenancy, RLS, durable HITL, or deployment contract |

## Verified local state

On 2026-08-15, the service-free suite completed with:

```text
672 passed, 8 skipped
```

The skipped tests are the opt-in real-Elasticsearch tier. The deterministic scorecard also ran
locally with 14/14 gate routes matching the fixture, 17 consistency fixture checks, and 2
sandbox fixture checks. The gate result is labelled `by_construction`: it verifies the coded
contract and is not an estimate of real-world accuracy.

The public CLI now provides `ask`, `retrieve`, `index-values`, and `build-demo-db` commands,
including JSON output and fail-fast checks for value-enabled presets. The original
`python -m agent QUESTION` and `--retrieval-only` forms remain compatible. The console and
service-free value lifecycle were verified locally; a real ES rerun was unavailable because
Docker Desktop was not connected to this WSL environment. Real ES behavior remains covered by
the opt-in integration workflow.

## What the measured runs say

### 1. Value-sensitive E2E: targeted gate passed

The Stage 3B run used real DeepSeek and Elasticsearch over four retrieval configurations,
10 primary cases, 4 controls, and 5 repeats: 280 records in one batch.

| Configuration | Execution match | Candidate recall | Context recall |
| --- | ---: | ---: | ---: |
| `current_hybrid` (default) | 13/50 (26%) | 0.30 | 0.40 |
| `rrf_hybrid` | 9/50 (18%) | 0.40 | 0.35 |
| `value_ablation` | 28/50 (56%) | 0.70 | 0.75 |
| `dense_value` | **32/50 (64%)** | **1.00** | **0.90** |

There were no PII leaks across 80 control runs. This supports value retrieval for questions
whose answer depends on high-cardinality entity values. It does not establish representative
accuracy because the fixture was deliberately value-sensitive.

Evidence: [Stage 3B report](reliability/2026-08-11-stage3b-current-hybrid-headtohead.md) and
[GitHub Actions run 31464161544](https://github.com/niguang220/cadence/actions/runs/31464161544).

### 2. General-mix E2E: no default cutover yet

Stage 3C compared `current_hybrid` and `dense_value` on the 30-case SaaS metric set, semantic
layer ON and OFF, with 5 repeats: 600 real-agent records. The schema has no searchable columns,
so the value channel was inert; this comparison isolates RRF + shortest-path relations against
legacy min-max + one-hop relations.

| Mode | `current_hybrid` | `dense_value` | Paired case result |
| --- | ---: | ---: | --- |
| Semantic OFF, 24 metric cases | 13/120 (10.8%) | 10/120 (8.3%) | 0 wins / 2 losses / 22 ties |
| Semantic ON, 24 metric cases | 93/120 (77.5%) | **101/120 (84.2%)** | 6 wins / 5 losses / 13 ties |
| Controls, either mode | 25/30 (83.3%) | 25/30 (83.3%) | no paired regressions |

The candidate improved the semantic-ON aggregate, but it also produced five case-level losses,
including one clean loss. Its selector dropped at least one gold table in 35 records. That is
not a clean production promotion signal, so `current_hybrid` remains the default.

Evidence: [GitHub Actions run 31473517414](https://github.com/niguang220/cadence/actions/runs/31473517414).
The raw JSON artifact is retained outside Git under the repository's artifact policy.

### 3. Retrieval diagnosis: the next hypothesis is narrower

The follow-up, LLM-free diagnostics used a 20-table non-saturated fixture:

- Dense retrieval alone reached candidate Recall@15 = 1.00.
- Equal-weight lexical+dense RRF reached candidate Recall@15 = 0.92 and selection recall = 0.576.
- Dense-only weighting reached selection recall = 0.715 with no control selection loss on that
  fixture.

This suggests equal-weight RRF can demote useful dense hits. It does **not** justify changing the
production weight: the hypothesis was generated and measured on repository-owned data and needs
validation on a separate or public set.

### 4. Spider external screen: RRF did not pass the cutover gate

A preregistered Spider dev slice compared `current_hybrid` and `rrf_hybrid` over 30 examples,
15 unseen databases, and 3 repeats: 180 full-agent records with no skips.

| Configuration | Execution match | Candidate recall | Context recall | `no_sql` |
| --- | ---: | ---: | ---: | ---: |
| `current_hybrid` | 55/90 (61.1%) | 96.1% | **100.0%** | **6** |
| `rrf_hybrid` | **56/90 (62.2%)** | **100.0%** | 98.3% | 9 |

RRF was ahead by one execution match and the paired comparison was 3 wins / 2 losses / 25 ties,
but it failed two locked conditions: context recall regressed and `no_sql` exceeded the allowed
increase. This is useful negative evidence, not a default-promotion result. The dominant absolute
failure class was still downstream answer mismatch with mostly complete table context.

Evidence: [preregistration](reliability/2026-08-15-spider-external-preregistration.md) and
[reviewed result](reliability/2026-08-15-spider-external-result.md).

## Current decision

The shipping default is `governed_rrf`: the typed RRF path is now the real product path rather
than an opt-in candidate. The lexical backend and RRF channel weights were fixed by a
deterministic, service-free matrix; the frozen surfaces could not distinguish the six cells, so
the standard BM25 implementation at the neutral equal weighting was taken rather than a weight
fitted to a surface that could not measure it.

Semantic governance is on by default on the Python API, the CLI, and the demo, with an explicit
opt-out. Where the metric registry does not govern a database at all, governance is inert and
traced rather than fatal.

`legacy_minmax` preserves the previous min-max + one-hop implementation for one release cycle as
a historical comparator, and `current_hybrid` remains as a deprecated alias. Elasticsearch value
retrieval stays opt-in. No LLM selector, reranker, or new vector backend was added.

The measured sections above predate this cutover and keep their original configuration names.
The selection matrix and the frozen full-agent gate run that supported it are recorded in the
[governed RRF cutover report](reliability/2026-08-18-governed-rrf-cutover.md).

## Next work, in order

1. **Localize the failed Spider gate.** Trace the extra full-context `no_sql` outcomes and retain
   the five paired divergent cases as a fixed diagnostic set.
2. **Repair selection without widening context blindly.** Test a bounded fix for the one Spider
   case where RRF candidates contained both gold tables but selection/context dropped one.
3. **Address generation failures.** Most external misses were executable answer mismatches despite
   high table recall; retrieval work alone will not fix them.
4. **Confirm before any cutover.** Only a clean diagnostic result should lead to a separately
   preregistered, larger public-benchmark comparison.
5. **Harden the project boundary.** Split the graph module, add durable HITL storage, dependency
   locking, lint/type/coverage checks, and a supported deployment shape only if the project moves
   beyond a local testbed.

Qdrant, an LLM selector, multi-tenancy, and a full semantic model are deliberately not immediate
priorities.
