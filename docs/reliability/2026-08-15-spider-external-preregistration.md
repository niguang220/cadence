# Spider external-validity run — preregistration

**Frozen:** 2026-08-15, before any Cadence model run on this slice  
**Status:** preregistered, not yet measured  
**Benchmark:** Spider dev  
**Primary comparison:** `current_hybrid` vs `rrf_hybrid`

## Question

Does Cadence's typed RRF retrieval path generalize beyond the repository-owned SaaS fixtures,
without regressing the current shipping retrieval preset?

This is an external-validity check for the full agent, not an official Spider leaderboard
submission. It uses Cadence's execution-match oracle rather than the official Spider evaluator.

## Frozen sample

The manifest is `evals/golden/spider_dev_slice.json`.

- Source: Spider `dev.json`, 1,034 examples.
- Required source SHA-256:
  `30d64a3fccde493226df79687aed9e4a1c0129525baf44f29c0573d914d758a4`.
- Selection: `sorted(random.Random(20260815).sample(range(1034), 30))`.
- Coverage: 30 examples across 15 databases.
- The manifest stores source indices and database ids, not copied questions or gold SQL.

The loader must reject a dataset hash, size, index, or database-id mismatch. A gold query that
does not execute is reported as skipped and remains in the manifest; it is never replaced.

## Locked run configuration

- Model: `deepseek-chat`, the repository default, temperature 0.
- Configurations: `current_hybrid`, `rrf_hybrid`.
- Repeats: 3 per case and configuration.
- Total planned records: 30 × 2 × 3 = 180.
- Runtime schema `k`: 5.
- Semantic layer: OFF; its SaaS metric registry does not apply to Spider schemas.
- Clarification: OFF; Spider is an answer-always benchmark.
- Same cases, model, concurrency batch, and repeat indices for both configurations.
- No Elasticsearch or value channel.

No prompt, fixture, retry budget, model, config, or oracle change is allowed between the two
cells. If infrastructure prevents completion, the run is incomplete rather than partially
reported as a comparison.

## Scoring

The gold SQL is executed against the source SQLite database. A prediction matches when Cadence's
existing execution oracle matches result rows and values; row order matters when the gold SQL has
an `ORDER BY` expression.

The report records:

- execution match by configuration and case;
- paired wins, losses, and ties over matched/3;
- `no_sql` counts;
- candidate, selection, and context recall against tables parsed from the gold SQL;
- latency and token use;
- predicted SQL, failure stage, retrieved tables, and retrieval stage events;
- dataset, manifest, configuration, model, and timestamp provenance.

Raw JSON is retained outside Git under `docs/reliability/*.json`. A reviewed Markdown result may
be committed after the run. Raw records refer to the manifest index and do not copy the benchmark
question or gold SQL.

## Candidate gate

`rrf_hybrid` passes this 30-case screening gate only if all are true:

1. Its total execution matches are no more than 2 records below `current_hybrid` over 90 records.
2. Paired per-case wins are at least paired losses, where a win/loss compares matched counts out
   of 3 repeats.
3. Mean candidate recall and mean context recall are each at least the corresponding baseline.
4. Its `no_sql` count is no more than 2 records above the baseline.

Passing licenses a larger confirmation run; it does not by itself change the production default.
Failing keeps `current_hybrid` and localizes the regression before more retrieval features are
added. The absolute external-validity score is descriptive and will be reported even if low.

## Known limitations

- Thirty examples are a screening slice, not the full Spider dev set.
- Three repeats expose some model variance but do not make the observations independent.
- Execution equivalence can accept a semantically different query that happens to return the
  same rows on one database state.
- Spider does not exercise Cadence's governed SaaS metric registry or value-linking channel.
- The source schemas carry no Cadence-specific governance metadata; this run measures query
  capability, not cross-database governance completeness.
