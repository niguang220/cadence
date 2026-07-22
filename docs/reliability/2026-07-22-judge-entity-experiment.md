# Reliability experiment — does the semantic-consistency judge's entity blind spot have a cheap fix?

**Date:** 2026-07-22 · **Component:** `agent/semantic_consistency.py` (the LLM judge) ·
**Outcome:** both candidate entity fixes rejected by pre-registered evaluation; the production prompt and
entity behavior were left unchanged (an unrelated non-string fail-open hardening did ship).

## Background

The reliability harness's first real-API scorecard surfaced a stable blind spot: the judge
misses certain **entity-substitution** errors — counting the wrong table for an entity
question. On the fuller fixture set this narrows to the two direct `account`↔`user`
substitutions (it catches the alias and other-table cases). This log records two attempted
fixes and why the pre-registered evaluation rejected both.

## Method

- **Frozen fixture set:** 10 adversarial + 7 clean consistency cases
  (`evals/golden/consistency.json`, SHA-256 `457abfb5…39aa9`), committed and hashed *before*
  running and not changed afterwards.
- **Design:** the set includes clean controls — with exact paired controls for the alias
  probes — so a high catch-rate cannot hide an over-flagging (trigger-happy) judge.
- **Runner:** one judge call per case; 5 repeat runs; `deepseek-chat` @ temperature 0.
- **Locked acceptance criteria (pre-registered):** the original 7 adversarial cases all 5/5;
  all 7 clean cases 0/5 false-positives; a candidate keeps the catalog only if the full
  catalog *strictly* out-catches the empty catalog on the same alias case in 5/5 runs.

## Fix 1 — feed the judge a compact schema catalog

Ablation: empty catalog vs full catalog, same prompt, 5 runs each.

| | empty catalog | full catalog |
| --- | --- | --- |
| adversarial caught 5/5 | 10/10 | 10/10 |
| clean cases falsely flagged | 1/7 | 1/7 |

**Result: the catalog adds nothing** — identical catch and identical false-positives with and
without it. (The one false-positive, `clean_alias_logins_ok`, comes from the tightened prompt
in place during this ablation, not from the catalog; it is examined in Fix 2.) Rejected per
the locked rule.

## Fix 2 — tighten the judge prompt (independent three-axis + primary-object entity rule)

One bounded attempt. It raised adversarial recall, but the clean controls caught the cost:
the *correct* query `clean_alias_logins_ok` ("How many individual login identities have been
provisioned?" → `COUNT(*) FROM "user"`) was flagged **5/5** — a false refusal of a legitimate
query, violating the conservative-judge invariant. The attempt failed the locked acceptance
criteria and was reverted.

## Decision

Both fixes reverted. The judge prompt is byte-identical to `main`. The entity blind spot
remains a documented, measured limitation.

## Shipped baseline (for the record)

The reverted (original) judge on the frozen 17-case set, five repeats, `deepseek-chat` @ temp
0, golden SHA-256 `457abfb5…`:

- adversarial caught 5/5: **8/10** — the two misses are the `account`↔`user` blind spot above
- clean cases falsely flagged: **0/7** — on this set, across five repeats, the judge did not
  falsely refuse a legitimate query

These remain small-set empirical results, not a stable-capability claim. The trade the
evaluation revealed: closing the recall gap cheaply cost precision, which for a check that
gates production queries is the wrong trade.
