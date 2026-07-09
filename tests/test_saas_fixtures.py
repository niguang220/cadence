import json, sqlite3
from pathlib import Path
from agent.db.build_saas_db import build
from agent.execution import run_query
from evalharness.oracle import execution_match

CASES = json.loads((Path(__file__).resolve().parent.parent / "evals/golden/saas_metrics.json").read_text())

def _tables(db):
    c = sqlite3.connect(db)
    return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}

def test_cases_loaded():
    assert len(CASES) >= 8

def test_each_gold_and_wrong_run_and_differ(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    for case in CASES:
        gold = run_query(db, case["gold_sql"], assume_safe=True)
        wrong = run_query(db, case["wrong_sql"], assume_safe=True)
        assert gold.ok, f'{case["id"]} gold: {gold.error}'
        assert wrong.ok, f'{case["id"]} wrong: {wrong.error}'
        ordered = "order by" in case["gold_sql"].lower()
        assert not execution_match(wrong.rows, gold.rows, ordered=ordered), \
            f'{case["id"]}: wrong_sql must differ from gold (no oracle teeth)'

def test_required_tables_subset(tmp_path):
    db = str(build(tmp_path / "saas.db"))
    tabs = _tables(db)
    for case in CASES:
        assert set(case["required_tables"]) <= tabs, case["id"]

def test_schema_retrieval_renders_every_required_table(tmp_path):
    # Causal-cleanliness gate: if the schema retriever misses a required table, an OFF/ON
    # miss would be a SCHEMA-LINKING failure, not a business-semantics one. Require recall 1.0.
    import re
    from agent.db.introspect import introspect, render_schema
    from agent.hybrid_retriever import retrieve

    def _rendered_tables(rendered_text, tables):
        """Return the set of table names whose header line appears in rendered_text.

        Anchored to the table header format emitted by render_table:
            TABLE tablename (N rows)
        so a column like ``account_id`` does not cause ``account`` to count as
        present (substring-match false pass fixed here).
        """
        present = set()
        for t in tables:
            # matches: TABLE user (  or  TABLE "user" (  or  CREATE TABLE user (
            if re.search(
                rf'(?im)^\s*(?:CREATE\s+)?TABLE\s+"?{re.escape(t.name)}"?[\s(]',
                rendered_text,
            ):
                present.add(t.name)
        return present

    db = str(build(tmp_path / "saas.db"))
    tables = introspect(db)
    for case in CASES:
        top = retrieve(case["question"], tables, k=5)
        rendered_text = render_schema(tables, only=top, include_fk_neighbors=True)
        rendered = _rendered_tables(rendered_text, tables)
        missing = set(case["required_tables"]) - rendered
        assert not missing, f'{case["id"]}: schema retrieval missed {missing} -> reword question/raise k, do not pass'
