"""Value-index governance allowlist (fail-closed).

Only columns explicitly marked policy=="searchable" may be indexed for entity-value search.
public/unknown columns are not indexed; pii is never indexed. A misconfigured policy (unknown
policy string, or a COLUMN_POLICIES entry naming a column that does not exist) must fail-fast --
a silent fall-through could unmark a PII column and leak it into the index."""
import pytest

import agent.retrieval.value_policy as vp
from agent.db.introspect import Column, Table
from agent.retrieval.value_policy import resolve_searchable_columns, validate_value_policy


def _col(name, policy="public"):
    return Column(name=name, type="TEXT", pk=False, notnull=False, policy=policy)


def _tables():
    # B2B pattern: company/product/sku names are searchable; contact email is PII.
    return [
        Table(name="account", columns=[_col("account_id"), _col("company_name", "searchable"),
                                        _col("contact_email", "pii")]),
        Table(name="product", columns=[_col("name", "searchable"), _col("sku", "searchable"),
                                        _col("price")]),
    ]


def test_resolve_returns_only_searchable_columns_sorted():
    assert resolve_searchable_columns(_tables()) == [
        ("account", "company_name"), ("product", "name"), ("product", "sku")]


def test_pii_and_public_are_never_resolved():
    resolved = set(resolve_searchable_columns(_tables()))
    assert ("account", "contact_email") not in resolved   # pii excluded
    assert ("account", "account_id") not in resolved      # public (default) excluded
    assert ("product", "price") not in resolved           # public excluded


def test_invalid_policy_string_fails_closed():
    tbls = [Table(name="x", columns=[_col("c", "piii")])]   # typo -> must not silently pass
    with pytest.raises(ValueError):
        validate_value_policy(tbls)
    with pytest.raises(ValueError):
        resolve_searchable_columns(tbls)                    # resolve validates first


def test_policy_entry_for_nonexistent_column_fails(monkeypatch):
    # A typo'd column name in COLUMN_POLICIES would leave the real PII column unmarked.
    monkeypatch.setattr(vp, "COLUMN_POLICIES", {"account": {"emial": "pii"}})
    tbls = [Table(name="account", columns=[_col("account_id"), _col("email", "pii")])]
    with pytest.raises(ValueError, match="does not exist"):
        validate_value_policy(tbls)


def test_policy_for_absent_table_is_ignored(monkeypatch):
    # A policy for a table not in this catalog (a different fixture) is not our concern here.
    monkeypatch.setattr(vp, "COLUMN_POLICIES", {"other_db_table": {"x": "pii"}})
    tbls = [Table(name="account", columns=[_col("company_name", "searchable")])]
    validate_value_policy(tbls)   # no raise
    assert resolve_searchable_columns(tbls) == [("account", "company_name")]
