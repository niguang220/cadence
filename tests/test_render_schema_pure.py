from agent.db.build_saas_db import build
from agent.db.introspect import introspect, render_schema, expand_with_fk_neighbors


def test_render_only_renders_given_tables_no_expansion(tmp_path):
    tables = introspect(str(build(tmp_path / "saas.db")))
    rendered = render_schema(tables, only=["subscription"])
    # subscription has an FK to account; a PURE renderer must NOT pull account in.
    assert "TABLE subscription" in rendered
    assert "TABLE account" not in rendered


def test_expand_with_fk_neighbors_still_available(tmp_path):
    tables = introspect(str(build(tmp_path / "saas.db")))
    closure = expand_with_fk_neighbors(tables, ["subscription"])
    assert "subscription" in closure and "account" in closure   # 1-hop neighbor
