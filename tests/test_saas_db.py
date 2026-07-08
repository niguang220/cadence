import sqlite3
from agent.db.build_saas_db import build, ASOF

def _conn(tmp_path):
    db = build(tmp_path / "saas.db")
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    return c

def test_schema_has_8_tables(tmp_path):
    c = _conn(tmp_path)
    names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"account","user","plan","subscription","mrr_movement",
            "activity_event","invoice","revenue_recognition"} <= names

def test_internal_and_partner_accounts_have_mrr(tmp_path):
    c = _conn(tmp_path)
    for t in ("demo","internal_test"):
        n = c.execute("""SELECT COUNT(*) FROM subscription s JOIN account a USING(account_id)
                         WHERE a.account_type=? AND s.ended_on IS NULL AND s.mrr>0""",(t,)).fetchone()[0]
        assert n >= 1, t
    partners = c.execute("""SELECT COUNT(DISTINCT a.account_id) FROM account a
                            JOIN subscription s USING(account_id)
                            WHERE a.account_type='partner' AND s.phase='paid'""").fetchone()[0]
    assert partners >= 2

def test_phases_present_with_active_comped_and_grandfathered(tmp_path):
    c = _conn(tmp_path)
    for ph in ("trial","paid","comped","grandfathered"):
        assert c.execute("SELECT COUNT(*) FROM subscription WHERE phase=?",(ph,)).fetchone()[0] >= 3, ph
    for ph in ("comped","grandfathered"):
        assert c.execute("SELECT COUNT(*) FROM subscription WHERE phase=? AND ended_on IS NULL",(ph,)).fetchone()[0] >= 2, ph
    assert c.execute("SELECT COUNT(*) FROM subscription WHERE phase='comped' AND ended_on IS NULL AND mrr<>0").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM subscription WHERE phase='grandfathered' AND ended_on IS NULL AND mrr>0").fetchone()[0] >= 2

def test_staff_tier_has_active_subscription(tmp_path):
    c = _conn(tmp_path)
    n = c.execute("""SELECT COUNT(*) FROM subscription s JOIN plan p USING(plan_id)
                     WHERE p.tier='staff' AND s.ended_on IS NULL""").fetchone()[0]
    assert n >= 1

def test_trial_cancellations_are_not_churn(tmp_path):
    c = _conn(tmp_path)
    canceled = c.execute("SELECT COUNT(*) FROM subscription WHERE phase='trial' AND ended_on IS NOT NULL").fetchone()[0]
    assert canceled >= 3
    trial_churn = c.execute("""SELECT COUNT(*) FROM mrr_movement m JOIN subscription s USING(subscription_id)
                               WHERE s.phase='trial' AND m.type='churn'""").fetchone()[0]
    assert trial_churn == 0

def test_at_least_two_annual_plans_and_normalized(tmp_path):
    c = _conn(tmp_path)
    assert c.execute("SELECT COUNT(*) FROM plan WHERE billing_interval='annual'").fetchone()[0] >= 2
    rows = c.execute("""SELECT s.mrr, p.list_price FROM subscription s JOIN plan p USING(plan_id)
                        WHERE p.billing_interval='annual' AND s.phase='paid' AND s.ended_on IS NULL""").fetchall()
    assert rows
    for r in rows:
        assert abs(r["mrr"] - r["list_price"]/12) < 1e-6

def test_activity_window_and_category_interference(tmp_path):
    c = _conn(tmp_path)
    cats = {r[0] for r in c.execute("SELECT DISTINCT event_category FROM activity_event")}
    assert cats == {"page_view","feature_use","api_call","admin_action"}
    qual = c.execute(f"""SELECT COUNT(DISTINCT e.user_id) FROM activity_event e JOIN account a USING(account_id)
        WHERE e.event_category IN ('feature_use','api_call')
          AND e.occurred_at >= date('{ASOF}','-28 day') AND e.occurred_at <= '{ASOF}'
          AND a.account_type NOT IN ('demo','internal_test')""").fetchone()[0]
    naive = c.execute(f"""SELECT COUNT(DISTINCT user_id) FROM activity_event
        WHERE occurred_at >= date('{ASOF}','-30 day') AND occurred_at <= '{ASOF}'""").fetchone()[0]
    assert qual >= 5 and naive > qual

def test_mrr_per_subscription_reconstruction(tmp_path):
    c = _conn(tmp_path)
    rows = c.execute(f"""
        SELECT s.subscription_id,
               CASE WHEN s.ended_on IS NULL AND s.phase IN ('paid','grandfathered') THEN s.mrr ELSE 0 END AS active_mrr,
               COALESCE((SELECT SUM(m.mrr_delta) FROM mrr_movement m
                         WHERE m.subscription_id=s.subscription_id AND m.occurred_on<='{ASOF}'),0) AS flow
        FROM subscription s""").fetchall()
    for r in rows:
        assert abs(r["active_mrr"] - r["flow"]) < 1e-6, r["subscription_id"]

def test_mrr_consistency_under_account_filter(tmp_path):
    c = _conn(tmp_path)
    snap = c.execute(f"""SELECT COALESCE(SUM(s.mrr),0) FROM subscription s JOIN account a USING(account_id)
        WHERE s.ended_on IS NULL AND s.phase IN ('paid','grandfathered')
          AND a.account_type NOT IN ('demo','internal_test')""").fetchone()[0]
    flow = c.execute(f"""SELECT COALESCE(SUM(m.mrr_delta),0) FROM mrr_movement m JOIN account a USING(account_id)
        WHERE m.occurred_on<='{ASOF}' AND a.account_type NOT IN ('demo','internal_test')""").fetchone()[0]
    assert abs(snap - flow) < 1e-6

def test_movement_signs_and_churn_dates(tmp_path):
    c = _conn(tmp_path)
    assert c.execute("SELECT COUNT(*) FROM mrr_movement WHERE type IN ('new','expansion','reactivation') AND mrr_delta<=0").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM mrr_movement WHERE type IN ('churn','contraction') AND mrr_delta>=0").fetchone()[0] == 0
    bad = c.execute("""SELECT COUNT(*) FROM mrr_movement m JOIN subscription s USING(subscription_id)
                       WHERE m.type='churn' AND m.occurred_on<>s.ended_on""").fetchone()[0]
    assert bad == 0

def test_revenue_recognition_sums_to_invoice(tmp_path):
    c = _conn(tmp_path)
    bad = c.execute("""SELECT i.invoice_id FROM invoice i
                       JOIN revenue_recognition r USING(invoice_id)
                       GROUP BY i.invoice_id HAVING ABS(SUM(r.amount)-i.amount)>1e-6""").fetchall()
    assert not bad

def test_seed_is_scaled_up(tmp_path):
    c = _conn(tmp_path)
    assert c.execute("SELECT COUNT(*) FROM account").fetchone()[0] >= 30

def test_test_accounts_have_invoices_in_jan_2025(tmp_path):
    c = _conn(tmp_path)
    for t in ("demo","internal_test"):
        n = c.execute("""SELECT COUNT(*) FROM invoice i JOIN account a USING(account_id)
                         WHERE a.account_type=? AND i.issued_on>='2025-01-01' AND i.issued_on<='2025-01-31'""",(t,)).fetchone()[0]
        assert n >= 1, t

def test_test_accounts_have_qualifying_activity_in_window(tmp_path):
    c = _conn(tmp_path)
    for t in ("demo","internal_test"):
        n = c.execute(f"""SELECT COUNT(*) FROM activity_event e JOIN account a USING(account_id)
            WHERE a.account_type=? AND e.event_category IN ('feature_use','api_call')
              AND e.occurred_at >= date('{ASOF}','-28 day') AND e.occurred_at <= '{ASOF}'""",(t,)).fetchone()[0]
        assert n >= 1, t

def test_excluding_test_accounts_changes_billed_jan(tmp_path):
    # the convention must have a non-zero effect, else the trap is dead
    c = _conn(tmp_path)
    incl = c.execute("SELECT SUM(amount) FROM invoice WHERE issued_on>='2025-01-01' AND issued_on<='2025-01-31'").fetchone()[0]
    excl = c.execute("""SELECT SUM(i.amount) FROM invoice i JOIN account a USING(account_id)
        WHERE i.issued_on>='2025-01-01' AND i.issued_on<='2025-01-31'
          AND a.account_type NOT IN ('demo','internal_test')""").fetchone()[0]
    assert incl != excl
