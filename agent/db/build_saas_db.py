"""Build the SaaS analytics SQLite database for the business-semantics experiment.

Company-specific metric conventions are baked into enum subsets so a raw LLM
cannot infer them from column names alone:

- demo/internal_test account_types look like paying customers but are excluded
  from business MRR and engagement metrics.
- trial phase cancellations are NOT churn (no mrr_movement of type 'churn' for
  trial subscriptions).
- comped subscriptions carry mrr=0; grandfathered carry mrr>0.
- The governed engagement window (28 days, feature_use/api_call only, non-test
  accounts) deliberately differs from the naive window (30 days, all events,
  all accounts), so a raw COUNT-DISTINCT query returns the wrong answer.

    python -m agent.db.build_saas_db          # writes agent/db/saas.db
"""
from __future__ import annotations

import random as _random
import sqlite3
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "saas.db"
ASOF = "2025-06-30"
SEED = 20260628

SCHEMA = """
CREATE TABLE account (
    account_id    INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    account_type  TEXT NOT NULL CHECK (account_type IN ('standard','partner','demo','internal_test')),
    region        TEXT NOT NULL,
    signup_date   TEXT NOT NULL
);
CREATE TABLE "user" (
    user_id     INTEGER PRIMARY KEY,
    account_id  INTEGER NOT NULL REFERENCES account(account_id),
    email       TEXT NOT NULL,
    role        TEXT NOT NULL
);
CREATE TABLE plan (
    plan_id          INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    billing_interval TEXT NOT NULL CHECK (billing_interval IN ('monthly','annual')),
    list_price       REAL NOT NULL,
    tier             TEXT NOT NULL CHECK (tier IN ('free','starter','pro','enterprise','staff'))
);
CREATE TABLE subscription (
    subscription_id INTEGER PRIMARY KEY,
    account_id      INTEGER NOT NULL REFERENCES account(account_id),
    plan_id         INTEGER NOT NULL REFERENCES plan(plan_id),
    started_on      TEXT NOT NULL,
    ended_on        TEXT,
    phase           TEXT NOT NULL CHECK (phase IN ('trial','paid','comped','grandfathered')),
    mrr             REAL NOT NULL
);
CREATE TABLE mrr_movement (
    movement_id     INTEGER PRIMARY KEY,
    account_id      INTEGER NOT NULL REFERENCES account(account_id),
    subscription_id INTEGER NOT NULL REFERENCES subscription(subscription_id),
    type            TEXT NOT NULL CHECK (type IN ('new','expansion','contraction','churn','reactivation')),
    occurred_on     TEXT NOT NULL,
    mrr_delta       REAL NOT NULL
);
CREATE TABLE activity_event (
    event_id       INTEGER PRIMARY KEY,
    user_id        INTEGER NOT NULL REFERENCES "user"(user_id),
    account_id     INTEGER NOT NULL REFERENCES account(account_id),
    event_category TEXT NOT NULL CHECK (event_category IN ('page_view','feature_use','api_call','admin_action')),
    occurred_at    TEXT NOT NULL
);
CREATE TABLE invoice (
    invoice_id   INTEGER PRIMARY KEY,
    account_id   INTEGER NOT NULL REFERENCES account(account_id),
    issued_on    TEXT NOT NULL,
    amount       REAL NOT NULL,
    period_start TEXT NOT NULL,
    period_end   TEXT NOT NULL
);
CREATE TABLE revenue_recognition (
    recognition_id INTEGER PRIMARY KEY,
    invoice_id     INTEGER NOT NULL REFERENCES invoice(invoice_id),
    recognized_on  TEXT NOT NULL,
    amount         REAL NOT NULL
);
"""

# ── static reference data ─────────────────────────────────────────────────────

_ACCOUNTS = [
    # (account_id, name, account_type, region, signup_date)
    # standard  1-5
    (1,  "Acme Corp",          "standard",      "us-east",   "2023-01-15"),
    (2,  "Globex Inc",         "standard",      "us-west",   "2023-03-01"),
    (3,  "Initech LLC",        "standard",      "eu-west",   "2023-05-10"),
    (4,  "Umbrella Ltd",       "standard",      "ap-south",  "2023-06-20"),
    (5,  "Stark Industries",   "standard",      "ca-central","2023-08-01"),
    # partner   6-9
    (6,  "Blue Reseller",      "partner",       "us-east",   "2023-02-01"),
    (7,  "Red Partner",        "partner",       "eu-west",   "2023-04-15"),
    (8,  "Green Alliance",     "partner",       "us-west",   "2023-07-01"),
    (9,  "Silver Channel",     "partner",       "ap-south",  "2023-09-10"),
    # demo      10-12
    (10, "Demo Alpha",         "demo",          "us-east",   "2024-01-01"),
    (11, "Demo Beta",          "demo",          "eu-west",   "2024-02-01"),
    (12, "Demo Gamma",         "demo",          "us-west",   "2024-03-01"),
    # internal_test  13-15
    (13, "Test Org A",         "internal_test", "us-east",   "2024-01-01"),
    (14, "Test Org B",         "internal_test", "eu-west",   "2024-02-01"),
    (15, "Test Org C",         "internal_test", "us-west",   "2024-03-01"),
]

# 3 users per account; user_id = (account_id - 1) * 3 + slot (1-indexed)
_USERS: list[tuple] = []
for _acc in _ACCOUNTS:
    _aid = _acc[0]
    for _slot in range(1, 4):
        _uid = (_aid - 1) * 3 + _slot
        _USERS.append((
            _uid, _aid,
            f"u{_uid}@acc{_aid}.example.com",
            ("admin" if _slot == 1 else "member"),
        ))

def _first_user(account_id: int) -> int:
    """Return the user_id of the first (admin) user for a given account."""
    return (account_id - 1) * 3 + 1

def _second_user(account_id: int) -> int:
    return (account_id - 1) * 3 + 2

_PLANS = [
    # (plan_id, name, billing_interval, list_price, tier)
    # Annual prices chosen so list_price/12 is exact in IEEE-754 float64
    (1, "Free Monthly",        "monthly", 0.0,    "free"),
    (2, "Starter Monthly",     "monthly", 49.0,   "starter"),
    (3, "Starter Annual",      "annual",  588.0,  "starter"),   # 588/12 = 49.0 exact
    (4, "Pro Monthly",         "monthly", 99.0,   "pro"),
    (5, "Pro Annual",          "annual",  1188.0, "pro"),        # 1188/12 = 99.0 exact
    (6, "Enterprise Monthly",  "monthly", 299.0,  "enterprise"),
    (7, "Enterprise Annual",   "annual",  3588.0, "enterprise"), # 3588/12 = 299.0 exact
    (8, "Staff Monthly",       "monthly", 0.0,    "staff"),
]

# ── subscriptions ──────────────────────────────────────────────────────────────
# (sub_id, account_id, plan_id, started_on, ended_on, phase, mrr)
#
# Invariant notes baked in by construction:
#  - trial subs: mrr=0; canceled ones have ended_on set
#  - paid annual: mrr = list_price/12 exactly
#  - comped active: mrr=0
#  - grandfathered active: mrr>0 (below-market legacy rate)
#  - staff tier (plan 8): active sub with mrr=0
#  - demo acc 10 has active paid mrr>0; internal_test acc 13 has active paid mrr>0
#  - partner accs 6,7,8 each have a paid sub (≥2 partners with paid)

_SUBSCRIPTIONS = [
    # ── trials (5 total; 3 canceled = subs 1,2,3) ────────────────────────────
    (1,  1,  2, "2025-01-01", "2025-01-15", "trial", 0.0),   # std acme, canceled
    (2,  2,  2, "2025-02-01", "2025-02-10", "trial", 0.0),   # std globex, canceled
    (3,  3,  4, "2025-03-01", "2025-03-15", "trial", 0.0),   # std initech, canceled
    (4,  4,  2, "2025-04-01", None,          "trial", 0.0),   # std umbrella, active
    (5,  5,  4, "2025-05-01", None,          "trial", 0.0),   # std stark, active
    # ── paid (7 total) ────────────────────────────────────────────────────────
    (6,  1,  2, "2024-01-01", None,          "paid",  49.0),  # std, starter monthly
    (7,  6,  4, "2024-02-01", None,          "paid",  99.0),  # partner, pro monthly
    (8,  7,  5, "2024-03-01", None,          "paid",  99.0),  # partner, pro annual (1188/12)
    (9,  10, 2, "2024-06-01", None,          "paid",  49.0),  # demo active paid mrr>0 ✓
    (10, 13, 4, "2024-06-01", None,          "paid",  99.0),  # internal_test active paid mrr>0 ✓
    (11, 2,  6, "2024-06-01", "2025-03-01",  "paid",  299.0), # std globex, churned
    (12, 8,  3, "2024-01-01", None,          "paid",  49.0),  # partner, starter annual (588/12)
    # ── comped (4 total; 3 active; all active mrr=0) ─────────────────────────
    (13, 9,  2, "2024-01-01", None,          "comped", 0.0),  # partner silver, active
    (14, 3,  4, "2024-03-01", None,          "comped", 0.0),  # std initech, active
    (15, 11, 2, "2024-06-01", "2025-01-01",  "comped", 0.0),  # demo beta, inactive
    (16, 14, 8, "2024-01-01", None,          "comped", 0.0),  # int_test B, staff plan, active ✓
    # ── grandfathered (3 total; 2 active mrr>0) ───────────────────────────────
    (17, 4,  2, "2023-06-01", None,          "grandfathered", 29.0),  # std umbrella, active ✓
    (18, 5,  4, "2023-01-01", None,          "grandfathered", 59.0),  # std stark, active ✓
    (19, 12, 6, "2022-01-01", "2024-12-31",  "grandfathered", 199.0), # demo gamma, inactive
]

# ── mrr_movement (derived deterministically from subscription lifecycle) ───────
#
# Rule: emit 'new' (+mrr) at started_on for every paid/grandfathered sub with
# mrr>0; emit 'churn' (-mrr) at ended_on if the sub is canceled.
# Trial/comped/zero-mrr subs produce NO movements (trial cancellations ≠ churn).

def _derive_movements(subscriptions: list[tuple]) -> list[tuple]:
    rows: list[tuple] = []
    mvt_id = 1
    for sub in subscriptions:
        sub_id, account_id, _plan_id, started_on, ended_on, phase, mrr = sub
        if phase not in ("paid", "grandfathered") or mrr <= 0.0:
            continue
        rows.append((mvt_id, account_id, sub_id, "new", started_on, mrr))
        mvt_id += 1
        if ended_on is not None:
            rows.append((mvt_id, account_id, sub_id, "churn", ended_on, -mrr))
            mvt_id += 1
    return rows

_MOVEMENTS = _derive_movements(_SUBSCRIPTIONS)

# ── seeded filler (accounts 16-35, plus biting test-account rows) ─────────────
#
# Uses random.Random(SEED) so the build is fully deterministic.
# Biting requirements:
#   demo AND internal_test accounts each get:
#     (a) ≥1 invoice issued in Jan 2025
#     (b) ≥1 feature_use/api_call within the 28-day ASOF window
#     (c) ≥1 active paid sub with mrr>0   (also satisfies existing invariant)
# The first filler demo account (acc 28) and first filler internal_test account
# (acc 32) carry the explicit biting rows; the rest are seeded from rng.

def _build_filler() -> tuple:
    rng = _random.Random(SEED)

    REGIONS = ["us-east", "us-west", "eu-west", "eu-central", "ap-south", "ca-central"]
    PAID_MONTHLY = [(2, 49.0), (4, 99.0), (6, 299.0)]  # (plan_id, mrr) all monthly

    # ── accounts 16-35: 8 standard, 4 partner, 4 demo, 4 internal_test ─────────
    acc_type_seq = ["standard"] * 8 + ["partner"] * 4 + ["demo"] * 4 + ["internal_test"] * 4
    filler_accounts: list[tuple] = []
    for i, atype in enumerate(acc_type_seq):
        acc_id = 16 + i
        region = rng.choice(REGIONS)
        year   = rng.randint(2022, 2024)
        month  = rng.randint(1, 12)
        day    = rng.randint(1, 28)
        filler_accounts.append((
            acc_id,
            f"Filler {atype.replace('_', ' ').title()} {i + 1}",
            atype,
            region,
            f"{year}-{month:02d}-{day:02d}",
        ))

    # ── users: 2 per filler account (IDs 46..) ──────────────────────────────────
    # Curated accounts 1-15 each have 3 users → 45 curated users.
    # Filler account N gets users 45 + (N-16)*2 + 1  and  45 + (N-16)*2 + 2.
    filler_users: list[tuple] = []
    for fa in filler_accounts:
        aid  = fa[0]
        base = 45 + (aid - 16) * 2 + 1
        filler_users.append((base,     aid, f"u{base}@acc{aid}.example.com",     "admin"))
        filler_users.append((base + 1, aid, f"u{base + 1}@acc{aid}.example.com", "member"))

    def _filler_first_user(aid: int) -> int:
        return 45 + (aid - 16) * 2 + 1

    # ── subscriptions (IDs start at 20) ─────────────────────────────────────────
    filler_subs: list[tuple] = []
    sub_id = 20

    # Standard (16-23): random phase mix
    for aid in range(16, 24):
        phase   = rng.choice(["paid", "paid", "trial", "comped", "grandfathered"])
        s_year  = rng.randint(2023, 2024)
        s_month = rng.randint(1, 12)
        started = f"{s_year}-{s_month:02d}-01"
        if phase == "paid":
            plan_id, mrr = rng.choice(PAID_MONTHLY)
            ended = None
            if rng.random() < 0.25:
                e_year  = s_year + (1 if s_month == 12 else 0)
                e_month = (s_month % 12) + 1
                ended   = f"{e_year}-{e_month:02d}-01"
            filler_subs.append((sub_id, aid, plan_id, started, ended, "paid", mrr))
        elif phase == "trial":
            plan_id = rng.choice([2, 4])
            ended   = None
            if rng.random() < 0.5:
                e_year  = s_year + (1 if s_month == 12 else 0)
                e_month = (s_month % 12) + 1
                ended   = f"{e_year}-{e_month:02d}-15"
            filler_subs.append((sub_id, aid, plan_id, started, ended, "trial", 0.0))
        elif phase == "comped":
            plan_id = rng.choice([2, 4, 8])
            filler_subs.append((sub_id, aid, plan_id, started, None, "comped", 0.0))
        else:  # grandfathered
            plan_id = rng.choice([2, 4])
            mrr     = rng.choice([29.0, 39.0, 59.0, 79.0])
            filler_subs.append((sub_id, aid, plan_id, started, None, "grandfathered", mrr))
        sub_id += 1

    # Partner (24-27): paid monthly
    for aid in range(24, 28):
        plan_id, mrr = rng.choice(PAID_MONTHLY)
        s_year  = rng.randint(2023, 2024)
        s_month = rng.randint(1, 12)
        filler_subs.append((
            sub_id, aid, plan_id,
            f"{s_year}-{s_month:02d}-01", None, "paid", mrr,
        ))
        sub_id += 1

    # Demo (28-31): acc 28 is the explicit biting account; 29-31 are seeded
    filler_subs.append((sub_id, 28, 4, "2024-06-01", None, "paid", 99.0))  # biting
    sub_id += 1
    for aid in range(29, 32):
        plan_id, mrr = rng.choice(PAID_MONTHLY)
        s_year  = rng.randint(2023, 2024)
        s_month = rng.randint(1, 12)
        filler_subs.append((
            sub_id, aid, plan_id,
            f"{s_year}-{s_month:02d}-01", None, "paid", mrr,
        ))
        sub_id += 1

    # internal_test (32-35): acc 32 is the explicit biting account; 33-35 are seeded
    filler_subs.append((sub_id, 32, 2, "2024-05-01", None, "paid", 49.0))  # biting
    sub_id += 1
    for aid in range(33, 36):
        plan_id, mrr = rng.choice(PAID_MONTHLY)
        s_year  = rng.randint(2023, 2024)
        s_month = rng.randint(1, 12)
        filler_subs.append((
            sub_id, aid, plan_id,
            f"{s_year}-{s_month:02d}-01", None, "paid", mrr,
        ))
        sub_id += 1

    # ── activity events (IDs start at 14) ────────────────────────────────────────
    # Explicit biting events come first; the rest are seeded.
    filler_events: list[tuple] = []
    ev_id = 14
    CATS = ["page_view", "feature_use", "api_call", "admin_action"]

    # (b) Biting: demo acc 28 — feature_use in 28-day window
    filler_events.append((ev_id, _filler_first_user(28), 28, "feature_use", "2025-06-10"))
    ev_id += 1
    # (b) Biting: internal_test acc 32 — api_call in 28-day window
    filler_events.append((ev_id, _filler_first_user(32), 32, "api_call",    "2025-06-15"))
    ev_id += 1

    # Seeded events for standard+partner filler accounts (mostly historical)
    for aid in range(16, 28):
        year  = rng.randint(2024, 2025)
        month = rng.randint(1, 6)
        day   = rng.randint(1, 28)
        cat   = rng.choice(CATS)
        filler_events.append((ev_id, _filler_first_user(aid), aid, cat,
                              f"{year}-{month:02d}-{day:02d}"))
        ev_id += 1

    # Seeded historical events for remaining demo+internal_test filler accounts
    for aid in list(range(29, 32)) + list(range(33, 36)):
        year  = rng.randint(2024, 2025)
        month = rng.randint(1, 5)
        day   = rng.randint(1, 28)
        cat   = rng.choice(CATS)
        filler_events.append((ev_id, _filler_first_user(aid), aid, cat,
                              f"{year}-{month:02d}-{day:02d}"))
        ev_id += 1

    # ── invoice specs ─────────────────────────────────────────────────────────────
    # Format: (account_id, amount, issued_on, period_start, period_end)
    # All filler invoices are monthly (one recognition row = full amount).
    filler_invoice_specs: list[tuple] = []

    # (a) Biting: demo acc 28 — invoice issued in Jan 2025
    filler_invoice_specs.append((28, 99.0, "2025-01-01", "2025-01-01", "2025-01-31"))
    # (a) Biting: internal_test acc 32 — invoice issued in Jan 2025
    filler_invoice_specs.append((32, 49.0, "2025-01-01", "2025-01-01", "2025-01-31"))

    # Seeded invoices for standard filler accounts
    for aid in range(16, 24):
        amt   = rng.choice([49.0, 99.0, 299.0])
        month = rng.randint(3, 6)
        issued = f"2025-{month:02d}-01"
        filler_invoice_specs.append((aid, amt, issued, issued, f"2025-{month:02d}-28"))

    return filler_accounts, filler_users, filler_subs, filler_events, filler_invoice_specs


# ── activity events ────────────────────────────────────────────────────────────
#
# ASOF = 2025-06-30
# 28-day window: occurred_at >= 2025-06-02 (date(ASOF,'-28 day'))
# 30-day window: occurred_at >= 2025-05-31 (date(ASOF,'-30 day'))
#
# Governed count  = DISTINCT users with feature_use/api_call in 28-day window
#                   at non-test accounts (not demo/internal_test)  → must be ≥5
# Naive count     = DISTINCT users with ANY event in 30-day window
#                   at ALL accounts                                 → must be > governed
#
# Design:
#   6 users from standard/partner accounts emit feature_use in 28-day window
#     → governed = 6
#   Additionally: account-4 user emits page_view (in 28d window, wrong category);
#                 account-5 user emits admin_action (in 28d window, wrong category);
#                 account-9 user emits api_call at 2025-05-31 (in 30d but not 28d);
#                 demo-10 user emits page_view in 28d window (test account, excluded);
#                 internal_test-13 user emits feature_use in 28d window (test acct)
#     → naive = 11 (6 governed + 2 wrong-category + 1 outside-28d + 2 test-acct)
#   All 4 categories appear.

_ACTIVITY: list[tuple] = [
    # event_id, user_id, account_id, event_category, occurred_at
    # ── 6 governed users: feature_use in 28-day window, non-test accounts ─────
    (1,  _first_user(1),  1,  "feature_use",  "2025-06-15"),
    (2,  _first_user(2),  2,  "feature_use",  "2025-06-15"),
    (3,  _first_user(3),  3,  "feature_use",  "2025-06-15"),
    (4,  _first_user(6),  6,  "feature_use",  "2025-06-15"),
    (5,  _first_user(7),  7,  "feature_use",  "2025-06-15"),
    (6,  _first_user(8),  8,  "feature_use",  "2025-06-15"),
    # api_call to cover that category (same user as event 1, doesn't add to count)
    (7,  _first_user(1),  1,  "api_call",     "2025-06-20"),
    # ── wrong-category events from non-test accts (in 28d window, not counted) ─
    (8,  _first_user(4),  4,  "page_view",    "2025-06-10"),
    (9,  _first_user(5),  5,  "admin_action", "2025-06-12"),
    # ── partner acct-9 user: api_call outside 28d but inside 30d window ───────
    (10, _first_user(9),  9,  "api_call",     "2025-05-31"),
    # ── test-account users: events in 30d window (not in governed) ────────────
    (11, _first_user(10), 10, "page_view",    "2025-06-05"),   # demo
    (12, _first_user(13), 13, "feature_use",  "2025-06-08"),   # internal_test
    # ── historical event outside 30d window (not counted anywhere) ────────────
    (13, _second_user(1), 1,  "page_view",    "2025-01-15"),
]

# ── invoices + revenue recognition ────────────────────────────────────────────
#
# Annual invoices → 12 monthly recognition rows (each = amount/12).
# Monthly invoices → 1 recognition row (= amount).
# All list_prices for annual plans are multiples of 12 so /12 is exact in float.

def _build_invoices_and_recognition() -> tuple[list[tuple], list[tuple]]:
    invoices: list[tuple] = []
    recs: list[tuple] = []
    inv_id = 1
    rec_id = 1

    def add_monthly(account_id: int, amount: float, issued: str,
                    period_start: str, period_end: str) -> None:
        nonlocal inv_id, rec_id
        invoices.append((inv_id, account_id, issued, amount, period_start, period_end))
        recs.append((rec_id, inv_id, period_start, amount))
        rec_id += 1
        inv_id += 1

    def add_annual(account_id: int, amount: float, issued: str, year: int) -> None:
        nonlocal inv_id, rec_id
        period_start = f"{year}-01-01"
        period_end   = f"{year}-12-31"
        invoices.append((inv_id, account_id, issued, amount, period_start, period_end))
        monthly = amount / 12  # exact for all multiples of 12
        for m in range(1, 13):
            recognized_on = date(year, m, 1).isoformat()
            recs.append((rec_id, inv_id, recognized_on, monthly))
            rec_id += 1
        inv_id += 1

    # sub 6: acme (acc 1), starter monthly $49
    add_monthly(1, 49.0,   "2025-06-01", "2025-06-01", "2025-06-30")
    # sub 7: blue reseller (acc 6), pro monthly $99
    add_monthly(6, 99.0,   "2025-06-01", "2025-06-01", "2025-06-30")
    # sub 8: red partner (acc 7), pro annual $1188
    add_annual(7,  1188.0, "2025-01-01", 2025)
    # sub 12: green alliance (acc 8), starter annual $588
    add_annual(8,  588.0,  "2024-01-01", 2024)
    # sub 11: globex (acc 2), enterprise monthly invoice before churn
    add_monthly(2, 299.0,  "2025-02-01", "2025-02-01", "2025-02-28")

    return invoices, recs


_INVOICES, _RECOGNITIONS = _build_invoices_and_recognition()

# ── seeded-filler call + combined lists (all curated data is defined above) ───

(_FILLER_ACCOUNTS, _FILLER_USERS, _FILLER_SUBS,
 _FILLER_EVENTS, _FILLER_INVOICE_SPECS) = _build_filler()

_ALL_ACCOUNTS  = _ACCOUNTS      + _FILLER_ACCOUNTS
_ALL_USERS     = _USERS         + _FILLER_USERS
_ALL_SUBS      = _SUBSCRIPTIONS + _FILLER_SUBS
_ALL_MOVEMENTS = _derive_movements(_ALL_SUBS)   # re-derives IDs over full sub list
_ALL_EVENTS    = _ACTIVITY      + _FILLER_EVENTS


def _build_all_invoices_and_recognition() -> tuple[list[tuple], list[tuple]]:
    """Extend the curated invoices/recognitions with seeded filler rows."""
    invoices, recs = _build_invoices_and_recognition()
    inv_id = len(invoices) + 1   # curated = 5 → filler starts at 6
    rec_id = len(recs) + 1       # curated = 27 → filler starts at 28
    for acct_id, amount, issued_on, period_start, period_end in _FILLER_INVOICE_SPECS:
        invoices.append((inv_id, acct_id, issued_on, amount, period_start, period_end))
        recs.append((rec_id, inv_id, period_start, amount))
        rec_id += 1
        inv_id += 1
    return invoices, recs


_ALL_INVOICES, _ALL_RECOGNITIONS = _build_all_invoices_and_recognition()


# ── public builder ─────────────────────────────────────────────────────────────

def build(path: str | Path = DB_PATH) -> Path:
    """Build the SaaS analytics DB at *path*. Replaces any existing file."""
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        cur = conn.cursor()

        cur.executemany("INSERT INTO account VALUES(?,?,?,?,?)", _ALL_ACCOUNTS)
        cur.executemany('INSERT INTO "user" VALUES(?,?,?,?)', _ALL_USERS)
        cur.executemany("INSERT INTO plan VALUES(?,?,?,?,?)", _PLANS)
        cur.executemany("INSERT INTO subscription VALUES(?,?,?,?,?,?,?)", _ALL_SUBS)
        cur.executemany("INSERT INTO mrr_movement VALUES(?,?,?,?,?,?)", _ALL_MOVEMENTS)
        cur.executemany("INSERT INTO activity_event VALUES(?,?,?,?,?)", _ALL_EVENTS)
        cur.executemany("INSERT INTO invoice VALUES(?,?,?,?,?,?)", _ALL_INVOICES)
        cur.executemany("INSERT INTO revenue_recognition VALUES(?,?,?,?)", _ALL_RECOGNITIONS)

        conn.commit()

        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"FK violations after build: {violations}")

        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        print(f"Built {path} — {len(tables)} tables:")
        for t in tables:
            n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            print(f"  {t:<28} {n:>5} rows")
    finally:
        conn.close()
    return path


if __name__ == "__main__":
    build()
