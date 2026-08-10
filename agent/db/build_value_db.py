"""Stage 2 value-linking fixture: a small B2B SaaS/CRM database with MORE than candidate_k (15)
tables, so candidate recall is no longer saturated (unlike the 8-table saas_metrics fixture).

It carries high-cardinality SEARCHABLE entity values (company / product / SKU / contract external
id / deal / campaign / vendor / plan / tracking no) and PII columns (contact & rep names, emails,
phones) that must never be indexed. Table names are deliberately distinct from the saas/demo
schemas so the shared COLUMN_POLICIES never collide. Deterministic seed (no randomness)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Referenced tables first so FKs resolve. 16 tables > candidate_k (15).
_SCHEMA = """
CREATE TABLE region_ref   (region_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE industry_ref (industry_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE team         (team_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE sales_rep    (rep_id INTEGER PRIMARY KEY, full_name TEXT NOT NULL, email TEXT NOT NULL,
                           team_id INTEGER REFERENCES team(team_id));
CREATE TABLE vendor       (vendor_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE plan_tier    (plan_id INTEGER PRIMARY KEY, name TEXT NOT NULL, monthly_price REAL NOT NULL);
CREATE TABLE company      (company_id INTEGER PRIMARY KEY, company_name TEXT NOT NULL,
                           region_id INTEGER REFERENCES region_ref(region_id),
                           industry_id INTEGER REFERENCES industry_ref(industry_id),
                           rep_id INTEGER REFERENCES sales_rep(rep_id));
CREATE TABLE person       (person_id INTEGER PRIMARY KEY, company_id INTEGER REFERENCES company(company_id),
                           full_name TEXT NOT NULL, email TEXT NOT NULL, phone TEXT);
CREATE TABLE catalog      (product_id INTEGER PRIMARY KEY, name TEXT NOT NULL, sku TEXT NOT NULL,
                           vendor_id INTEGER REFERENCES vendor(vendor_id));
CREATE TABLE agreement    (agreement_id INTEGER PRIMARY KEY, external_id TEXT NOT NULL,
                           company_id INTEGER REFERENCES company(company_id),
                           plan_id INTEGER REFERENCES plan_tier(plan_id),
                           start_on TEXT NOT NULL, end_on TEXT);
CREATE TABLE deal         (deal_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                           company_id INTEGER REFERENCES company(company_id), stage TEXT NOT NULL,
                           amount REAL NOT NULL);
CREATE TABLE campaign     (campaign_id INTEGER PRIMARY KEY, name TEXT NOT NULL, channel TEXT NOT NULL);
CREATE TABLE ticket       (ticket_id INTEGER PRIMARY KEY, company_id INTEGER REFERENCES company(company_id),
                           subject TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE ticket_note  (note_id INTEGER PRIMARY KEY, ticket_id INTEGER REFERENCES ticket(ticket_id),
                           body TEXT NOT NULL);
CREATE TABLE shipment     (shipment_id INTEGER PRIMARY KEY, tracking_no TEXT NOT NULL,
                           agreement_id INTEGER REFERENCES agreement(agreement_id));
CREATE TABLE usage_log    (event_id INTEGER PRIMARY KEY, company_id INTEGER REFERENCES company(company_id),
                           feature TEXT NOT NULL, occurred_on TEXT NOT NULL);
"""

# Deterministic seed. Distinctive searchable values (incl. a Chinese company name) + PII rows.
_SEED = """
INSERT INTO region_ref VALUES (1,'us-east'),(2,'eu-west'),(3,'apac');
INSERT INTO industry_ref VALUES (1,'software'),(2,'healthcare'),(3,'finance');
INSERT INTO team VALUES (1,'Enterprise'),(2,'SMB');
INSERT INTO sales_rep VALUES (1,'Alice Reynolds','alice@cadence.io',1),(2,'Bob Tan','bob@cadence.io',2);
INSERT INTO vendor VALUES (1,'Contoso Supplies'),(2,'Fabrikam Parts');
INSERT INTO plan_tier VALUES (1,'Starter',49.0),(2,'Professional',199.0),(3,'Enterprise',999.0);
INSERT INTO company VALUES
  (1,'Globex Corporation',1,1,1),
  (2,'Initech LLC',1,3,2),
  (3,'北京数据科技有限公司',3,1,1),
  (4,'Umbrella Health',2,2,1),(5,'Stark Industries',1,1,2),(6,'Wayne Enterprises',1,3,1),
  (7,'Cyberdyne Systems',3,1,2),(8,'Wonka Industries',2,2,1),(9,'Soylent Corp',1,3,2),
  (10,'Tyrell Corporation',3,1,1),(11,'Massive Dynamic',1,1,2),(12,'Aperture Science',2,1,1),
  (13,'Hooli',1,1,2),(14,'Pied Piper',1,1,1),(15,'Vandelay Industries',2,3,2),(16,'Gekko Capital',1,3,1);
INSERT INTO person VALUES
  (1,1,'John Smith','john.smith@globex.com','+1-202-555-0101'),
  (2,1,'Mary Jones','mary.jones@globex.com','+1-202-555-0102'),
  (3,2,'Peter Gibbons','peter@initech.com','+1-202-555-0103'),
  (4,3,'李伟','li.wei@bjdata.cn','+86-10-5555-0104');
INSERT INTO catalog VALUES
  (1,'Acme Widget','WGT-100',1),
  (2,'HyperGadget','HGT-200',2),(3,'DataSync Pro','DSP-300',1),(4,'CloudVault','CLV-400',1),
  (5,'StreamForge','STF-500',2),(6,'PixelPerfect','PXP-600',1),(7,'QuantumCore','QTC-700',2),
  (8,'NimbusEdge','NBE-800',1),(9,'IronGate','IRG-900',2),(10,'SwiftLedger','SWL-1000',1),
  (11,'BrightAnalytics','BRA-1100',2),(12,'SecureMailer','SCM-1200',1),(13,'FleetTracker','FLT-1300',2),
  (14,'OmniDash','OMD-1400',1);
INSERT INTO agreement VALUES
  (1,'CT-2025-0042',1,3,'2025-01-01','2025-12-31'),
  (2,'CT-2025-0099',3,2,'2025-02-01','2026-01-31'),(3,'CT-2025-0100',2,1,'2025-03-01',NULL),
  (4,'CT-2025-0101',4,2,'2025-01-15',NULL),(5,'CT-2025-0102',5,3,'2025-02-01',NULL),
  (6,'CT-2025-0103',6,2,'2025-02-10',NULL),(7,'CT-2025-0104',7,1,'2025-03-01',NULL),
  (8,'CT-2025-0105',8,2,'2025-03-15',NULL),(9,'CT-2025-0106',9,3,'2025-04-01',NULL),
  (10,'CT-2025-0107',10,2,'2025-04-10',NULL),(11,'CT-2025-0108',11,1,'2025-05-01',NULL),
  (12,'CT-2025-0109',12,2,'2025-05-15',NULL),(13,'CT-2025-0110',13,3,'2025-06-01',NULL),
  (14,'CT-2025-0111',14,2,'2025-06-10',NULL);
INSERT INTO deal VALUES
  (1,'Globex Renewal 2025',1,'won',120000.0),
  (2,'Initech Expansion',2,'open',45000.0),
  (3,'北京数据科技 Upsell',3,'open',60000.0);
INSERT INTO campaign VALUES (1,'Spring Launch','email'),(2,'Webinar Series','webinar');
INSERT INTO ticket VALUES
  (1,1,'Login issue','open'),
  (2,1,'Billing question','closed'),
  (3,2,'API error','open'),
  (4,3,'Feature request','open');
INSERT INTO ticket_note VALUES (1,1,'Investigating the SSO callback.'),(2,3,'Rate limit hit.');
INSERT INTO shipment VALUES (1,'TRK-77-ABCD',1),(2,'TRK-88-WXYZ',2);
INSERT INTO usage_log VALUES
  (1,1,'export','2025-06-01'),(2,1,'api_call','2025-06-02'),(3,2,'export','2025-06-03');
"""


def build(db_path: str | Path) -> str:
    path = Path(db_path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        conn.executescript(_SEED)
        conn.commit()
    finally:
        conn.close()
    return str(path)
