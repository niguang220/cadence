"""Build the demo SQLite database (Chinook-style media store, 14 tables).

Reproducible: deterministic seed, so the same DB (and therefore the same
evaluation ground truth) is produced every run. Data volume is deliberately
non-trivial so that a wrong SQL query does not accidentally match a gold result.

    python -m agent.db.build_demo_db          # writes agent/db/demo.db
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "demo.db"

SCHEMA = """
CREATE TABLE artist (
    artist_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);
CREATE TABLE genre (
    genre_id    INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);
CREATE TABLE media_type (
    media_type_id INTEGER PRIMARY KEY,
    name          TEXT NOT NULL
);
CREATE TABLE album (
    album_id    INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    artist_id   INTEGER NOT NULL REFERENCES artist(artist_id)
);
CREATE TABLE track (
    track_id      INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    album_id      INTEGER REFERENCES album(album_id),          -- NULL = single
    genre_id      INTEGER REFERENCES genre(genre_id),
    media_type_id INTEGER NOT NULL REFERENCES media_type(media_type_id),
    unit_price    REAL NOT NULL,
    milliseconds  INTEGER NOT NULL
);
CREATE TABLE playlist (
    playlist_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);
CREATE TABLE playlist_track (
    playlist_id INTEGER NOT NULL REFERENCES playlist(playlist_id),
    track_id    INTEGER NOT NULL REFERENCES track(track_id),
    PRIMARY KEY (playlist_id, track_id)
);
CREATE TABLE employee (
    employee_id INTEGER PRIMARY KEY,
    first_name  TEXT NOT NULL,
    last_name   TEXT NOT NULL,
    title       TEXT,
    reports_to  INTEGER REFERENCES employee(employee_id),
    hire_date   TEXT
);
CREATE TABLE customer (
    customer_id    INTEGER PRIMARY KEY,
    first_name     TEXT NOT NULL,
    last_name      TEXT NOT NULL,
    country        TEXT,
    email          TEXT,
    support_rep_id INTEGER REFERENCES employee(employee_id)
);
CREATE TABLE invoice (
    invoice_id      INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customer(customer_id),
    invoice_date    TEXT NOT NULL,
    billing_country TEXT,
    total           REAL NOT NULL
);
CREATE TABLE invoice_line (
    invoice_line_id INTEGER PRIMARY KEY,
    invoice_id      INTEGER NOT NULL REFERENCES invoice(invoice_id),
    track_id        INTEGER NOT NULL REFERENCES track(track_id),
    unit_price      REAL NOT NULL,
    quantity        INTEGER NOT NULL
);
CREATE TABLE supplier (
    supplier_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT
);
CREATE TABLE track_supplier (
    track_id    INTEGER NOT NULL REFERENCES track(track_id),
    supplier_id INTEGER NOT NULL REFERENCES supplier(supplier_id),
    cost        REAL NOT NULL,
    PRIMARY KEY (track_id, supplier_id)
);
CREATE TABLE review (
    review_id   INTEGER PRIMARY KEY,
    track_id    INTEGER NOT NULL REFERENCES track(track_id),
    customer_id INTEGER NOT NULL REFERENCES customer(customer_id),
    rating      INTEGER NOT NULL,
    comment     TEXT
);
"""

GENRES = ["Rock", "Jazz", "Classical", "Pop", "Metal", "Hip-Hop", "Electronic", "Blues"]
MEDIA = ["MPEG audio", "AAC audio", "Protected AAC", "Purchased AAC"]
COUNTRIES = ["USA", "Germany", "Brazil", "Canada", "France", "UK", "Singapore", "China", "India", "Japan"]


def _seed(conn: sqlite3.Connection) -> None:
    rng = random.Random(42)
    cur = conn.cursor()

    cur.executemany("INSERT INTO genre(genre_id,name) VALUES(?,?)",
                    [(i + 1, g) for i, g in enumerate(GENRES)])
    cur.executemany("INSERT INTO media_type(media_type_id,name) VALUES(?,?)",
                    [(i + 1, m) for i, m in enumerate(MEDIA)])

    # employees: a small org with a reports_to hierarchy (employee 1 is the top)
    employees = [(1, "Andrew", "Adams", "General Manager", None, "2023-01-10")]
    for i in range(2, 9):
        mgr = 1 if i <= 3 else rng.randint(2, 3)
        employees.append((i, f"Emp{i}", f"Last{i}", "Sales Support Agent", mgr, "2023-06-01"))
    cur.executemany("INSERT INTO employee VALUES(?,?,?,?,?,?)", employees)

    cur.executemany("INSERT INTO artist(artist_id,name) VALUES(?,?)",
                    [(i, f"Artist {i}") for i in range(1, 26)])

    albums = [(i, f"Album {i}", rng.randint(1, 25)) for i in range(1, 61)]
    cur.executemany("INSERT INTO album VALUES(?,?,?)", albums)

    # tracks: ~5% are singles (album_id NULL) to exercise LEFT JOIN / NULL handling
    tracks = []
    for i in range(1, 301):
        album_id = None if rng.random() < 0.05 else rng.randint(1, 60)
        tracks.append((i, f"Track {i}", album_id, rng.randint(1, len(GENRES)),
                       rng.randint(1, len(MEDIA)), round(rng.choice([0.99, 1.29]), 2),
                       rng.randint(120_000, 360_000)))
    cur.executemany("INSERT INTO track VALUES(?,?,?,?,?,?,?)", tracks)

    cur.executemany("INSERT INTO playlist(playlist_id,name) VALUES(?,?)",
                    [(i, f"Playlist {i}") for i in range(1, 11)])
    pt = {(rng.randint(1, 10), rng.randint(1, 300)) for _ in range(700)}
    cur.executemany("INSERT INTO playlist_track VALUES(?,?)", list(pt))

    customers = []
    for i in range(1, 61):
        customers.append((i, f"Cust{i}", f"Last{i}", rng.choice(COUNTRIES),
                          f"cust{i}@example.com", rng.randint(2, 8)))
    cur.executemany("INSERT INTO customer VALUES(?,?,?,?,?,?)", customers)

    suppliers = [(i, f"Supplier {i}", rng.choice(COUNTRIES)) for i in range(1, 7)]
    cur.executemany("INSERT INTO supplier VALUES(?,?,?)", suppliers)
    ts = {(rng.randint(1, 300), rng.randint(1, 6)) for _ in range(400)}
    cur.executemany("INSERT INTO track_supplier VALUES(?,?,?)",
                    [(t, s, round(rng.uniform(0.3, 0.8), 2)) for t, s in ts])

    # invoices + lines: total is the sum of its lines, so aggregates are checkable
    start = date(2024, 1, 1)
    line_id = 0
    for inv_id in range(1, 201):
        cust = rng.randint(1, 60)
        inv_date = (start + timedelta(days=rng.randint(0, 540))).isoformat()
        n_lines = rng.randint(1, 5)
        total = 0.0
        lines = []
        for _ in range(n_lines):
            line_id += 1
            track_id = rng.randint(1, 300)
            price = rng.choice([0.99, 1.29])
            qty = rng.randint(1, 3)
            total += price * qty
            lines.append((line_id, inv_id, track_id, price, qty))
        billing = rng.choice(COUNTRIES)
        cur.execute("INSERT INTO invoice VALUES(?,?,?,?,?)",
                    (inv_id, cust, inv_date, billing, round(total, 2)))
        cur.executemany("INSERT INTO invoice_line VALUES(?,?,?,?,?)", lines)

    sample_comments = ["Great track", "Loved it", "Not my taste", "Solid", "Meh", "On repeat"]
    reviews = []
    for i in range(1, 151):
        # ~1/3 carry a comment, ~2/3 NULL: COUNT(*) vs COUNT(comment) still differ,
        # but comment-based questions aren't degenerate.
        comment = rng.choice(sample_comments) if i % 3 == 0 else None
        reviews.append((i, rng.randint(1, 300), rng.randint(1, 60),
                        rng.randint(1, 5), comment))
    cur.executemany("INSERT INTO review VALUES(?,?,?,?,?)", reviews)

    conn.commit()


def _insert_edge_fixtures(conn: sqlite3.Connection) -> None:
    """Pinned, named edge-case rows so hard cases are deterministic and readable
    in bad-case analysis (not random byproducts). IDs offset to 9000+ to avoid
    colliding with the random seed."""
    cur = conn.cursor()
    # No-invoice customer -> LEFT JOIN customer->invoice must keep zero-spenders.
    cur.execute("INSERT INTO customer VALUES(?,?,?,?,?,?)",
                (9001, "Noinvoice", "Customer", "Singapore", "noinvoice@example.com", 2))
    # No-track album -> LEFT JOIN album->track must keep empty albums.
    cur.execute("INSERT INTO album VALUES(?,?,?)", (9001, "Empty Album (no tracks)", 1))
    # Zero-sales track -> LEFT JOIN track->invoice_line must keep unsold tracks.
    cur.execute("INSERT INTO track VALUES(?,?,?,?,?,?,?)",
                (9001, "Zero Sales Track", 2, 1, 1, 0.99, 200000))
    # Null-album single -> NULL album_id / LEFT JOIN.
    cur.execute("INSERT INTO track VALUES(?,?,?,?,?,?,?)",
                (9002, "Null Album Single", None, 1, 1, 0.99, 180000))
    # Fanout-trap customer: 2 invoices x 2 lines. Correct spend is 4.00; joining
    # invoice to its lines and SUM(invoice.total) double-counts to 8.00.
    cur.execute("INSERT INTO customer VALUES(?,?,?,?,?,?)",
                (9002, "Fanout", "Trap", "China", "fanout@example.com", 2))
    line_id = 900000
    for inv_id in (9001, 9002):
        cur.execute("INSERT INTO invoice VALUES(?,?,?,?,?)",
                    (inv_id, 9002, "2024-06-01", "China", 2.00))
        for _ in range(2):
            line_id += 1
            cur.execute("INSERT INTO invoice_line VALUES(?,?,?,?,?)",
                        (line_id, inv_id, 1, 1.00, 1))
    # Distinct-trap playlist: 3 tracks by ONE artist. COUNT(DISTINCT artist)=1 but
    # COUNT(track)=3, so "how many artists in this playlist" needs DISTINCT.
    cur.execute("INSERT INTO album VALUES(?,?,?)", (9002, "Distinct Trap Album", 5))
    for tid in (9003, 9004, 9005):
        cur.execute("INSERT INTO track VALUES(?,?,?,?,?,?,?)",
                    (tid, f"Trap Track {tid}", 9002, 1, 1, 0.99, 150000))
    cur.execute("INSERT INTO playlist VALUES(?,?)", (9001, "Distinct Trap Playlist"))
    for tid in (9003, 9004, 9005):
        cur.execute("INSERT INTO playlist_track VALUES(?,?)", (9001, tid))
    # Country-mismatch customer: home country (Singapore) != an invoice's billing
    # country (China) -> pins the customer.country vs invoice.billing_country trap.
    cur.execute("INSERT INTO customer VALUES(?,?,?,?,?,?)",
                (9003, "CountryMismatch", "Customer", "Singapore", "mismatch@example.com", 2))
    cur.execute("INSERT INTO invoice VALUES(?,?,?,?,?)", (9003, 9003, "2024-06-02", "China", 0.99))
    cur.execute("INSERT INTO invoice_line VALUES(?,?,?,?,?)", (900100, 9003, 1, 0.99, 1))
    # Supplier-fanout track: 2 suppliers but 1 sale. Correct sales count is 1;
    # joining track->track_supplier before counting inflates it to 2.
    cur.execute("INSERT INTO track VALUES(?,?,?,?,?,?,?)",
                (9006, "Supplier Fanout Track", 2, 1, 1, 0.99, 160000))
    cur.execute("INSERT INTO track_supplier VALUES(?,?,?)", (9006, 1, 0.40))
    cur.execute("INSERT INTO track_supplier VALUES(?,?,?)", (9006, 2, 0.45))
    cur.execute("INSERT INTO invoice VALUES(?,?,?,?,?)", (9004, 1, "2024-06-03", "USA", 0.99))
    cur.execute("INSERT INTO invoice_line VALUES(?,?,?,?,?)", (900101, 9004, 9006, 0.99, 1))
    conn.commit()


def build(db_path: Path = DB_PATH, overwrite: bool = True) -> Path:
    """Build the demo DB at ``db_path``. Destructive: replaces any existing file.
    Pass ``overwrite=False`` to refuse rather than overwrite an existing file."""
    if db_path.exists():
        if not overwrite:
            raise FileExistsError(f"{db_path} exists; pass overwrite=True to replace it")
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")  # SQLite doesn't enforce FKs by default
        conn.executescript(SCHEMA)
        _seed(conn)
        _insert_edge_fixtures(conn)
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"FK violations after build: {violations}")
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        print(f"Built {db_path} — {len(tables)} tables:")
        for t in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<16} {n:>5} rows")
    finally:
        conn.close()
    return db_path


if __name__ == "__main__":
    build()
