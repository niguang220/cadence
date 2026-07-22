"""Human-written table/column descriptions for the demo database.

The schema retriever and the SQL-generation prompt use these so the agent links
natural-language terms to the right tables/columns (schema linking). Kept
separate from the build script because they are documentation, not data.
"""

TABLE_DESCRIPTIONS: dict[str, str] = {
    "artist": "Music artists / bands.",
    "album": "Albums, each belonging to one artist.",
    "genre": "Music genres (Rock, Jazz, ...).",
    "media_type": "Track media format (MPEG, AAC, ...).",
    "track": "Individual songs; the sellable catalog item. Has a unit price.",
    "playlist": "User-curated playlists.",
    "playlist_track": "Bridge: which tracks are in which playlist (many-to-many).",
    "customer": "Customers who place orders. Has a home/profile country and an assigned support rep.",
    "employee": "Staff; support reps serve customers. Self-referencing reports_to hierarchy.",
    "invoice": "A customer order/receipt with a date, billing country and total amount.",
    "invoice_line": "Line items of an invoice: a track, its unit price and quantity.",
    "supplier": "Suppliers that provide tracks to the store.",
    "track_supplier": "Bridge: which supplier provides which track, and at what cost (many-to-many).",
    "review": "Customer star ratings (1-5) and comments on tracks.",
}

# Only non-obvious / business-meaningful columns are described.
COLUMN_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "track": {
        "unit_price": "Catalog list price of the track.",
        "milliseconds": "Track length in milliseconds.",
        "album_id": "FK -> album. NULL for singles not tied to an album.",
    },
    "customer": {
        "country": "Customer's own country (may differ from an invoice's billing country).",
        "support_rep_id": "FK -> employee; the support rep assigned to this customer.",
    },
    "employee": {
        "reports_to": "FK -> employee; this employee's manager. NULL for the top of the org.",
        "title": "Job title (e.g. 'Sales Support Agent').",
    },
    "invoice": {
        "billing_country": "Country recorded on the invoice's billing address (may differ from the customer's own country).",
        "total": "Sum of the invoice's line items.",
        "invoice_date": "When the order was placed.",
    },
    "invoice_line": {
        "unit_price": "Price charged for this line (may differ from current catalog price).",
        "quantity": "Number of units bought on this line.",
    },
    "track_supplier": {"cost": "What the store pays the supplier per track."},
    "review": {"rating": "Star rating 1-5."},
}

# Data-governance policy for prompt rendering and SQL execution.
#
# v1 is intentionally conservative: PII columns are hidden from schema prompts
# and cannot be selected/filtered/aggregated in generated SQL. Aggregates over
# PII-derived values (email domain, hashes, etc.) are deferred to a later policy.
COLUMN_POLICIES: dict[str, dict[str, str]] = {
    "customer": {
        "first_name": "pii",
        "last_name": "pii",
        "email": "pii",
    },
    "employee": {
        "first_name": "pii",
        "last_name": "pii",
    },
    "user": {                # SaaS schema: individual login emails are PII
        "email": "pii",
    },
}
