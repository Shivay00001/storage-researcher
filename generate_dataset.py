"""
generate_dataset.py — builds a representative "large database" extract to
run the storage-researcher engine against. Mixed column types on purpose
(sequential id, near-monotonic timestamp, low-cardinality categoricals,
free-text, floats, and a small embedding vector per row) so every block in
blocks.py has a column it's actually good at, and at least one it isn't —
an evaluator that only ever sees favorable data can't tell a real win from
a lucky one.
"""
import random
import time

CATEGORIES = ["electronics", "apparel", "grocery", "home", "books", "toys",
              "sports", "beauty", "automotive", "garden", "office", "pet",
              "health", "jewelry", "music", "software", "furniture",
              "outdoor", "baby", "industrial"]
STATUSES = ["active", "pending", "closed"]
WORDS = ("order shipment delayed customer refund processed warehouse "
         "restock priority standard express region north south east west "
         "central invoice batch reconciled pending review flagged manual "
         "audit passed threshold exceeded normal range").split()


def make_row(i: int, rnd: random.Random, base_ts: int) -> dict:
    return {
        "id": i,
        "ts": base_ts + i * 60 + rnd.randint(-5, 5),  # near-monotonic, ~1/min
        "category": rnd.choice(CATEGORIES),
        "status": STATUSES[0] if rnd.random() < 0.7 else rnd.choice(STATUSES),
        "description": " ".join(rnd.choices(WORDS, k=rnd.randint(4, 9))),
        "value": round(rnd.gauss(500, 180), 2),
        "embedding": [rnd.gauss(0, 1) for _ in range(16)],
    }


def generate(n_rows: int = 300_000, seed: int = 42) -> dict:
    """Returns column-major dict: {column_name: [values...]}."""
    rnd = random.Random(seed)
    base_ts = int(time.time()) - n_rows * 60
    cols = {"id": [], "ts": [], "category": [], "status": [],
            "description": [], "value": [], "embedding": []}
    for i in range(n_rows):
        row = make_row(i, rnd, base_ts)
        for k, v in row.items():
            cols[k].append(v)
    return cols


if __name__ == "__main__":
    data = generate(1000)
    print("Sample row 0:", {k: v[0] for k, v in data.items()})
    print("Columns:", list(data.keys()))
    print("Row count:", len(data["id"]))
