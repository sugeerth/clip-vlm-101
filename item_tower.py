"""The item tower: agent-verified item embeddings, computed offline.

pipeline: image ──► agent (propose ⇄ critique) ──► [item_tower] ──► items.sqlite

Two-tower recommendation in one paragraph: a recommender scores
(user, item) pairs. Instead of one big model per pair, train/build TWO
encoders — an ITEM tower and a USER tower — that project into the same
vector space, so the score is a single dot product:

    ITEM tower  (this file)   image ─► agent-verified fused_emb, stored.
                              Runs OFFLINE, once per item, ahead of time.
    USER tower  (later)       user history ─► one vector, same space.
                              Runs ONLINE, once per request.
    serving                   score(user, item) = user_vec · item_vec
                              → rank all items with ONE matrix multiply.

Because the item side is finished offline, serving never touches an image
model — just this table. `item_matrix()` hands you the whole tower as one
(n, 1024) array, ready to dot against a user vector.

Every row was published by the embedding agent ONLY after its critic was
satisfied (see agent.py) — unverified drafts never reach this table.

Usage:
    python3 item_tower.py images/*.jpg        # build the tower
    python3 item_tower.py images/*.jpg --all  # publish even unsatisfied drafts
"""
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from db import from_blob, to_blob

# embedder.MODEL_ID, inlined so importing this file never pulls in torch
MODEL_ID = "openai/clip-vit-base-patch32"

ITEM_DB_PATH = "items.sqlite"

SCHEMA = """CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY,
    path       TEXT UNIQUE,
    caption    TEXT,
    labels     TEXT,   -- JSON {tag: probability} — the dynamic multi-label set
    item_emb   BLOB,   -- 1024 x float32 fused [image ; text] / sqrt(2)
    model      TEXT,   -- which encoder produced it (embeddings don't mix!)
    created_at TEXT
)"""


def connect(path: str = ITEM_DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    return con


def add_item(con, record, model: str = MODEL_ID):
    """Publish one agent-verified record into the tower."""
    con.execute(
        "INSERT OR REPLACE INTO items"
        " (path, caption, labels, item_emb, model, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (record["path"], record["caption"], json.dumps(record["labels"]),
         to_blob(record["fused_emb"]), model,
         datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    con.commit()


def all_items(con):
    rows = con.execute(
        "SELECT path, caption, labels, item_emb, model, created_at FROM items"
    ).fetchall()
    return [{"path": r[0], "caption": r[1], "labels": json.loads(r[2]),
             "item_emb": from_blob(r[3]), "model": r[4], "created_at": r[5]}
            for r in rows]


def item_matrix(con):
    """The whole tower at once: (paths, (n, 1024) float32 matrix).

    This is what a serving system loads. Ranking every item for a user is
    then `item_matrix @ user_vec` — one matrix multiply, no image model.
    """
    items = all_items(con)
    if not items:
        return [], np.empty((0, 0), dtype=np.float32)
    return ([it["path"] for it in items],
            np.stack([it["item_emb"] for it in items]))


if __name__ == "__main__":
    import argparse

    from agent import EmbeddingAgent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="item images to embed and publish")
    ap.add_argument("--db", default=ITEM_DB_PATH)
    ap.add_argument("--all", action="store_true",
                    help="also publish drafts the critic rejected")
    args = ap.parse_args()

    bot = EmbeddingAgent()
    con = connect(args.db)
    published = 0
    for path in args.paths:
        record, verdict = bot.run(path)
        if verdict.satisfied or args.all:
            add_item(con, record)
            published += 1
            print(f"  + {path}\n      labels  {record['labels']}\n      critic  {verdict}")
        else:
            print(f"  ! {path} NOT published — critic unsatisfied\n      critic  {verdict}")
    print(f"done — published {published}/{len(args.paths)}; "
          f"{len(all_items(con))} items in {args.db}")
