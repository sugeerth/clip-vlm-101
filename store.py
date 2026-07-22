"""The database at scale: records in SQLite, vectors in packed memmaps.

pipeline: parquet rows ──► [store] ──► sqlite (records) + f16 memmaps (vectors)

db.py keeps every vector a BLOB inside SQLite — perfect at 14 rows, wrong at a
million (a scan would decode a million BLOBs per query). At scale the two jobs
split, the way every real vector store splits them:

  data/scale.sqlite      the RECORDS — name, caption, cafe, url. Row id i here
                         IS row i of every matrix below; SQLite is superb at
                         "give me rows 17, 40312, 998001".
  data/img_emb_f16.npy   the SCAN — one packed (N, 512) float16 matrix per
  data/txt_emb_f16.npy   tower, memory-mapped, so a query is a chunked
                         matrix @ vector, not N BLOB decodes.

Same raw-little-endian-floats trick as db.py, different layout: rows together
for lookups, columns of vectors together for scanning. Everything else in the
scale engine (scan.py, ivf.py, intq.py, opq.py, binary.py) reads these.
"""
import json
import os
import sqlite3
import time

import numpy as np

DATA = "data"
DB = os.path.join(DATA, "scale.sqlite")
IMG = os.path.join(DATA, "img_emb_f16.npy")
TXT = os.path.join(DATA, "txt_emb_f16.npy")
CHUNK = 131_072          # rows per scan/build chunk: ~256 MB of f32 at 512-d

SCHEMA = """CREATE TABLE IF NOT EXISTS items (
    id      INTEGER PRIMARY KEY,   -- row i here = row i in every matrix
    name    TEXT,
    caption TEXT,
    cafe    TEXT,
    url     TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);"""


def connect(path=DB):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def n_rows(con) -> int:
    row = con.execute("SELECT value FROM meta WHERE key='rows'").fetchone()
    return int(row[0]) if row else 0


def fetch(con, ids):
    """The lookup half: given row ids from a scan, pull their records back in
    that order. SQLite's `IN (...)` is the fast path SQLite exists for."""
    marks = ",".join("?" * len(ids))
    rows = {r[0]: r for r in con.execute(
        f"SELECT id, name, caption, cafe, url FROM items WHERE id IN ({marks})",
        [int(i) for i in ids])}
    return [rows[int(i)] for i in ids]


def open_mats(mode):
    """The tower(s) a search scores, memory-mapped. Fused never materialises a
    1024-d matrix — it's two towers at weight 1/2, summed per chunk."""
    img = np.load(IMG, mmap_mode="r")
    if mode == "image":
        return [(img, 1.0)]
    txt = np.load(TXT, mmap_mode="r")
    return [(txt, 1.0)] if mode == "text" else [(img, 0.5), (txt, 0.5)]


# ---------------------------------------------------------------- ingest --
def parse_vec(v):
    return np.asarray(json.loads(v) if isinstance(v, str) else v, dtype=np.float32)


def clean(r, skipped):
    """One parquet record -> (metadata row, unit vector, name) or None."""
    try:
        v = parse_vec(r["vector"])
        if v.shape != (512,) or not np.isfinite(v).all():
            raise ValueError
    except Exception:
        skipped.append(1)
        return None
    cafe = r.get("cafe") or {}
    if isinstance(cafe, str):
        try:
            cafe = json.loads(cafe)
        except Exception:
            cafe = {}
    name = (r.get("name") or "").strip() or (r.get("description") or "food")[:60]
    rec = (name, (r.get("description") or "")[:300],
           (cafe.get("name") or "")[:80], r.get("image") or "")
    return rec, v / (np.linalg.norm(v) or 1.0), name   # dataset ships unnormalized


def ingest(parts, rows_wanted, batch=512):
    """parquet parts -> the DB: image vectors straight in, text vectors computed
    here (one dish name at a time through the text tower). Row id i lands in row
    i of both memmaps and the items table, so everything downstream can assume it."""
    import pyarrow.parquet as pq
    from embedder import ClipEmbedder

    os.makedirs(DATA, exist_ok=True)
    total = min(rows_wanted, sum(pq.ParquetFile(p).metadata.num_rows for p in parts))
    print(f"{len(parts)} parts, ingesting up to {total:,} rows")
    img = np.lib.format.open_memmap(IMG, mode="w+", dtype=np.float16, shape=(total, 512))
    txt = np.lib.format.open_memmap(TXT, mode="w+", dtype=np.float16, shape=(total, 512))
    emb = ClipEmbedder()
    con = connect()
    con.execute("DELETE FROM items")
    at, t0, skipped = 0, time.time(), []
    for p in parts:
        for rows in pq.ParquetFile(p).iter_batches(batch_size=10_000):
            got = [c for c in (clean(r, skipped) for r in rows.to_pylist()) if c]
            got = got[:total - at]
            if not got:
                continue
            recs, vecs, names = zip(*got)
            img[at:at + len(vecs)] = np.stack(vecs).astype(np.float16)
            for lo in range(0, len(names), batch):         # a million names, batched
                T = emb.embed_texts(list(names[lo:lo + batch]))
                txt[at + lo:at + lo + len(T)] = T.astype(np.float16)
            con.executemany(
                "INSERT INTO items (id, name, caption, cafe, url) VALUES (?, ?, ?, ?, ?)",
                [(at + i, *r) for i, r in enumerate(recs)])
            at += len(recs)
            con.execute("INSERT OR REPLACE INTO meta VALUES ('rows', ?)", (str(at),))
            con.commit()
            if at >= total:
                break
        rate = at / (time.time() - t0)
        print(f"  {os.path.basename(p)}: {at:,}/{total:,} rows  ({rate:,.0f} rows/s)")
        if at >= total:
            break
    img.flush(); txt.flush()
    print(f"done: {at:,} rows in {time.time() - t0:,.0f}s, {len(skipped)} skipped"
          f"\n  {DB}  {os.path.getsize(DB) / 1e6:,.0f} MB"
          f"\n  {IMG}  {os.path.getsize(IMG) / 1e9:.2f} GB"
          f"\n  {TXT}  {os.path.getsize(TXT) / 1e9:.2f} GB")
