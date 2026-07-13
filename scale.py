"""One million rows: the point where the database's shape starts to matter.

pipeline: 1M parquet rows ──► [scale] ──► sqlite + packed matrices ──► ms search

db.py's design — every vector a BLOB inside SQLite — is perfect at 14 rows
and wrong at a million: a brute-force scan would decode a million BLOBs per
query. At scale every real vector store splits the two jobs this file keeps
separate on disk:

  data/scale.sqlite      the RECORDS — name, caption, cafe, url. Row id i
                         here IS row i of every matrix below. SQLite is
                         superb at "give me rows 17, 40312, 998001".
  data/img_emb_f16.npy   the SCAN — one packed (N, 512) float16 matrix per
  data/txt_emb_f16.npy   tower, memory-mapped, so a query is chunked
                         matrix @ vector instead of a million BLOB decodes.

Same trick as db.py (raw little-endian floats), different layout: rows
together for lookups, columns of vectors together for scanning.

The three searches this enables, all in the SAME 512-d CLIP space the rest
of the repo (and the browser demo) lives in:

  brute   the exact answer: scan all N — the baseline everything is judged by
  ivf     ann.py's inverted-file index, industrial size: k-means the vectors
          into cells, scan only the few cells nearest the query
  int8    quantize.py's lesson applied: one shared max-abs scale, every
          value in a signed byte — half the bytes of f16. Too coarse on its
          own at 1M (CLIP has outlier dimensions that hog the byte's range),
          so it plays its industrial role: FETCH a cheap top-100, then
          re-score exactly — two-stage retrieval, the fix real systems use

Vectors come from Qdrant's public Wolt food dataset (1.75M dishes; image
embeddings precomputed with the SAME clip-ViT-B-32 checkpoint this repo
uses) — we take the first 1M. Text embeddings are computed HERE, by
embedder.py, from each dish's name: a million sentences through the text
tower of a laptop. Fused search then needs no third matrix at all —
score = (image·q + text·q) / 2, fusion.py's identity, applied lazily.

Run me:  python3 scale.py selftest                (synthetic, no model, CI-safe)
         python3 scale.py ingest                  (data/part-*.parquet -> the DB)
         python3 scale.py search "quattro formaggi pizza" --mode fused
         python3 scale.py bench --queries "sushi,burger,ramen"
"""
import argparse
import json
import os
import sqlite3
import time

import numpy as np

DATA = "data"
DB = os.path.join(DATA, "scale.sqlite")
IMG = os.path.join(DATA, "img_emb_f16.npy")
TXT = os.path.join(DATA, "txt_emb_f16.npy")
I8 = os.path.join(DATA, "img_emb_i8.npy")
I8S = os.path.join(DATA, "img_emb_i8_scale.npy")
IVF = os.path.join(DATA, "ivf.npz")
CHUNK = 131_072          # rows per scan chunk: ~256 MB of f32 at 512-d

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


# ------------------------------------------------------------------ scan --
def topk_merge(ids, scores, new_ids, new_scores, k):
    """Keep the best k of (old ∪ new) — how a chunked scan stays O(k) memory."""
    ids = np.concatenate([ids, new_ids])
    scores = np.concatenate([scores, new_scores])
    keep = np.argpartition(scores, -min(k, len(scores)))[-k:]
    order = keep[np.argsort(scores[keep])[::-1]]
    return ids[order], scores[order]


def scan(mats, q, k=10, rows=None, chunk=CHUNK):
    """Exact top-k over one pass of the memmapped matrices.

    mats = [(matrix, weight), ...] — fused search is just two towers with
    weight 1/2 each, summed per chunk: fusion.py's identity, never
    materialising a 1024-d matrix."""
    rows = rows if rows is not None else len(mats[0][0])
    ids = np.empty(0, dtype=np.int64)
    scores = np.empty(0, dtype=np.float32)
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        s = np.zeros(hi - lo, dtype=np.float32)
        for M, w in mats:
            s += w * (np.asarray(M[lo:hi], dtype=np.float32) @ q)
        ids, scores = topk_merge(ids, scores, np.arange(lo, hi), s, k)
    return ids, scores


# ------------------------------------------------------------------- ivf --
def ivf_train(M, rows, n_lists=1024, sample=100_000, iters=6, seed=0, log=print):
    """ann.py's index, sized for a million: k-means on a SAMPLE (fitting
    cells doesn't need every point), then every row joins its nearest cell.
    Lists are stored CSR-style — one row-id array sorted by cell, plus per-
    cell offsets — because a million tiny python lists is its own scan."""
    rng = np.random.default_rng(seed)
    S = np.asarray(M[np.sort(rng.choice(rows, min(sample, rows), replace=False))],
                   dtype=np.float32)
    C = S[rng.choice(len(S), n_lists, replace=False)]
    for it in range(iters):
        assign = np.argmax(S @ C.T, axis=1)
        for j in range(n_lists):
            members = S[assign == j]
            if len(members):
                v = members.mean(axis=0)
                C[j] = v / (np.linalg.norm(v) or 1.0)
            else:                       # empty cell: reseed on a random point
                C[j] = S[rng.integers(len(S))]
        log(f"  k-means iter {it + 1}/{iters}")
    assign = np.empty(rows, dtype=np.int32)
    for lo in range(0, rows, CHUNK):    # now every row picks its cell
        hi = min(lo + CHUNK, rows)
        assign[lo:hi] = np.argmax(np.asarray(M[lo:hi], dtype=np.float32) @ C.T, axis=1)
    order = np.argsort(assign, kind="stable").astype(np.int64)
    counts = np.bincount(assign, minlength=n_lists)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return C, order, starts


def ivf_search(q, mats, C, order, starts, k=10, probes=8):
    """Scan only the `probes` cells nearest the query. Returns
    (top-k ids, scores, rows actually scanned)."""
    near = np.argsort(q @ C.T)[::-1][:probes]
    cand = np.concatenate([order[starts[j]:starts[j + 1]] for j in near])
    s = np.zeros(len(cand), dtype=np.float32)
    for M, w in mats:
        s += w * (np.asarray(M[cand], dtype=np.float32) @ q)
    top = np.argsort(s)[::-1][:k]
    return cand[top], s[top], len(cand)


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


# ---------------------------------------------------------------- search --
def fetch(con, ids):
    marks = ",".join("?" * len(ids))
    rows = {r[0]: r for r in con.execute(
        f"SELECT id, name, caption, cafe, url FROM items WHERE id IN ({marks})",
        [int(i) for i in ids])}
    return [rows[int(i)] for i in ids]


def open_mats(mode):
    img = np.load(IMG, mmap_mode="r")
    if mode == "image":
        return [(img, 1.0)]
    txt = np.load(TXT, mmap_mode="r")
    return [(txt, 1.0)] if mode == "text" else [(img, 0.5), (txt, 0.5)]


def load_ivf():
    z = np.load(IVF)
    return z["C"], z["order"], z["starts"]


def cmd_search(args):
    from embedder import ClipEmbedder
    con = connect()
    rows = n_rows(con)
    q = ClipEmbedder().embed_texts([args.query])[0]
    mats = open_mats(args.mode)
    t0 = time.time()
    if args.ann:
        ids, scores, scanned = ivf_search(q, mats, *load_ivf(), k=args.k, probes=args.probes)
        how = f"ivf probes={args.probes}, scanned {scanned / rows:.1%}"
    else:
        ids, scores = scan(mats, q, k=args.k, rows=rows)
        how = f"exact scan of {rows:,}"
    ms = (time.time() - t0) * 1e3
    print(f"“{args.query}” — {args.mode} search, {how}: {ms:,.0f} ms\n")
    for (i, name, caption, cafe, url), s in zip(fetch(con, ids), scores):
        print(f"  {s:+.3f}  {name}" + (f"  — {cafe}" if cafe else ""))
        print(f"          {url}")


# ----------------------------------------------------------------- bench --
def cmd_bench(args):
    con = connect()
    rows = n_rows(con)
    img = np.load(IMG, mmap_mode="r")
    if args.queries:
        from embedder import ClipEmbedder
        Q = ClipEmbedder().embed_texts(args.queries.split(","))
    else:                                   # model-free: random unit queries
        rng = np.random.default_rng(0)
        Q = rng.normal(size=(8, 512)).astype(np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    print(f"{rows:,} rows on disk:")
    for p in (DB, IMG, TXT, I8, IVF):
        if os.path.exists(p):
            print(f"  {p:<24} {os.path.getsize(p) / 1e6:>8,.0f} MB")

    def med(f):                             # median-of-runs, warm cache
        ts = []
        for q in Q:
            t0 = time.time(); f(q); ts.append(time.time() - t0)
        return 1e3 * float(np.median(ts))

    scan([(img, 1.0)], Q[0], rows=rows)     # first touch: page the file in
    t_brute = med(lambda q: scan([(img, 1.0)], q, rows=rows))
    truth = [scan([(img, 1.0)], q, k=10, rows=rows)[0] for q in Q]
    print(f"\n  brute f16→f32   {t_brute:>8,.0f} ms/query   recall 1.00 (the truth)")

    if not (os.path.exists(I8) and os.path.exists(I8S)):
        # quantize.py's scheme, chunked: ONE shared scale so the largest
        # value maps to ±127 — naive 127·x wastes the byte on unit vectors,
        # whose 512-d components rarely leave ±0.1
        amax = max(float(np.abs(np.asarray(img[lo:lo + CHUNK], np.float32)).max())
                   for lo in range(0, rows, CHUNK))
        np.save(I8S, np.float32(amax / 127.0))
        i8 = np.lib.format.open_memmap(I8, mode="w+", dtype=np.int8, shape=(rows, 512))
        for lo in range(0, rows, CHUNK):
            hi = min(lo + CHUNK, rows)
            i8[lo:hi] = np.round(np.asarray(img[lo:hi], np.float32)
                                 / (amax / 127.0)).astype(np.int8)
        i8.flush()
    i8 = np.load(I8, mmap_mode="r")
    i8_scale = float(np.load(I8S))
    def scan8(q, k=10):
        return scan([(i8, i8_scale)], q, k=k, rows=rows)
    t8 = med(scan8)
    r8 = np.mean([len(set(scan8(q)[0]) & set(t)) / 10 for q, t in zip(Q, truth)])
    print(f"  brute int8      {t8:>8,.0f} ms/query   recall {r8:.2f}   (half the bytes)")

    def scan8_rerank(q, k=10):
        # two-stage: int8 nominates 100 candidates, f16 scores them exactly.
        # Quantization noise reshuffles the top-10 but rarely knocks a true
        # neighbour out of the top-100 — so the re-rank recovers the truth.
        cand = scan([(i8, i8_scale)], q, k=100, rows=rows)[0]
        s = np.asarray(img[cand], np.float32) @ q
        top = np.argsort(s)[::-1][:k]
        return cand[top], s[top]
    t8r = med(scan8_rerank)
    r8r = np.mean([len(set(scan8_rerank(q)[0]) & set(t)) / 10 for q, t in zip(Q, truth)])
    print(f"  int8 → re-rank  {t8r:>8,.0f} ms/query   recall {r8r:.2f}   (fetch cheap, score exactly)")

    if not os.path.exists(IVF):
        print("\n  building the ivf index (k-means on a 100k sample) …")
        t0 = time.time()
        C, order, starts = ivf_train(img, rows, log=lambda s: print(s, flush=True))
        np.savez(IVF, C=C, order=order, starts=starts)
        print(f"  built in {time.time() - t0:,.0f}s")
    C, order, starts = load_ivf()
    print("\n  probes   ms/query   recall@10   scanned")
    for probes in (1, 2, 4, 8, 16, 32):
        t = med(lambda q: ivf_search(q, [(img, 1.0)], C, order, starts, probes=probes))
        rec = np.mean([len(set(ivf_search(q, [(img, 1.0)], C, order, starts,
                                          probes=probes)[0]) & set(tr)) / 10
                       for q, tr in zip(Q, truth)])
        frac = np.mean([ivf_search(q, [(img, 1.0)], C, order, starts,
                                   probes=probes)[2] for q in Q]) / rows
        print(f"  {probes:>6}   {t:>8.1f}   {rec:>9.2f}   {frac:>6.1%}")
    print("\nsame story as ann.py, a thousandfold louder: scan ~1%, keep ~all of it.")


# -------------------------------------------------------------- selftest --
def cmd_selftest(args):
    """Everything above on synthetic clustered vectors — no model, no data
    files, no network. This is what CI runs."""
    import tempfile
    rng = np.random.default_rng(0)
    n, d = 20_000, 64
    centers = rng.normal(size=(64, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X = centers[rng.integers(0, 64, n)] + 0.2 * rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    q = X[rng.integers(n)]
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  {'pass' if cond else 'FAIL'} {msg}")
        ok &= bool(cond)

    with tempfile.TemporaryDirectory() as tmp:
        M = np.lib.format.open_memmap(os.path.join(tmp, "m.npy"), mode="w+",
                                      dtype=np.float16, shape=(n, d))
        M[:] = X.astype(np.float16)
        truth = np.argsort(X @ q)[::-1][:10]
        ids, scores = scan([(M, 1.0)], q, k=10, chunk=4096)
        check(set(ids) == set(truth), "chunked scan == naive argsort")
        check(np.all(np.diff(scores) <= 1e-6), "scores come back sorted")
        two = scan([(M, 0.5), (M, 0.5)], q, k=10, chunk=4096)[1]
        check(np.allclose(two, scores, atol=1e-3), "fused = weighted sum of towers")
        C, order, starts = ivf_train(M, n, n_lists=64, sample=5_000, iters=4,
                                     log=lambda s: None)
        check(starts[-1] == n and len(np.unique(order)) == n,
              "ivf lists partition every row exactly once")
        found, _, scanned = ivf_search(q, [(M, 1.0)], C, order, starts, probes=8)
        check(len(set(found) & set(truth)) >= 7, "ivf probes=8 keeps most of the truth")
        check(scanned < n / 2, "…while scanning under half the rows")
        s8 = float(np.abs(X).max()) / 127.0          # quantize.py's shared scale
        i8 = np.round(X / s8).astype(np.int8)
        f8 = np.argsort((i8.astype(np.float32) * s8) @ q)[::-1][:10]
        check(len(set(f8) & set(truth)) >= 9, "int8 (max-abs scale) keeps ≥9/10 of the truth")
        cand = np.argsort((i8.astype(np.float32) * s8) @ q)[::-1][:100]
        rr = cand[np.argsort(X[cand] @ q)[::-1][:10]]
        check(set(rr) == set(truth), "int8 top-100 → exact re-rank recovers 10/10")
    print("all scale.py checks passed" if ok else "some scale.py checks FAILED")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="parquet parts -> sqlite + packed matrices")
    p.add_argument("--parts", nargs="*", default=None)
    p.add_argument("--rows", type=int, default=1_000_000)
    p = sub.add_parser("search", help="live text query against the million")
    p.add_argument("query")
    p.add_argument("--mode", choices=("image", "text", "fused"), default="image")
    p.add_argument("--ann", action="store_true")
    p.add_argument("--probes", type=int, default=8)
    p.add_argument("-k", type=int, default=10)
    p = sub.add_parser("bench", help="sizes, latency, recall — the honest table")
    p.add_argument("--queries", help="comma-separated real queries (else random vectors)")
    sub.add_parser("selftest", help="synthetic end-to-end check, CI-safe")
    args = ap.parse_args()
    if args.cmd == "ingest":
        import glob
        parts = args.parts or sorted(glob.glob(os.path.join(DATA, "part-*.parquet")))
        ingest(parts, args.rows)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "bench":
        cmd_bench(args)
    else:
        cmd_selftest(args)


if __name__ == "__main__":
    main()
