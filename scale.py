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
         python3 scale.py serve                   (the million, live at localhost)

serve is the punchline: both towers promoted from memmap to RAM (the scan
was I/O-bound — paging + f16→f32 casting was ~90% of the 868 ms; in RAM the
same exact scan is a single matmul), IVF on top when you want milliseconds,
queries ensembled over two phrasings (ensemble.py's lesson), fused ranking
by default, and a one-file UI with the latency printed on every answer.
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
I8S = os.path.join(DATA, "img_emb_i8_scale.npy")   # (d+1,): per-dim scale then row count
IVF = os.path.join(DATA, "ivf.npz")
OPQC = os.path.join(DATA, "img_opq_codes.npy")     # (rows, m) uint8 — 64 bytes/vector
OPQZ = os.path.join(DATA, "img_opq.npz")           # rotation R, codebooks, row count
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


def scan_ram(mats, q, k=10):
    """The same exact top-k as scan(), for matrices already promoted to f32
    RAM: one matmul per tower, one argpartition. No chunks — RAM needs none."""
    s = mats[0][1] * (mats[0][0] @ q)
    for M, w in mats[1:]:
        s += w * (M @ q)
    keep = np.argpartition(s, -min(k, len(s)))[-k:]
    order = keep[np.argsort(s[keep])[::-1]]
    return order.astype(np.int64), s[order]


def gpu_promote(mats):
    """One more rung of the storage hierarchy: RAM -> the GPU. On Apple
    silicon this is nearly free (unified memory), and the exact scan becomes
    a device matmul + topk. Returns None when there's no usable device."""
    import torch
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        return None
    return dev, [(torch.from_numpy(np.ascontiguousarray(M)).to(dev), w)
                 for M, w in mats]


def scan_gpu(gpu, q, k=10):
    """scan_ram's device twin — identical answers, another order of magnitude."""
    import torch
    dev, gmats = gpu
    qt = torch.from_numpy(np.ascontiguousarray(q)).to(dev)
    s = gmats[0][1] * (gmats[0][0] @ qt)
    for M, w in gmats[1:]:
        s = s + w * (M @ qt)
    scores, ids = torch.topk(s, min(k, s.shape[0]))
    return ids.cpu().numpy().astype(np.int64), scores.cpu().numpy()


def ensemble_query(embed, text):
    """ensemble.py's lesson on the QUERY side: average two phrasings of the
    same intent, renormalise. Worth a few recall points for free."""
    E = embed([text, f"a photo of {text}"])
    v = E.mean(axis=0)
    return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)


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


# ------------------------------------------------- int8 (per-dimension) --
# quantize.py used ONE shared scale for the whole matrix. That wastes the byte
# on CLIP: a few outlier dimensions are ~10× wider than the rest, so one global
# scale sizes the byte for the outliers and quantizes everything else to a
# pinhead. A PER-DIMENSION scale gives every column its own ±127 range — same
# one byte per value, measurably better recall (0.85 → 0.87 at a million).
def build_i8(img, rows, log=print):
    """Write the int8 image tower + its per-dim scales, keyed to the row count
    so a grown corpus rebuilds it. scale[d] = max|col d| / 127."""
    amax = np.zeros(img.shape[1], dtype=np.float32)
    for lo in range(0, rows, CHUNK):
        amax = np.maximum(amax, np.abs(np.asarray(img[lo:lo + CHUNK], np.float32)).max(0))
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    i8 = np.lib.format.open_memmap(I8, mode="w+", dtype=np.int8, shape=(rows, 512))
    for lo in range(0, rows, CHUNK):
        hi = min(lo + CHUNK, rows)
        i8[lo:hi] = np.round(np.asarray(img[lo:hi], np.float32) / scale).astype(np.int8)
    i8.flush()
    np.save(I8S, np.concatenate([scale, [float(rows)]]).astype(np.float64))
    log(f"  int8 per-dim: {I8} ({os.path.getsize(I8) / 1e6:,.0f} MB)")


def i8_stale(rows):
    if not (os.path.exists(I8) and os.path.exists(I8S)):
        return True
    meta = np.atleast_1d(np.load(I8S))
    return meta.size != 513 or int(meta[-1]) != rows


def load_i8():
    """Returns (i8 memmap, per-dim scale (d,)). Dequant folds into the query:
    x·q ≈ Σ_d i8[d]·scale[d]·q[d] = i8 · (scale ⊙ q), so the scan stays integer
    on the matrix and the ranking is exact up to rounding."""
    meta = np.load(I8S)
    return np.load(I8, mmap_mode="r"), meta[:512].astype(np.float32)


def i8_search(i8, scale, q, k=10, rows=None):
    return scan([(i8, 1.0)], (scale * q).astype(np.float32), k=k, rows=rows)


def i8_rerank(i8, scale, img, q, k=10, cand=100, rows=None):
    """int8 nominates a cheap top-`cand`; f16 scores exactly. The workhorse of
    real systems: quantization reshuffles a tight top-k but rarely evicts a true
    neighbour from a top-100, so the exact re-rank recovers the truth."""
    pool = i8_search(i8, scale, q, k=cand, rows=rows)[0]
    e = np.asarray(img[pool], np.float32) @ q
    top = np.argsort(e)[::-1][:k]
    return pool[top], e[top]


# -------------------------------------------------- OPQ (64 bytes/vector) --
# opq.py, sized for the million: the extreme end of the storage hierarchy — one
# byte per 8-dim block, 64 bytes for a 512-d vector (32× smaller than float32).
# On its own it's a coarse filter (recall ~0.56); its real job is stage one of a
# two-stage search — nominate a few hundred candidates for an exact re-rank.
def build_opq(img, rows, sample=150_000, log=print):
    from opq import opq_encode, opq_train
    stride = max(1, rows // sample)
    tr = np.asarray(img[:rows][::stride][:sample], dtype=np.float32)   # only real rows
    log(f"  training OPQ on {len(tr):,} rows (rotation + 64 codebooks) …")
    R, books = opq_train(tr, log=lambda s: log(s))
    codes = np.lib.format.open_memmap(OPQC, mode="w+", dtype=np.uint8, shape=(rows, books.shape[0]))
    for lo in range(0, rows, CHUNK):
        hi = min(lo + CHUNK, rows)
        codes[lo:hi] = opq_encode(np.asarray(img[lo:hi], np.float32), R, books)
    codes.flush()
    np.savez(OPQZ, R=R, books=books, rows=rows)
    log(f"  OPQ: {OPQC} ({os.path.getsize(OPQC) / 1e6:,.0f} MB) + rotation/codebooks")


def opq_stale(rows):
    if not (os.path.exists(OPQC) and os.path.exists(OPQZ)):
        return True
    with np.load(OPQZ) as z:
        return "rows" not in z or int(z["rows"]) != rows


def load_opq():
    z = np.load(OPQZ)
    return z["R"], z["books"], np.load(OPQC, mmap_mode="r")


def opq_rerank(codes, R, books, img, q, k=10, cand=400, rows=None):
    """Stage one: OPQ's table-lookup scan over every row (no multiplies) picks a
    shortlist. Stage two: score that shortlist exactly on f16. Turns 64 bytes a
    vector into ~0.9 recall — the compression AND the accuracy."""
    from opq import opq_search, opq_tables
    n = rows if rows is not None else len(codes)
    pool = opq_search(np.asarray(codes[:n]), opq_tables(q, R, books), k=cand)[0]
    e = np.asarray(img[pool], np.float32) @ q
    top = np.argsort(e)[::-1][:k]
    return pool[top], e[top]


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
    rows = int(z["rows"]) if "rows" in z else 0
    return z["C"], z["order"], z["starts"], rows


def ivf_stale(rows):
    if not os.path.exists(IVF):
        return True
    with np.load(IVF) as z:
        return "rows" not in z or int(z["rows"]) != rows


def build_indexes(img, rows, log=print, want=("i8", "opq", "ivf")):
    """Rebuild every derived index that's missing or keyed to a different row
    count. This is the self-healing step: grow the corpus, and the int8 tower,
    the OPQ codes and the IVF lists all rebuild themselves on next touch —
    nothing downstream can silently read a stale index for the wrong rows."""
    if "i8" in want and i8_stale(rows):
        log("  building int8 (per-dimension scale) …"); build_i8(img, rows, log)
    if "opq" in want and opq_stale(rows):
        log("  building OPQ (64 bytes/vector) …"); build_opq(img, rows, log=log)
    if "ivf" in want and ivf_stale(rows):
        log("  building IVF (k-means on a 100k sample) …")
        t0 = time.time()
        C, order, starts = ivf_train(img, rows, log=lambda s: log(s))
        np.savez(IVF, C=C, order=order, starts=starts, rows=rows)
        log(f"  IVF built in {time.time() - t0:,.0f}s")


def cmd_search(args):
    from embedder import ClipEmbedder
    con = connect()
    rows = n_rows(con)
    q = ensemble_query(ClipEmbedder().embed_texts, args.query)
    img = np.load(IMG, mmap_mode="r")
    t0 = time.time()
    if args.method == "opq":                # 64 bytes/vector coarse → exact re-rank
        R, books, codes = load_opq()
        ids, scores = opq_rerank(codes, R, books, img, q, k=args.k, cand=args.cand, rows=rows)
        how = f"OPQ 64B → exact re-rank of {args.cand} (image tower)"
    elif args.method == "int8":             # half the bytes → exact re-rank
        i8, sc = load_i8()
        ids, scores = i8_rerank(i8, sc, img, q, k=args.k, cand=args.cand, rows=rows)
        how = f"int8 → exact re-rank of {args.cand} (image tower)"
    elif args.method == "ivf":
        C, order, starts, irows = load_ivf()
        if irows != rows:
            raise SystemExit(f"ivf index is for {irows:,} rows, corpus has {rows:,} — "
                             "run `scale.py bench` to rebuild it")
        ids, scores, scanned = ivf_search(q, open_mats(args.mode), C, order, starts,
                                          k=args.k, probes=args.probes)
        how = f"ivf probes={args.probes}, scanned {scanned / rows:.1%}"
    else:                                   # exact
        ids, scores = scan(open_mats(args.mode), q, k=args.k, rows=rows)
        how = f"exact scan of {rows:,}"
    ms = (time.time() - t0) * 1e3
    print(f"“{args.query}” — {args.method} search, {how}: {ms:,.0f} ms\n")
    for (i, name, caption, cafe, url), s in zip(fetch(con, ids), scores):
        print(f"  {s:+.3f}  {name}" + (f"  — {cafe}" if cafe else ""))
        print(f"          {url}")


# ----------------------------------------------------------------- bench --
def cmd_bench(args):
    """The honest table: every method's recall@{1,10,100} and latency, measured
    — never claimed. Recall is storage-independent, so it's measured in RAM over
    a large query set; latency is measured on each method's HONEST storage tier
    (int8/OPQ off their packed files, exact off f16 memmap and f32 RAM)."""
    con = connect()
    rows = n_rows(con)
    img = np.load(IMG, mmap_mode="r")
    if rows == 0:
        raise SystemExit("no rows — run `scale.py ingest` first")

    print(f"{rows:,} rows.  building any missing / stale indexes:")
    build_indexes(img, rows, log=lambda s: print(s, flush=True))
    print("\nsizes on disk — a million-scale corpus fits in a coat pocket:")
    labels = {DB: "records", IMG: "image f16", TXT: "text f16",
              I8: "image int8", OPQC: "image OPQ 64B", IVF: "ivf lists"}
    for p in (DB, IMG, TXT, I8, OPQC, IVF):
        if os.path.exists(p):
            print(f"  {labels[p]:<16} {os.path.basename(p):<22} {os.path.getsize(p) / 1e6:>8,.0f} MB")

    # ---- queries: real text (needs the model) or self-queries (model-free) ----
    if args.queries:
        from embedder import ClipEmbedder
        Q = np.stack([ensemble_query(ClipEmbedder().embed_texts, t)
                      for t in args.queries.split(",")])
        qtag = f"{len(Q)} real text queries"
    else:                                   # database vectors ARE the queries:
        rng = np.random.default_rng(0)      # the standard model-free recall proxy
        idx = np.sort(rng.choice(rows, min(args.nq, rows), replace=False))
        Q = np.asarray(img[idx], dtype=np.float32)
        qtag = f"{len(Q)} self-queries (database vectors — the honest model-free proxy)"

    # ---- promote to RAM for fast, storage-independent recall + exact truth ----
    print(f"\npromoting image tower → f32 RAM for recall over {qtag} …")
    Rimg = np.asarray(img[:rows], dtype=np.float32)
    R, books, ocodes = load_opq(); Rcodes = np.asarray(ocodes[:rows])   # 64B/vec in RAM
    i8, sc = load_i8()
    C, order, starts, _ = load_ivf()

    def exact(q, k):
        s = Rimg @ q; top = np.argpartition(s, -k)[-k:]
        return top[np.argsort(s[top])[::-1]]
    truth = [exact(q, 100) for q in Q]
    T = {1: [{int(t[0])} for t in truth], 10: [set(t[:10]) for t in truth],
         100: [set(t) for t in truth]}

    def recall(fn, k):
        return float(np.mean([len(set(np.asarray(fn(q, k))[:k].tolist()) & T[k][i]) / k
                              for i, q in enumerate(Q)]))

    def med(fn, qs):                        # median latency, warm cache
        ts = []
        for q in qs:
            t0 = time.time(); fn(q); ts.append(time.time() - t0)
        return 1e3 * float(np.median(ts))

    qlat = Q[:min(12, len(Q))]              # a small set is plenty for a median

    # exact latency on both storage tiers (the storage-hierarchy story)
    scan([(img, 1.0)], Q[0], rows=rows)     # first touch: page the file in
    t_mm = med(lambda q: scan([(img, 1.0)], q, rows=rows), qlat[:6])
    t_ram = med(lambda q: exact(q, 10), qlat)

    def opq_scan(q, k):
        from opq import opq_search, opq_tables
        return opq_search(Rcodes, opq_tables(q, R, books), k=k)[0]

    print(f"\n  {'method':<26}{'ms/q':>8}{'R@1':>7}{'R@10':>7}{'R@100':>8}  bytes/vec")
    print(f"  {'brute f16 (memmap)':<26}{t_mm:>8.0f}{1.00:>7.2f}{1.00:>7.2f}{1.00:>8.2f}  1024  (cold)")
    print(f"  {'brute f32 (RAM)':<26}{t_ram:>8.1f}{1.00:>7.2f}{1.00:>7.2f}{1.00:>8.2f}  2048  (serve)")
    # int8: latency off its packed file (half the bytes on disk is its whole win)
    t_i8 = med(lambda q: i8_search(i8, sc, q, k=10, rows=rows), qlat[:6])
    print(f"  {'int8 per-dim (memmap)':<26}{t_i8:>8.0f}"
          f"{recall(lambda q,k: i8_search(i8,sc,q,k=k,rows=rows)[0],1):>7.2f}"
          f"{recall(lambda q,k: i8_search(i8,sc,q,k=k,rows=rows)[0],10):>7.2f}"
          f"{recall(lambda q,k: i8_search(i8,sc,q,k=k,rows=rows)[0],100):>8.2f}  512")
    t_i8r = med(lambda q: i8_rerank(i8, sc, Rimg, q, k=10, cand=args.cand, rows=rows), qlat[:6])
    print(f"  {'int8 → exact re-rank':<26}{t_i8r:>8.0f}"
          f"{recall(lambda q,k: i8_rerank(i8,sc,Rimg,q,k=k,cand=args.cand,rows=rows)[0],1):>7.2f}"
          f"{recall(lambda q,k: i8_rerank(i8,sc,Rimg,q,k=k,cand=args.cand,rows=rows)[0],10):>7.2f}"
          f"{recall(lambda q,k: i8_rerank(i8,sc,Rimg,q,k=k,cand=args.cand,rows=rows)[0],100):>8.2f}  512")
    t_pq = med(lambda q: opq_scan(q, 10), qlat)
    print(f"  {'OPQ 64B (lookup scan)':<26}{t_pq:>8.1f}"
          f"{recall(opq_scan,1):>7.2f}{recall(opq_scan,10):>7.2f}{recall(opq_scan,100):>8.2f}  64")
    def opq_rr(q, k):
        return opq_rerank(Rcodes, R, books, Rimg, q, k=k, cand=args.cand, rows=rows)[0]
    t_pqr = med(lambda q: opq_rr(q, 10), qlat)
    print(f"  {'OPQ 64B → exact re-rank':<26}{t_pqr:>8.1f}"
          f"{recall(opq_rr,1):>7.2f}{recall(opq_rr,10):>7.2f}{recall(opq_rr,100):>8.2f}  64")

    # ---- IVF: recall is a dial priced in rows scanned ----
    print(f"\n  IVF — recall is a DIAL, priced in rows scanned (f32 RAM):")
    print(f"  {'probes':>7}{'ms/q':>8}{'R@10':>8}{'scanned':>10}")
    for probes in (1, 2, 4, 8, 16, 32):
        def ivf_fn(q, k=10, p=probes):
            return ivf_search(q, [(Rimg, 1.0)], C, order, starts, k=k, probes=p)[0]
        t = med(lambda q: ivf_fn(q), qlat)
        r = recall(lambda q, k: ivf_fn(q, k), 10)
        frac = float(np.mean([ivf_search(q, [(Rimg, 1.0)], C, order, starts,
                                         probes=probes)[2] for q in qlat])) / rows
        print(f"  {probes:>7}{t:>8.1f}{r:>8.2f}{frac:>9.1%}")

    # ---- one more rung: the GPU ----
    try:
        gpu = gpu_promote([(Rimg, 1.0)])
    except Exception:
        gpu = None
    if gpu:
        scan_gpu(gpu, Q[0])                 # first call pays the device warm-up
        t_gpu = med(lambda q: scan_gpu(gpu, q), qlat)
        print(f"\n  brute on {gpu[0]}: {t_gpu:.1f} ms/q, recall 1.00 "
              f"({t_mm / t_gpu:,.0f}× the memmap scan)")

    print("\nthe story, a thousandfold louder than ann.py: 64 bytes a vector for the"
          "\nfilter, an exact re-rank of a few hundred for the truth — scan ~1%, keep ~all.")


# ----------------------------------------------------------------- serve --
PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>one in a million — live</title><style>
:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--hair:#2c2c2a;--accent:#3987e5;--accent2:#86b6ef}
@media(prefers-color-scheme:light){:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;
--ink2:#52514e;--muted:#898781;--hair:#e1e0d9;--accent:#2a78d6;--accent2:#1c5cab}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:16px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:900px;margin:0 auto;padding:48px 20px}
h1{font-size:1.7rem;letter-spacing:-.02em;margin:0 0 2px}h1 em{font-style:normal;color:var(--accent2)}
.sub{color:var(--muted);font-size:.85rem;margin:0 0 22px}
.pill{display:flex;gap:8px;align-items:center;background:var(--surface);
border:1px solid var(--hair);border-radius:999px;padding:8px 8px 8px 18px}
.pill input[type=search]{flex:1;min-width:0;border:none;background:none;color:var(--ink);
font:inherit;font-size:1.05rem;outline:none}
.pill button{border:none;background:none;font-size:1.15rem;cursor:pointer;
padding:4px 10px;border-radius:999px}
.pill button:hover{background:var(--page)}
body.dragging .pill{border-color:var(--accent)}
.ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:12px 2px 0;
font-size:.8rem;color:var(--ink2)}
.ctl label{display:flex;gap:5px;align-items:center;cursor:pointer}
.ctl input[type=range]{accent-color:var(--accent);width:110px}
.ctl select{font:inherit;font-size:.8rem;color:var(--ink2);background:var(--surface);
border:1px solid var(--hair);border-radius:8px;padding:2px 6px}
.lat{margin:18px 2px 0;font-size:.85rem;color:var(--muted);min-height:1.4em}
.lat b{color:var(--accent2)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:14px;margin-top:14px}
figure{margin:0;background:var(--surface);border:1px solid var(--hair);border-radius:12px;
overflow:hidden;opacity:0;animation:rise .3s ease forwards}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1}}
figure img{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:var(--hair)}
figcaption{padding:6px 9px 8px;font-size:.72rem;color:var(--ink2)}
figcaption .s{color:var(--accent2);font-variant-numeric:tabular-nums}
.hidden{display:none!important}</style></head><body><main>
<h1>one in a <em>million</em>, live</h1>
<p class="sub">__ROWS__ real dishes · both CLIP towers in RAM · every answer prints its own cost</p>
<div class="pill"><input id="q" type="search" autofocus
  placeholder="ramen with a soft-boiled egg… — or drop a photo anywhere">
<button id="cam" title="reverse image search: pick a photo">📷</button>
<input id="file" type="file" accept="image/*" class="hidden"></div>
<div class="ctl">
  <label><input type="checkbox" id="ann" checked> ivf (milliseconds)</label>
  <label>probes <input type="range" id="probes" min="1" max="32" value="8">
    <b id="pv">8</b></label>
  <label>mode <select id="mode"><option value="fused" selected>fused</option>
    <option value="image">image</option><option value="text">text</option></select></label>
</div>
<div class="lat" id="lat"></div><div class="grid" id="grid"></div></main><script>
const $=id=>document.getElementById(id);let deb=0,seq=0;
function render(r,label){
$('lat').innerHTML=`<b>${r.ms} ms</b> across ${r.rows.toLocaleString()} rows (${r.engine})`+
(r.scanned<r.rows?` — scanned ${(100*r.scanned/r.rows).toFixed(1)}%`:'')+
` · ${label} embedded in ${r.embed_ms} ms`;
$('grid').replaceChildren(...r.results.map(x=>{const f=document.createElement('figure');
const i=document.createElement('img');i.src=x.url;i.alt=x.name;i.loading='lazy';
i.onerror=()=>f.classList.add('hidden');
const c=document.createElement('figcaption');
c.innerHTML=`<span class="s">${x.score>=0?'+':''}${x.score}</span> · `;
c.append(x.name+(x.cafe?' — '+x.cafe:''));f.append(i,c);return f}))}
async function go(){const q=$('q').value.trim();if(!q)return;const my=++seq;
$('lat').textContent='searching…';
const p=new URLSearchParams({q,mode:$('mode').value,ann:$('ann').checked?1:0,
probes:$('probes').value,k:24});
const r=await(await fetch('/api/search?'+p)).json();if(my!==seq)return;
render(r,'query')}
async function goImage(file){if(!file||!file.type.startsWith('image/'))return;
const my=++seq;$('lat').textContent='embedding your photo…';
const p=new URLSearchParams({ann:$('ann').checked?1:0,probes:$('probes').value,k:24});
const r=await(await fetch('/api/search-image?'+p,{method:'POST',body:file})).json();
if(my!==seq)return;render(r,'photo')}
$('q').addEventListener('input',()=>{clearTimeout(deb);deb=setTimeout(go,250)});
$('q').addEventListener('keydown',e=>{if(e.key==='Enter')go()});
$('probes').addEventListener('input',()=>{$('pv').textContent=$('probes').value;go()});
$('ann').addEventListener('change',go);$('mode').addEventListener('change',go);
$('cam').addEventListener('click',()=>$('file').click());
$('file').addEventListener('change',()=>goImage($('file').files[0]));
addEventListener('dragover',e=>{e.preventDefault();document.body.classList.add('dragging')});
addEventListener('dragleave',e=>{if(!e.relatedTarget)document.body.classList.remove('dragging')});
addEventListener('drop',e=>{e.preventDefault();document.body.classList.remove('dragging');
goImage(e.dataTransfer.files[0])});
addEventListener('paste',e=>{const f=[...(e.clipboardData?.items||[])]
.find(i=>i.type.startsWith('image/'));if(f)goImage(f.getAsFile())});
</script></body></html>"""


def cmd_serve(args):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse
    from embedder import ClipEmbedder

    rows = n_rows(connect())
    print(f"promoting both towers to f32 RAM (~4 GB) for {rows:,} rows …")
    t0 = time.time()
    img = np.asarray(np.load(IMG, mmap_mode="r")[:rows], dtype=np.float32)
    txt = np.asarray(np.load(TXT, mmap_mode="r")[:rows], dtype=np.float32)
    print(f"  towers in RAM in {time.time() - t0:,.0f}s")
    ivf = None                              # only trust an index built for THESE rows —
    if os.path.exists(IVF):                 # a stale index would point at rows that no
        Cv, orderv, startsv, irows = load_ivf()   # longer exist (an out-of-bounds crash)
        if irows == rows:
            ivf = (Cv, orderv, startsv)
        else:
            print(f"  (ignoring ivf index built for {irows:,} rows — this corpus has "
                  f"{rows:,}; run `python3 scale.py bench` to rebuild it)")
    mats = {"image": [(img, 1.0)], "text": [(txt, 1.0)],
            "fused": [(img, 0.5), (txt, 0.5)]}
    gpu = {}
    try:                                    # one more rung: both towers on device
        g = gpu_promote([(img, 1.0), (txt, 1.0)])
        if g:
            dev, ((gi, _), (gt, _)) = g
            gpu = {"image": (dev, [(gi, 1.0)]), "text": (dev, [(gt, 1.0)]),
                   "fused": (dev, [(gi, 0.5), (gt, 0.5)])}
            scan_gpu(gpu["fused"], np.zeros(512, np.float32))   # warm the device
            print(f"  towers on {dev} — exact scans run on the GPU")
    except Exception as e:
        print(f"  (no GPU path: {e} — exact scans stay in RAM)")
    emb = ClipEmbedder()
    emb.embed_texts(["warm-up"])            # first MPS call pays compile cost
    mps = threading.Lock()                  # one query through the model at a time

    def run_query(qv, mode, ann, probes, k, embed_ms):
        t0 = time.time()
        if ann and ivf:
            ids, scores, scanned = ivf_search(qv, mats[mode], *ivf, k=k, probes=probes)
            engine = "ivf"
        elif gpu:
            ids, scores = scan_gpu(gpu[mode], qv, k=k)
            scanned, engine = rows, f"exact on {gpu[mode][0]}"
        else:
            ids, scores = scan_ram(mats[mode], qv, k=k)
            scanned, engine = rows, "exact in RAM"
        ms = (time.time() - t0) * 1e3
        con = connect()                     # sqlite connections are per-thread
        recs = fetch(con, ids)
        con.close()
        return json.dumps({
            "ms": round(ms, 1), "embed_ms": round(embed_ms, 1),
            "scanned": int(scanned), "rows": rows, "engine": engine,
            "results": [{"id": int(i), "name": n, "cafe": c, "url": url,
                         "score": round(float(s), 3)}
                        for (i, n, cap, c, url), s in zip(recs, scores)],
        }).encode()

    def knobs(p):
        return (p.get("mode", ["fused"])[0],
                p.get("ann", ["0"])[0] == "1",
                max(1, min(64, int(p.get("probes", ["8"])[0]))),
                max(1, min(50, int(p.get("k", ["24"])[0]))))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # quiet: the UI shows every cost
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                page = PAGE.replace("__ROWS__", f"{rows:,}").encode()
                return self.send(200, page, "text/html; charset=utf-8")
            if u.path != "/api/search":
                return self.send(404, b"{}", "application/json")
            p = parse_qs(u.query)
            query = p.get("q", [""])[0].strip()
            if not query:
                return self.send(400, b'{"error":"empty query"}', "application/json")
            mode, ann, probes, k = knobs(p)
            if mode not in mats:
                return self.send(400, b'{"error":"bad mode"}', "application/json")
            t0 = time.time()
            with mps:
                qv = ensemble_query(emb.embed_texts, query)
            embed_ms = (time.time() - t0) * 1e3
            return self.send(200, run_query(qv, mode, ann, probes, k, embed_ms),
                             "application/json")

        def do_POST(self):
            # reverse image search: the request body IS the photo. The vision
            # tower embeds it here; from then on it's the same scan as text.
            import io
            from PIL import Image
            u = urlparse(self.path)
            if u.path != "/api/search-image":
                return self.send(404, b"{}", "application/json")
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 20_000_000:
                return self.send(400, b'{"error":"bad image size"}', "application/json")
            _, ann, probes, k = knobs(parse_qs(u.query))
            try:
                pil = Image.open(io.BytesIO(self.rfile.read(size))).convert("RGB")
            except Exception:
                return self.send(400, b'{"error":"not an image"}', "application/json")
            t0 = time.time()
            with mps:
                inputs = emb.processor(images=[pil], return_tensors="pt").to(emb.device)
                import torch
                with torch.no_grad():
                    feats = emb.model.get_image_features(**inputs)
                qv = (feats.pooler_output if hasattr(feats, "pooler_output") else feats)
                qv = qv[0].float().cpu().numpy()
                qv /= np.linalg.norm(qv) or 1.0
            embed_ms = (time.time() - t0) * 1e3
            return self.send(200, run_query(qv.astype(np.float32), "image", ann,
                                            probes, k, embed_ms), "application/json")

    port = args.port
    while True:                             # never fight another process for a port
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    print(f"the million is live: http://localhost:{port}")
    srv.serve_forever()


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
        # int8, PER-DIMENSION scale — through the real helpers (no global files
        # touched: i8_search/i8_rerank take the arrays as arguments).
        sc = (np.abs(X).max(0) / 127.0).astype(np.float32); sc[sc == 0] = 1.0
        i8 = np.round(X / sc).astype(np.int8)
        f8 = i8_search(i8, sc, q, k=10, rows=n)[0]
        check(len(set(f8) & set(truth)) >= 9, "int8 per-dim keeps ≥9/10 of the truth")
        rr = i8_rerank(i8, sc, M, q, k=10, cand=100, rows=n)[0]
        check(set(rr) == set(truth), "int8 → exact re-rank recovers 10/10")
        # OPQ, end to end through opq_rerank: 64-block codes → exact re-rank.
        # Coarse 64-byte codes are a filter, not the answer — the exact re-rank
        # of their shortlist is what recovers the truth (and never does worse).
        from opq import opq_encode, opq_search, opq_tables, opq_train
        Ro, books = opq_train(X, m=8, ks=64, outer=4, inner=4, log=lambda s: None)
        ocodes = opq_encode(X, Ro, books)
        coarse = set(opq_search(ocodes, opq_tables(q, Ro, books), k=10)[0].tolist())
        orr = set(opq_rerank(ocodes, Ro, books, M, q, k=10, cand=300, rows=n)[0].tolist())
        check(len(orr & set(truth)) >= len(coarse & set(truth)),
              "OPQ → exact re-rank is never worse than coarse OPQ")
        check(len(orr & set(truth)) >= 5,
              "OPQ 64-block → exact re-rank recovers a majority of the truth")
        rids, rscores = scan_ram([(X.astype(np.float32), 1.0)], q, k=10)
        check(set(rids) == set(truth) and np.allclose(rscores, scores, atol=1e-3),
              "scan_ram == chunked scan == the truth")
        fake = lambda texts: np.stack([X[0], X[1]])   # two phrasings, canned
        e = ensemble_query(fake, "anything")
        mid = (X[0] + X[1]) / 2
        check(abs(np.linalg.norm(e) - 1) < 1e-6
              and np.allclose(e, mid / np.linalg.norm(mid), atol=1e-6),
              "ensemble_query = renormalised mean of the phrasings")
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
    p.add_argument("--method", choices=("exact", "ivf", "int8", "opq"), default="exact",
                   help="exact scan · ivf cells · int8/opq coarse → exact re-rank")
    p.add_argument("--mode", choices=("image", "text", "fused"), default="image",
                   help="which tower(s) to score (int8/opq are image-tower)")
    p.add_argument("--probes", type=int, default=8, help="ivf: cells to scan")
    p.add_argument("--cand", type=int, default=400, help="int8/opq: candidates to re-rank")
    p.add_argument("-k", type=int, default=10)
    p = sub.add_parser("bench", help="sizes, latency, recall — the honest table")
    p.add_argument("--queries", help="comma-separated real text queries (else self-queries)")
    p.add_argument("--nq", type=int, default=200, help="self-query sample size for recall")
    p.add_argument("--cand", type=int, default=400, help="candidates for the re-rank stage")
    p = sub.add_parser("serve", help="the million live at localhost — towers in RAM")
    p.add_argument("--port", type=int, default=8071)
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
    elif args.cmd == "serve":
        cmd_serve(args)
    else:
        cmd_selftest(args)


if __name__ == "__main__":
    main()
