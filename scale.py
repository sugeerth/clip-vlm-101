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

    stale8 = True                            # rebuild whenever the corpus grew
    if os.path.exists(I8) and os.path.exists(I8S):
        meta8 = np.atleast_1d(np.load(I8S))
        stale8 = meta8.size < 2 or int(meta8[1]) != rows
    if stale8:
        # quantize.py's scheme, chunked: ONE shared scale so the largest
        # value maps to ±127 — naive 127·x wastes the byte on unit vectors,
        # whose 512-d components rarely leave ±0.1
        amax = max(float(np.abs(np.asarray(img[lo:lo + CHUNK], np.float32)).max())
                   for lo in range(0, rows, CHUNK))
        i8 = np.lib.format.open_memmap(I8, mode="w+", dtype=np.int8, shape=(rows, 512))
        for lo in range(0, rows, CHUNK):
            hi = min(lo + CHUNK, rows)
            i8[lo:hi] = np.round(np.asarray(img[lo:hi], np.float32)
                                 / (amax / 127.0)).astype(np.int8)
        i8.flush()
        np.save(I8S, np.array([amax / 127.0, rows], dtype=np.float64))
    i8 = np.load(I8, mmap_mode="r")
    i8_scale = float(np.atleast_1d(np.load(I8S))[0])
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

    staleivf = True
    if os.path.exists(IVF):
        with np.load(IVF) as z:
            staleivf = "rows" not in z or int(z["rows"]) != rows
    if staleivf:
        print("\n  building the ivf index (k-means on a 100k sample) …")
        t0 = time.time()
        C, order, starts = ivf_train(img, rows, log=lambda s: print(s, flush=True))
        np.savez(IVF, C=C, order=order, starts=starts, rows=rows)
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
    if args.ram:
        # the memmap scan is I/O-bound: paging + f16→f32 casting, not math.
        # Promote the tower to f32 RAM once and the SAME exact scan is a
        # single matmul — this is what `serve` does.
        t0 = time.time()
        R = np.asarray(img[:rows], dtype=np.float32)
        print(f"\n  tower → f32 RAM ({R.nbytes / 1e9:.1f} GB) in {time.time() - t0:,.1f}s")
        t_ram = med(lambda q: scan_ram([(R, 1.0)], q))
        print(f"  brute f32 RAM   {t_ram:>8,.1f} ms/query   recall 1.00 "
              f"({t_brute / t_ram:,.0f}× the memmap scan)")
        t_ivfr = med(lambda q: ivf_search(q, [(R, 1.0)], C, order, starts, probes=8))
        print(f"  ivf-RAM p=8     {t_ivfr:>8,.1f} ms/query   recall as above")
        try:
            gpu = gpu_promote([(R, 1.0)])
        except Exception:
            gpu = None
        if gpu:
            scan_gpu(gpu, Q[0])            # first call pays the device warm-up
            t_gpu = med(lambda q: scan_gpu(gpu, q))
            print(f"  brute {gpu[0]:>4}      {t_gpu:>8,.1f} ms/query   recall 1.00 "
                  f"({t_brute / t_gpu:,.0f}× the memmap scan)")

    print("\nsame story as ann.py, a thousandfold louder: scan ~1%, keep ~all of it.")


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
    ivf = load_ivf() if os.path.exists(IVF) else None
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
        s8 = float(np.abs(X).max()) / 127.0          # quantize.py's shared scale
        i8 = np.round(X / s8).astype(np.int8)
        f8 = np.argsort((i8.astype(np.float32) * s8) @ q)[::-1][:10]
        check(len(set(f8) & set(truth)) >= 9, "int8 (max-abs scale) keeps ≥9/10 of the truth")
        cand = np.argsort((i8.astype(np.float32) * s8) @ q)[::-1][:100]
        rr = cand[np.argsort(X[cand] @ q)[::-1][:10]]
        check(set(rr) == set(truth), "int8 top-100 → exact re-rank recovers 10/10")
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
    p.add_argument("--mode", choices=("image", "text", "fused"), default="image")
    p.add_argument("--ann", action="store_true")
    p.add_argument("--probes", type=int, default=8)
    p.add_argument("-k", type=int, default=10)
    p = sub.add_parser("bench", help="sizes, latency, recall — the honest table")
    p.add_argument("--queries", help="comma-separated real queries (else random vectors)")
    p.add_argument("--ram", action="store_true", help="also time the towers promoted to f32 RAM")
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
