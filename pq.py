"""Product quantization: a million-scale index small enough for a BROWSER.

pipeline: (n, 512) f32 vectors ──► [pq] ──► 64 bytes per vector + codebooks

quantize.py shrank every NUMBER to one byte (512 bytes/vector). PQ shrinks
every VECTOR to m bytes, almost regardless of dimension:

    1. SPLIT   512 dims into m=64 subvectors of 8 dims each
    2. TRAIN   k-means 256 centroids PER subspace (the "codebooks")
    3. ENCODE  each subvector -> the byte naming its nearest centroid

A vector is now 64 bytes: 32× smaller than float32, 16× smaller than f16.
And the search trick (ADC — asymmetric distance computation) never
reconstructs anything: for a query q, precompute one 256-entry table of
dot products per subspace (m·256 tiny dots, microseconds), then a row's
score is just m table lookups summed. No multiplies per row at all.

That last property is why this file exists: the export fits GitHub Pages
(100k dishes ≈ 6.4 MB of codes + 0.5 MB of codebooks) and js/pq.js — this
file's browser twin — runs the SAME table-lookup search over it in
milliseconds, against a query embedded by the demo's own text tower.

Two things make the shipped index actually accurate, not just small:
  · OPQ (opq.py) learns a rotation before the split, so the 64 codes carry
    ~10 recall points more than plain PQ for free — the export ships the
    rotation (512×512 f32, ~1 MB) and js/pq.js rotates the query.
  · a two-stage search: the coarse codes nominate a few hundred candidates,
    then an int8 REFINE tier (the same dishes at one byte per dimension,
    ~51 MB, loaded only on demand) re-scores them exactly. Coarse alone keeps
    ~0.56 of the exact top-10; with the refine tier, ~0.84.

Accuracy is a measured tradeoff, as always in this repo: cmd_export measures
BOTH numbers on the real slice and bakes them into the manifest, so the web
page prints what it actually delivers, not what it hopes to.

Run me:  python3 pq.py selftest                    (synthetic, CI-safe)
         python3 pq.py export                      (data/ -> docs/million/)
"""
import argparse
import json
import os
import time

import numpy as np

M_SUB = 64            # subvectors per vector -> bytes per vector
KS = 256              # centroids per subspace -> one byte names one
OUT = os.path.join("docs", "million")
WEB_ROWS = 100_000    # the slice that ships: every 10th row of the million


def pq_train(X, m=M_SUB, ks=KS, iters=8, seed=0, log=print):
    """k-means per subspace, vectorised Lloyd. X (n, d) f32, d % m == 0.
    Returns codebooks (m, ks, d//m)."""
    n, d = X.shape
    sub = d // m
    rng = np.random.default_rng(seed)
    books = np.empty((m, ks, sub), dtype=np.float32)
    for j in range(m):
        S = X[:, j * sub:(j + 1) * sub]
        C = S[rng.choice(n, ks, replace=False)].copy()
        for _ in range(iters):
            # nearest centroid by L2: argmax(2 x·c - |c|²) — one matmul
            assign = np.argmax(2 * (S @ C.T) - (C * C).sum(1), axis=1)
            for c in range(ks):
                members = S[assign == c]
                C[c] = members.mean(0) if len(members) else S[rng.integers(n)]
        books[j] = C
        if (j + 1) % 16 == 0:
            log(f"  trained subspace {j + 1}/{m}")
    return books


def pq_encode(X, books, chunk=131_072):
    """Every subvector -> the byte of its nearest centroid. Returns (n, m) u8."""
    n, d = X.shape
    m, ks, sub = books.shape
    codes = np.empty((n, m), dtype=np.uint8)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        for j in range(m):
            S = X[lo:hi, j * sub:(j + 1) * sub]
            C = books[j]
            codes[lo:hi, j] = np.argmax(2 * (S @ C.T) - (C * C).sum(1), axis=1)
    return codes


def adc_tables(q, books):
    """The query-side half of ADC: per subspace, q_sub · every centroid.
    (m, ks) f32 — computed once per query, in microseconds."""
    m, ks, sub = books.shape
    return np.stack([books[j] @ q[j * sub:(j + 1) * sub] for j in range(m)])


def pq_search(codes, tables, k=10):
    """score(row) = sum over subspaces of table[j][code[j]] — lookups, no math.
    This line IS what js/pq.js does per keystroke."""
    scores = tables[np.arange(codes.shape[1]), codes].sum(axis=1)
    top = np.argpartition(scores, -min(k, len(scores)))[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return top, scores[top]


def _rotated(q, R):
    return (R @ q).astype(np.float32) if R is not None else q


def recall_opq(X, codes, books, R=None, n_queries=100, seed=0, within=10):
    """How much of the exact top-10 the coarse codes keep in THEIR top-`within`,
    self-queries (the hardest case). This is the base pack's honest number."""
    rng = np.random.default_rng(seed)
    hits = 0
    for i in rng.choice(len(X), n_queries, replace=False):
        truth = set(np.argsort(X @ X[i])[::-1][:10].tolist())
        found, _ = pq_search(codes, adc_tables(_rotated(X[i], R), books), k=within)
        hits += len(set(found.tolist()) & truth)
    return hits / (10 * n_queries)


def recall_rerank(X, codes, books, R, i8, i8_scale, cand=400, n_queries=100, seed=0):
    """The two-stage number the browser actually delivers with the refine tier
    loaded: coarse codes nominate `cand`, the int8 vectors re-score them exactly
    (per-dim dequant folded into the query). Measured against the exact top-10."""
    rng = np.random.default_rng(seed)
    hits = 0
    for i in rng.choice(len(X), n_queries, replace=False):
        q = X[i]
        truth = set(np.argsort(X @ q)[::-1][:10].tolist())
        c, _ = pq_search(codes, adc_tables(_rotated(q, R), books), k=cand)
        s = i8[c].astype(np.float32) @ (i8_scale * q)          # int8 · (scale ⊙ q)
        top = c[np.argsort(s)[::-1][:10]]
        hits += len(set(top.tolist()) & truth)
    return hits / (10 * n_queries)


# ---------------------------------------------------------------- export --
def cmd_export(args):
    """Two tiers, both browser-ready:
      base    OPQ 64-byte codes + the rotation — ~7 MB, loads on open, coarse.
      refine  the same dishes in int8 (per-dim) — ~51 MB, loads only when the
              user asks for precision, then a two-stage search re-ranks exactly.
    The manifest bakes in the MEASURED recall of each tier so js/pq.js can
    print honest numbers, not hopeful ones."""
    import scale
    from opq import opq_encode, opq_train

    con = scale.connect()
    rows = scale.n_rows(con)
    stride = max(1, rows // args.rows)
    pick = np.arange(0, rows, stride)[:args.rows]
    print(f"slice: every {stride}th row of {rows:,} -> {len(pick):,} dishes")
    X = np.asarray(np.load(scale.IMG, mmap_mode="r")[pick], dtype=np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    t0 = time.time()
    R, books = opq_train(X[:: max(1, len(X) // 200_000)], m=M_SUB, ks=KS, log=print)
    codes = opq_encode(X, R, books)                            # (n, 64) uint8
    # int8 refine tier: one scale PER DIMENSION so CLIP's outlier dims don't
    # eat the byte's range (measurably better than a single global scale).
    i8_scale = np.abs(X).max(0) / 127.0
    i8_scale[i8_scale == 0] = 1.0
    i8 = np.round(X / i8_scale).astype(np.int8)                # (n, 512) int8
    rec = recall_opq(X, codes, books, R)
    rec_rr = recall_rerank(X, codes, books, R, i8, i8_scale.astype(np.float32), cand=args.cand)
    print(f"trained + encoded in {time.time() - t0:,.0f}s — coarse OPQ recall@10 "
          f"{rec:.2f}; with the int8 re-rank of {args.cand}: {rec_rr:.2f}")

    os.makedirs(OUT, exist_ok=True)
    books.astype("<f4").tofile(os.path.join(OUT, "pq_books.bin"))       # OPQ codebooks
    codes.tofile(os.path.join(OUT, "pq_codes.bin"))                     # OPQ codes
    R.astype("<f4").tofile(os.path.join(OUT, "opq_rotation.bin"))       # 512×512 f32 (JS-friendly)
    i8.tofile(os.path.join(OUT, "refine_i8.bin"))                       # n×512 int8
    i8_scale.astype("<f4").tofile(os.path.join(OUT, "refine_scale.bin"))  # 512 f32

    # metadata, trimmed for the wire: shared URL prefix factored out once
    by_id = {}
    for lo in range(0, len(pick), 500):    # sqlite caps '?' placeholders
        batch = [int(i) for i in pick[lo:lo + 500]]
        for r in con.execute(
                f"SELECT id, name, cafe, url FROM items WHERE id IN ({','.join('?' * len(batch))})",
                batch):
            by_id[r[0]] = r
    prefix = os.path.commonprefix([by_id[int(i)][3] for i in pick[:200] if by_id[int(i)][3]])
    items = [[by_id[int(i)][1][:60], (by_id[int(i)][2] or "")[:30],
              by_id[int(i)][3][len(prefix):] if by_id[int(i)][3].startswith(prefix)
              else by_id[int(i)][3]] for i in pick]
    manifest = {"n": len(pick), "m": M_SUB, "ks": KS, "sub": 512 // M_SUB, "d": 512,
                "of_rows": rows, "stride": stride,
                "recall10": round(float(rec), 2),          # base pack (coarse)
                "recall10_rerank": round(float(rec_rr), 2),  # + refine tier
                "rerank_cand": args.cand, "url_prefix": prefix,
                "refine": {"codes": "refine_i8.bin", "scale": "refine_scale.bin",
                           "bytes": int(i8.nbytes)}}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump({"manifest": manifest, "items": items}, f, ensure_ascii=False,
                  separators=(",", ":"))
    print("wrote:")
    for name in ("pq_books.bin", "pq_codes.bin", "opq_rotation.bin",
                 "refine_i8.bin", "refine_scale.bin", "meta.json"):
        p = os.path.join(OUT, name)
        print(f"  {p:<30} {os.path.getsize(p) / 1e6:>6.1f} MB")


# -------------------------------------------------------------- selftest --
def cmd_selftest(args):
    """The PQ math on synthetic clustered vectors — no data, no model."""
    rng = np.random.default_rng(0)
    n, d = 5_000, 64
    centers = rng.normal(size=(32, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X = centers[rng.integers(0, 32, n)] + 0.15 * rng.normal(size=(n, d))
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  {'pass' if cond else 'FAIL'} {msg}")
        ok &= bool(cond)

    books = pq_train(X, m=8, ks=64, iters=6, log=lambda s: None)
    codes = pq_encode(X, books)
    check(codes.shape == (n, 8) and codes.dtype == np.uint8,
          "codes: one byte per subspace per vector")
    q = X[rng.integers(n)]
    T = adc_tables(q, books)
    check(T.shape == (8, 64), "adc tables: one row of dots per subspace")
    approx = T[np.arange(8), codes].sum(1)
    recon = np.stack([books[j][codes[:, j]] for j in range(8)]).transpose(1, 0, 2)
    exact_on_recon = (recon.reshape(n, -1) @ q)
    check(np.allclose(approx, exact_on_recon, atol=1e-4),
          "ADC score == dot with the reconstruction (never materialised)")
    qi = int(rng.integers(n))
    self_found, _ = pq_search(codes, adc_tables(X[qi], books), k=10)
    check(qi in set(self_found), "a vector finds ITSELF in PQ's top-10")
    rec50 = recall_opq(X, codes, books, R=None, n_queries=30, within=50)
    check(rec50 >= 0.7, f"exact top-10 stays in PQ's top-50 (kept {rec50:.2f})")
    found, scores = pq_search(codes, T, k=10)
    check(np.all(np.diff(scores) <= 1e-6), "pq_search returns scores sorted")
    print("all pq.py checks passed" if ok else "some pq.py checks FAILED")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="OPQ-pack a slice of the million for docs/ (+ int8 refine tier)")
    p.add_argument("--rows", type=int, default=WEB_ROWS)
    p.add_argument("--cand", type=int, default=400, help="candidates the refine tier re-ranks")
    sub.add_parser("selftest", help="synthetic PQ math check, CI-safe")
    args = ap.parse_args()
    cmd_export(args) if args.cmd == "export" else cmd_selftest(args)


if __name__ == "__main__":
    main()
