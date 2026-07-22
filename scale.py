"""At scale: where the database's shape starts to matter — the CLI that ties the
one-concept-per-file engine together.

pipeline: parquet ──► [store] ──► f16 memmaps ──► [binary|opq|int8|ivf] ──► ms search

db.py keeps every vector a BLOB inside SQLite — perfect at 14 rows, wrong at a
million. At scale the engine splits into files that each teach ONE idea, and this
file is only the conductor:

  store.py    records in SQLite + packed f16 memmaps; the ingest
  scan.py     the exact answer, at three rungs: memmap → RAM → GPU
  ivf.py      the inverted-file index — scan a few cells, not all rows
  intq.py     int8, per dimension — half the bytes
  opq.py      optimised product quantization — 64 bytes/vector
  binary.py   1 bit/dimension — the cheapest distance a CPU has
  engine.py   the APPROXIMATION CASCADE: binary → OPQ → int8 → exact, and the
              self-healing index build
  bench.py    the honest recall/latency table
  serve.py    the whole thing live at localhost, low-memory via the cascade

Vectors come from Qdrant's public Wolt food dataset; text embeddings are computed
here by embedder.py. Row id i in the DB is row i of every matrix.

Run me:  python3 scale.py selftest                (synthetic, no model, CI-safe)
         python3 scale.py ingest                  (data/part-*.parquet -> the DB)
         python3 scale.py search "pad thai" --method cascade
         python3 scale.py bench                    (the honest table)
         python3 scale.py serve                    (live at localhost)
"""
import argparse
import time

import numpy as np

import binary
import engine
import intq
import ivf as ivfmod
import opq
from scan import ensemble_query, scan, scan_ram
from store import IMG, connect, fetch, n_rows, open_mats


# ---------------------------------------------------------------- search --
def cmd_search(args):
    from embedder import ClipEmbedder
    con = connect()
    rows = n_rows(con)
    q = ensemble_query(ClipEmbedder().embed_texts, args.query)
    img = np.load(IMG, mmap_mode="r")
    t0 = time.time()
    if args.method == "cascade":            # binary → OPQ → int8 → exact
        ids, scores, scanned = engine.cascade_search(
            q, img, engine.load_indexes(rows), k=args.k, rows=rows)
        how = f"cascade, {scanned:,} rows filtered to a handful"
    elif args.method == "opq":              # 64 bytes/vector coarse → exact re-rank
        R, books, codes = opq.load_opq()
        ids, scores = opq.opq_rerank(codes, R, books, img, q, k=args.k, cand=args.cand, rows=rows)
        how = f"OPQ 64B → exact re-rank of {args.cand}"
    elif args.method == "int8":             # half the bytes → exact re-rank
        i8, sc = intq.load_i8()
        ids, scores = intq.i8_rerank(i8, sc, img, q, k=args.k, cand=args.cand, rows=rows)
        how = f"int8 → exact re-rank of {args.cand}"
    elif args.method == "binary":           # 1 bit/dim → exact re-rank
        bits = binary.load_binary()
        ids, scores = binary.binary_rerank(bits, img, q, k=args.k, cand=args.cand, rows=rows)
        how = f"binary 1-bit → exact re-rank of {args.cand}"
    elif args.method == "ivf":
        C, order, starts, irows = ivfmod.load_ivf()
        if irows != rows:
            raise SystemExit(f"ivf index is for {irows:,} rows, corpus has {rows:,} — "
                             "run `python3 scale.py bench` to rebuild it")
        ids, scores, scanned = ivfmod.ivf_search(q, open_mats(args.mode), C, order, starts,
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


# -------------------------------------------------------------- selftest --
def cmd_selftest(args):
    """The whole engine on synthetic clustered vectors — no model, no data
    files, no network. Imports every module and checks the maths lines up. CI."""
    import os
    import tempfile
    from scan import scan_ram as _scan_ram  # noqa (kept explicit for the check below)
    rng = np.random.default_rng(0)
    n, d = 20_000, 64
    centers = rng.normal(size=(64, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X = centers[rng.integers(0, 64, n)] + 0.2 * rng.normal(size=(n, d))
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    q = X[rng.integers(n)]
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  {'pass' if cond else 'FAIL'} {msg}")
        ok &= bool(cond)

    with tempfile.TemporaryDirectory() as tmp:
        M = np.lib.format.open_memmap(os.path.join(tmp, "m.npy"), mode="w+",
                                      dtype=np.float16, shape=(n, d))
        M[:] = X
        truth = np.argsort(X @ q)[::-1][:10]
        Tset = set(truth.tolist())

        # scan.py
        ids, scores = scan([(M, 1.0)], q, k=10, chunk=4096)
        check(set(ids) == Tset, "scan.py: chunked scan == naive argsort")
        two = scan([(M, 0.5), (M, 0.5)], q, k=10, chunk=4096)[1]
        check(np.allclose(two, scores, atol=1e-3), "scan.py: fused = weighted sum of towers")
        rids, rscores = scan_ram([(X, 1.0)], q, k=10)
        check(set(rids) == Tset and np.allclose(rscores, scores, atol=1e-3),
              "scan.py: scan_ram == chunked scan == the truth")

        # ivf.py
        C, order, starts = ivfmod.ivf_train(M, n, n_lists=64, sample=5_000, iters=4, log=lambda s: None)
        check(starts[-1] == n and len(np.unique(order)) == n,
              "ivf.py: lists partition every row exactly once")
        found, _, scanned = ivfmod.ivf_search(q, [(M, 1.0)], C, order, starts, probes=8)
        check(len(set(found) & Tset) >= 7 and scanned < n / 2,
              "ivf.py: probes=8 keeps most of the truth scanning <½ the rows")

        # intq.py (per-dim int8)
        sc = (np.abs(X).max(0) / 127.0).astype(np.float32); sc[sc == 0] = 1.0
        i8 = np.round(X / sc).astype(np.int8)
        check(len(set(intq.i8_search(i8, sc, q, k=10, rows=n)[0]) & Tset) >= 9,
              "intq.py: int8 per-dim keeps ≥9/10 of the truth")
        check(set(intq.i8_rerank(i8, sc, M, q, k=10, cand=100, rows=n)[0]) == Tset,
              "intq.py: int8 → exact re-rank recovers 10/10")

        # binary.py (1 bit/dim)
        bits = binary.binarize(X)
        check(bits.shape == (n, d // 8) and bits.dtype == np.uint8,
              "binary.py: 1 bit/dim, packed 8 to a byte")
        qi = int(rng.integers(n))
        check(qi in set(binary.binary_search(bits, X[qi], k=10, rows=n)[0].tolist()),
              "binary.py: a vector finds ITSELF in the popcount top-10")
        check(len(set(binary.binary_rerank(bits, M, q, k=10, cand=500, rows=n)[0]) & Tset) >= 7,
              "binary.py: 1-bit filter → exact re-rank keeps most of the truth")

        # opq.py (rotation + PQ)
        Ro, books = opq.opq_train(X, m=8, ks=64, outer=4, inner=4, log=lambda s: None)
        codes = opq.opq_encode(X, Ro, books)
        orr = set(opq.opq_rerank(codes, Ro, books, M, q, k=10, cand=300, rows=n)[0].tolist())
        coarse = set(opq.opq_search(codes, opq.opq_tables(q, Ro, books), k=10)[0].tolist())
        check(len(orr & Tset) >= len(coarse & Tset) and len(orr & Tset) >= 5,
              "opq.py: 64-block → exact re-rank recovers a majority (≥ coarse)")

        # engine.py — the cascade: binary → OPQ → int8 → exact
        idx = {"bits": bits, "R": Ro, "books": books, "codes": codes, "i8": i8, "i8scale": sc}
        cids, _, cscanned = engine.cascade_search(q, M, idx, k=10, stages=(2000, 400, 100), rows=n)
        check(cscanned == n, "engine.py: the cascade's coarse pass sees every row")
        check(len(set(cids.tolist()) & Tset) >= 8,
              "engine.py: cascade keeps most of the exact top-10 (approximate everywhere)")

        # scan.py ensemble_query
        e = ensemble_query(lambda texts: np.stack([X[0], X[1]]), "anything")
        mid = (X[0] + X[1]) / 2
        check(abs(np.linalg.norm(e) - 1) < 1e-6
              and np.allclose(e, mid / np.linalg.norm(mid), atol=1e-6),
              "scan.py: ensemble_query = renormalised mean of the phrasings")
    print("all scale.py checks passed" if ok else "some scale.py checks FAILED")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="parquet parts -> sqlite + packed matrices")
    p.add_argument("--parts", nargs="*", default=None)
    p.add_argument("--rows", type=int, default=1_000_000)
    p = sub.add_parser("search", help="live text query against the corpus")
    p.add_argument("query")
    p.add_argument("--method",
                   choices=("exact", "ivf", "binary", "int8", "opq", "cascade"),
                   default="cascade",
                   help="exact · ivf cells · binary/int8/opq coarse→re-rank · full cascade")
    p.add_argument("--mode", choices=("image", "text", "fused"), default="image",
                   help="which tower(s) to score (compressed indexes are image-tower)")
    p.add_argument("--probes", type=int, default=8, help="ivf: cells to scan")
    p.add_argument("--cand", type=int, default=400, help="coarse→re-rank: candidates")
    p.add_argument("-k", type=int, default=10)
    p = sub.add_parser("bench", help="sizes, latency, recall — the honest table")
    p.add_argument("--queries", help="comma-separated real text queries (else self-queries)")
    p.add_argument("--nq", type=int, default=200, help="self-query sample size for recall")
    p.add_argument("--cand", type=int, default=400, help="candidates for the re-rank stage")
    p = sub.add_parser("serve", help="the corpus live at localhost — cascade in RAM")
    p.add_argument("--port", type=int, default=8071)
    sub.add_parser("selftest", help="synthetic end-to-end check, CI-safe")
    args = ap.parse_args()
    if args.cmd == "ingest":
        import glob
        import os
        import store
        parts = args.parts or sorted(glob.glob(os.path.join(store.DATA, "part-*.parquet")))
        store.ingest(parts, args.rows)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "bench":
        import bench
        bench.cmd_bench(args)
    elif args.cmd == "serve":
        import serve
        serve.cmd_serve(args)
    else:
        cmd_selftest(args)


if __name__ == "__main__":
    main()
