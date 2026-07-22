"""The honest table: every method's recall@{1,10,100} and latency, measured.

Recall is storage-independent, so it's measured in RAM over a large query set;
latency is measured on each method's honest storage tier. Self-queries (database
vectors) are the model-free proxy — the hardest case, and what CI-safe scale.py
selftest checks the machinery of.
"""
import os
import time

import numpy as np

import binary
import engine
import intq
import ivf as ivfmod
import opq
from scan import ensemble_query, gpu_promote, scan, scan_gpu
from store import DB, IMG, TXT, connect, n_rows


def cmd_bench(args):
    con = connect()
    rows = n_rows(con)
    img = np.load(IMG, mmap_mode="r")
    if rows == 0:
        raise SystemExit("no rows — run `python3 scale.py ingest` first")

    print(f"{rows:,} rows.  building any missing / stale indexes:")
    engine.build_indexes(img, rows, log=lambda s: print(s, flush=True))
    print("\nsizes on disk — the whole corpus, and its compressed indexes:")
    labels = [(DB, "records"), (IMG, "image f16"), (TXT, "text f16"),
              (intq.I8, "image int8"), (opq.OPQC, "image OPQ 64B"),
              (binary.BIN, "image binary 1-bit"), (ivfmod.IVF, "ivf lists")]
    for p, lab in labels:
        if os.path.exists(p):
            print(f"  {lab:<20} {os.path.basename(p):<22} {os.path.getsize(p) / 1e6:>8,.0f} MB")

    if args.queries:
        from embedder import ClipEmbedder
        Q = np.stack([ensemble_query(ClipEmbedder().embed_texts, t)
                      for t in args.queries.split(",")])
        qtag = f"{len(Q)} real text queries"
    else:
        rng = np.random.default_rng(0)
        idxq = np.sort(rng.choice(rows, min(args.nq, rows), replace=False))
        Q = np.asarray(img[idxq], dtype=np.float32)
        qtag = f"{len(Q)} self-queries (the honest model-free proxy)"

    print(f"\npromoting image tower → f32 RAM for recall over {qtag} …")
    Rimg = np.asarray(img[:rows], dtype=np.float32)
    ix = engine.load_indexes(rows, to_ram=True)
    C, order, starts, _ = ivfmod.load_ivf()

    def exact(q, k):
        s = Rimg @ q; top = np.argpartition(s, -k)[-k:]
        return top[np.argsort(s[top])[::-1]]
    truth = [exact(q, 100) for q in Q]
    T = {1: [{int(t[0])} for t in truth], 10: [set(t[:10]) for t in truth],
         100: [set(t) for t in truth]}

    def recall(fn, k):
        return float(np.mean([len(set(np.asarray(fn(q, k))[:k].tolist()) & T[k][i]) / k
                              for i, q in enumerate(Q)]))

    def med(fn, qs):
        ts = []
        for q in qs:
            t0 = time.time(); fn(q); ts.append(time.time() - t0)
        return 1e3 * float(np.median(ts))

    def row3(fn):                            # recall@1/@10/@100 from one k=100 call
        return recall(fn, 1), recall(fn, 10), recall(fn, 100)

    qlat = Q[:min(12, len(Q))]
    scan([(img, 1.0)], Q[0], rows=rows)     # first touch: page the file in
    t_mm = med(lambda q: scan([(img, 1.0)], q, rows=rows), qlat[:6])
    t_ram = med(lambda q: exact(q, 10), qlat)

    def opq_scan(q, k):
        return opq.opq_search(ix["codes"], opq.opq_tables(q, ix["R"], ix["books"]), k=k)[0]
    def bin_scan(q, k):
        return binary.binary_search(ix["bits"], q, k=k, rows=rows)[0]
    def i8_scan(q, k):
        return intq.i8_search(ix["i8"], ix["i8scale"], q, k=k, rows=rows)[0]
    def i8_rr(q, k):
        return intq.i8_rerank(ix["i8"], ix["i8scale"], Rimg, q, k=k, cand=args.cand, rows=rows)[0]
    def opq_rr(q, k):
        return opq.opq_rerank(ix["codes"], ix["R"], ix["books"], Rimg, q, k=k, cand=args.cand, rows=rows)[0]
    def casc(q, k):
        return engine.cascade_search(q, Rimg, ix, k=k, rows=rows)[0]

    print(f"\n  {'method':<28}{'ms/q':>8}{'R@1':>7}{'R@10':>7}{'R@100':>8}  bytes/vec")
    print(f"  {'brute f16 (memmap)':<28}{t_mm:>8.0f}{1.00:>7.2f}{1.00:>7.2f}{1.00:>8.2f}  1024  cold")
    print(f"  {'brute f32 (RAM)':<28}{t_ram:>8.1f}{1.00:>7.2f}{1.00:>7.2f}{1.00:>8.2f}  2048  serve")
    for name, fn, b, lat in [
            ("binary 1-bit (popcount)", bin_scan, 64, med(lambda q: bin_scan(q, 10), qlat)),
            ("int8 per-dim", i8_scan, 512, med(lambda q: i8_scan(q, 10), qlat[:6])),
            ("int8 → exact re-rank", i8_rr, 512, med(lambda q: i8_rr(q, 10), qlat[:6])),
            ("OPQ 64B (lookup scan)", opq_scan, 64, med(lambda q: opq_scan(q, 10), qlat)),
            ("OPQ 64B → exact re-rank", opq_rr, 64, med(lambda q: opq_rr(q, 10), qlat)),
            ("cascade (bin→OPQ→int8→exact)", casc, 64, med(lambda q: casc(q, 10), qlat))]:
        r1, r10, r100 = row3(fn)
        print(f"  {name:<28}{lat:>8.1f}{r1:>7.2f}{r10:>7.2f}{r100:>8.2f}  {b}")

    print(f"\n  IVF — recall is a DIAL, priced in rows scanned (f32 RAM):")
    print(f"  {'probes':>7}{'ms/q':>8}{'R@10':>8}{'scanned':>10}")
    for probes in (1, 2, 4, 8, 16, 32):
        def ivf_fn(q, k=10, p=probes):
            return ivfmod.ivf_search(q, [(Rimg, 1.0)], C, order, starts, k=k, probes=p)[0]
        t = med(lambda q: ivf_fn(q), qlat)
        r = recall(lambda q, k: ivf_fn(q, k), 10)
        frac = float(np.mean([ivfmod.ivf_search(q, [(Rimg, 1.0)], C, order, starts,
                                                probes=probes)[2] for q in qlat])) / rows
        print(f"  {probes:>7}{t:>8.1f}{r:>8.2f}{frac:>9.1%}")

    try:
        gpu = gpu_promote([(Rimg, 1.0)])
    except Exception:
        gpu = None
    if gpu:
        scan_gpu(gpu, Q[0])
        t_gpu = med(lambda q: scan_gpu(gpu, q), qlat)
        print(f"\n  brute on {gpu[0]}: {t_gpu:.1f} ms/q, recall 1.00 "
              f"({t_mm / t_gpu:,.0f}× the memmap scan)")

    print("\nthe cascade is the punchline: quantize at EVERY level, keep the truth —"
          "\nonly the compressed indexes in RAM, only ~100 vectors read at full precision.")
