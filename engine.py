"""The approximation cascade, wired into the million-scale engine.

pipeline: query ──► [binary → OPQ → int8 → exact] ──► the right top-k

cascade.py teaches this on 14 images; this runs it over the packed indexes at
a million-plus. Each level is COARSER and CHEAPER, run on a SHORTER list, and
need only keep the true neighbours ALIVE — the final exact stage, on a tiny
handful, does the ordering. So accuracy survives even though every stage before
the last is approximate. And the point for SCALE: only the compressed indexes
live in RAM (binary 1 bit/dim + OPQ 64 B + int8 512 B ≈ ⅛ the f32 towers), and
only ~100 vectors per query are ever read at full precision.

  LEVEL 0  binary  1 bit/dim, popcount        all rows → a few thousand
  LEVEL 1  OPQ     64 bytes, table lookups     thousands → hundreds
  LEVEL 2  int8    512 bytes, integer dots     hundreds → tens
  LEVEL 3  exact   f32, the truth              orders the finalists

build_indexes keeps every level self-healing: grow the corpus and each rebuilds.
"""
import time

import numpy as np

import binary
import intq
import ivf
import opq


def build_indexes(img, rows, log=print, want=("binary", "i8", "opq", "ivf")):
    """Rebuild every derived index that's missing or keyed to a different row
    count — nothing downstream can silently read a stale index for the wrong rows."""
    if "binary" in want and binary.bin_stale(rows):
        log("  building binary (1 bit/dim) …"); binary.build_binary(img, rows, log)
    if "i8" in want and intq.i8_stale(rows):
        log("  building int8 (per-dimension scale) …"); intq.build_i8(img, rows, log)
    if "opq" in want and opq.opq_stale(rows):
        log("  building OPQ (64 bytes/vector) …"); opq.build_opq(img, rows, log=log)
    if "ivf" in want and ivf.ivf_stale(rows):
        log("  building IVF (k-means on a 100k sample) …")
        t0 = time.time()
        C, order, starts = ivf.ivf_train(img, rows, log=lambda s: log(s))
        ivf.save_ivf(C, order, starts, rows)
        log(f"  IVF built in {time.time() - t0:,.0f}s")


def load_indexes(rows, to_ram=False):
    """Load the compressed image-tower indexes the cascade rides on. With
    to_ram, the binary/OPQ/int8 codes are really COPIED into RAM (≈ ⅛ the f32
    towers) — np.array, not np.asarray, since same-dtype slicing only makes a
    view; what `serve` does to search a million-plus in a fraction of the memory."""
    bits = binary.load_binary()
    R, books, codes = opq.load_opq()
    i8, i8scale = intq.load_i8()
    if to_ram:      # np.array forces a real copy off the memmap (asarray would view)
        bits = np.array(bits[:rows]); codes = np.array(codes[:rows]); i8 = np.array(i8[:rows])
    return {"bits": bits, "R": R, "books": books, "codes": codes,
            "i8": i8, "i8scale": i8scale}


def cascade_search(q, img, idx, k=10, stages=None, rows=None):
    """binary → OPQ → int8 → exact, each re-scoring a shorter list. `img` (full
    precision) is touched for ONLY the final ~stages[-1] vectors. Returns
    (ids, scores, n read at full precision) — the honest "touched" number.

    The binary top-n1 is the recall ceiling (nothing it drops is recoverable), so
    n1 grows with the corpus by default (~2% of rows): coarse but cheap, it just
    has to keep the true neighbours ALIVE for the finer stages to order."""
    n = rows if rows is not None else len(idx["bits"])
    if stages is None:
        stages = (min(n, max(4000, n // 50)), 400, 100)
    n1, n2, n3 = stages
    # LEVEL 0 — binary over all rows: the cheapest distance, the widest net
    pool = binary.binary_search(idx["bits"], q, k=n1, rows=n)[0]
    # LEVEL 1 — OPQ ADC on the pool's codes (rotated-space table lookups)
    T = opq.opq_tables(q, idx["R"], idx["books"])
    codes = np.asarray(idx["codes"])[pool]
    s = T[np.arange(codes.shape[1]), codes].sum(1)
    pool = pool[np.argsort(s)[::-1][:n2]]
    # LEVEL 2 — int8 integer dots on the shorter pool (dequant folded into q)
    s = np.asarray(idx["i8"])[pool].astype(np.float32) @ (idx["i8scale"] * q)
    pool = pool[np.argsort(s)[::-1][:n3]]
    # LEVEL 3 — exact f32, on the tiny finalist list: the ordering
    reads = len(pool)                        # the only vectors read at full precision
    e = np.asarray(img[pool], np.float32) @ q
    top = np.argsort(e)[::-1][:k]
    return pool[top], e[top], reads
