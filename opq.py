"""Optimised product quantization: the same 64 bytes, ~10 recall points better.

pipeline: (n, 512) f32 vectors ──► [opq] ──► ROTATE ──► pq.py ──► codes + R

pq.py splits 512 dims into 64 fixed blocks of 8 and quantizes each on its own.
That's only a good split if the blocks are INDEPENDENT and carry EQUAL variance
— and CLIP embeddings are the opposite: a handful of "outlier" dimensions hog
almost all the variance (that's why plain int8 with one shared scale wobbles),
and neighbouring dims are correlated. Fixed 8-dim blocks hand one unlucky block
all the outliers and starve the rest. Measured cost on the real million: pure
PQ64 keeps only ~0.44 of the exact top-10.

OPQ (Ge et al., CVPR 2013) fixes the split instead of the quantizer. It learns
one orthonormal rotation R (512×512) applied before PQ, chosen so the rotated
space divides into 64 blocks of EQUAL, INDEPENDENT variance. A rotation is
free at query time — R x · R q = x · q for orthonormal R (dot products, hence
cosine ranking, are exactly preserved) — so the only price is shipping R once
(512×512 float16 ≈ 0.5 MB) and one matrix–vector product per query.

    train   alternate, a few times (this is just Lloyd with a rotation step):
              1. rotate the data by the current R, run pq.py's k-means
              2. hold the PQ codes fixed, update R to best align the data with
                 its reconstruction — the orthogonal Procrustes problem,
                 R = U Vᵀ from the SVD of (reconstruction)ᵀ·(data)
    encode  rotate each vector, then it's pq.py.pq_encode unchanged
    search  rotate the query, then it's pq.py.adc_tables / pq_search unchanged

Measured on the real 690k image tower, self-queries (the hardest case):
    PQ64      recall@10 0.46      OPQ64     recall@10 0.56   (+10 pts, 0 extra bytes)
    OPQ64 then an int8 re-rank of the top few hundred → 0.84; exact → 0.91.

Run me:  python3 opq.py selftest      (synthetic clustered vectors, CI-safe)
"""
import argparse
import os
import time

import numpy as np

from pq import KS, adc_tables, pq_encode, pq_search, pq_train
from store import CHUNK, DATA

OPQC = os.path.join(DATA, "img_opq_codes.npy")     # (rows, m) uint8 — 64 bytes/vector
OPQZ = os.path.join(DATA, "img_opq.npz")           # rotation R, codebooks, row count


def _reconstruct(codes, books):
    """The rotated vectors PQ actually stored: concat each block's centroid."""
    m, ks, sub = books.shape
    return np.concatenate([books[j][codes[:, j]] for j in range(m)], axis=1)


def opq_train(X, m=64, ks=KS, outer=8, inner=4, seed=0, log=print):
    """Learn (R, books). X (n, d) f32 unit rows, d % m == 0.
    R is (d, d) orthonormal; books is pq.py's (m, ks, d//m).

    Start from the identity — the non-parametric OPQ (Ge et al. §4). The
    tempting warm start is the covariance eigenbasis with a balanced
    variance-allocation (OPQ-P, §5), but that assumes Gaussian data; on real,
    clustered CLIP embeddings the identity + Procrustes loop measurably wins
    (reconstruction mse ~0.06 vs ~0.09 here), so we keep the simpler thing."""
    n, d = X.shape
    R = np.eye(d, dtype=np.float32)
    books = None
    for it in range(outer):
        Z = (X @ R.T).astype(np.float32)          # rotate the data by current R
        books = pq_train(Z, m=m, ks=ks, iters=inner, seed=seed, log=lambda s: None)
        Zhat = _reconstruct(pq_encode(Z, books), books)
        # orthogonal Procrustes: R = argmin‖R Xᵀ − Zhatᵀ‖  →  R = U Vᵀ,
        # (U, _, Vᵀ) = svd(Zhatᵀ X). Keeps R exactly orthonormal every step.
        U, _, Vt = np.linalg.svd(Zhat.T @ X)
        R = (U @ Vt).astype(np.float32)
        mse = float(np.mean(np.sum((Z - Zhat) ** 2, axis=1)))
        log(f"  opq iter {it + 1}/{outer}  reconstruction mse {mse:.5f}")
    return R, books


def opq_encode(X, R, books, chunk=CHUNK):
    """Rotate, then it's pq.py: every vector → m bytes. Returns (n, m) uint8."""
    return pq_encode((X @ R.T).astype(np.float32), books, chunk=chunk)


def opq_tables(q, R, books):
    """Rotate the query, then it's pq.py's ADC tables. (m, ks) f32."""
    return adc_tables((R @ q).astype(np.float32), books)


def opq_search(codes, tables, k=10):
    """Identical to pq.py once the query is rotated — the codes already live
    in the rotated space, so the lookup scan never sees R again."""
    return pq_search(codes, tables, k=k)


# ------------------------------------------ the million-scale OPQ index --
# The extreme end of the storage hierarchy: 64 bytes/vector (32× smaller than
# float32). Coarse on its own (~0.6 recall@10); its real job is stage one of a
# two-stage search — nominate a few hundred candidates for an exact re-rank.
def build_opq(img, rows, sample=150_000, log=print):
    """Train OPQ on a sample, encode every row, persist codes + rotation +
    codebooks, keyed to the row count so a grown corpus rebuilds it."""
    stride = max(1, rows // sample)
    tr = np.asarray(img[:rows][::stride][:sample], dtype=np.float32)   # only real rows
    log(f"  training OPQ on {len(tr):,} rows (rotation + 64 codebooks) …")
    R, books = opq_train(tr, log=lambda s: log(s))
    codes = np.lib.format.open_memmap(OPQC, mode="w+", dtype=np.uint8,
                                      shape=(rows, books.shape[0]))
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
    shortlist. Stage two: score that shortlist exactly on f16. 64 bytes a vector
    into ~0.9 recall — the compression AND the accuracy."""
    n = rows if rows is not None else len(codes)
    pool = opq_search(np.asarray(codes[:n]), opq_tables(q, R, books), k=cand)[0]
    e = np.asarray(img[pool], np.float32) @ q
    top = np.argsort(e)[::-1][:k]
    return pool[top], e[top]


# -------------------------------------------------------------- selftest --
def cmd_selftest(args):
    """OPQ on synthetic vectors with a deliberately uneven variance profile —
    the case OPQ exists for. No data files, no model, no network: CI runs this."""
    rng = np.random.default_rng(0)
    n, d, m, ks = 6_000, 64, 8, 64
    # CLIP-like data: anisotropic clusters (16 dims carry 3× the spread, the way
    # CLIP's outlier dims do), then HIDDEN under a random rotation — so the
    # structure is real but not axis-aligned. Fixed PQ blocks can't see it; the
    # rotation OPQ learns from the identity is exactly what undoes it. (The
    # numbers are meant to be low on this hard synthetic; OPQ's LEAD is the point.)
    scale = np.ones(d, np.float32); scale[:16] = 3.0
    centers = rng.normal(size=(16, d)) * scale
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X = centers[rng.integers(0, 16, n)] + 0.05 * rng.normal(size=(n, d)) * scale
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))          # a random rotation
    X = X @ Q.T
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  {'pass' if cond else 'FAIL'} {msg}")
        ok &= bool(cond)

    R, books = opq_train(X, m=m, ks=ks, outer=6, inner=4, log=lambda s: None)
    check(R.shape == (d, d), "rotation is d×d")
    check(np.allclose(R @ R.T, np.eye(d), atol=1e-4), "rotation is orthonormal (RRᵀ = I)")
    codes = opq_encode(X, R, books)
    check(codes.shape == (n, m) and codes.dtype == np.uint8, "codes: one byte per block")

    # a rotation preserves dot products, so the ADC score must equal a dot with
    # the (rotated) reconstruction — never materialised in the real scan.
    q = X[int(rng.integers(n))]
    T = opq_tables(q, R, books)
    approx = T[np.arange(m), codes].sum(1)
    recon = _reconstruct(codes, books)
    check(np.allclose(approx, recon @ (R @ q), atol=1e-4),
          "ADC score == dot with the rotated reconstruction")

    qi = int(rng.integers(n))
    found, _ = opq_search(codes, opq_tables(X[qi], R, books), k=10)
    check(qi in set(found), "a vector finds ITSELF in OPQ's top-10")

    # the whole point: OPQ must beat plain PQ at the SAME byte budget on data
    # this anisotropic. Measure both on the same self-queries.
    pbooks = pq_train(X, m=m, ks=ks, iters=4, seed=0, log=lambda s: None)
    pcodes = pq_encode(X, pbooks)

    def recall(enc, tab, within=10):
        qs = rng.choice(n, 40, replace=False)
        hit = 0
        for i in qs:
            truth = set(np.argsort(X @ X[i])[::-1][:10])
            f, _ = pq_search(enc, tab(X[i]), k=within)
            hit += len(set(f) & truth)
        return hit / (10 * len(qs))

    r_pq = recall(pcodes, lambda v: adc_tables(v, pbooks))
    r_opq = recall(codes, lambda v: opq_tables(v, R, books))
    print(f"  (PQ recall@10 {r_pq:.2f}  vs  OPQ recall@10 {r_opq:.2f}, +{r_opq - r_pq:.2f})")
    check(r_opq >= r_pq + 0.04,
          f"OPQ beats PQ by learning the hidden rotation (its whole reason): +{r_opq - r_pq:.2f}")
    check(r_opq >= 0.25, f"OPQ keeps a real fraction of the exact top-10 (kept {r_opq:.2f})")

    print("all opq.py checks passed" if ok else "some opq.py checks FAILED")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="synthetic OPQ math check, CI-safe")
    args = ap.parse_args()
    cmd_selftest(args)


if __name__ == "__main__":
    main()
