"""IVF at industrial size — ann.py's inverted-file index, sized for the million.

pipeline: N vectors ──► [ivf] ──► k-means cells ──► probe a few ──► top-k

ann.py teaches the idea on 5,000 vectors with one Python list per cell. That's
the wrong data structure at a million (a million tiny lists is its own scan), so
here the lists are stored CSR-style: one row-id array sorted by cell, plus per-
cell offsets. k-means fits on a SAMPLE (fitting cells doesn't need every point);
every row then joins its nearest cell. Search compares the query with the C
centroids (cheap), then scans ONLY the P nearest cells — recall is the dial P.

The index is keyed to the row count it was built for (ivf_stale), so a grown
corpus rebuilds it instead of silently pointing at rows that no longer exist.
"""
import os

import numpy as np

from store import CHUNK, DATA

IVF = os.path.join(DATA, "ivf.npz")     # centroids + row order (CSR) + cell offsets + row count


def ivf_train(M, rows, n_lists=1024, sample=100_000, iters=6, seed=0, log=print):
    """k-means on a sample, then assign every row to its nearest cell (chunked).
    Returns (centroids C, row order sorted by cell, per-cell start offsets)."""
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
    """Scan only the `probes` cells nearest the query, scoring their members
    EXACTLY. Returns (top-k ids, scores, rows actually scanned)."""
    near = np.argsort(q @ C.T)[::-1][:probes]
    cand = np.concatenate([order[starts[j]:starts[j + 1]] for j in near])
    s = np.zeros(len(cand), dtype=np.float32)
    for M, w in mats:
        s += w * (np.asarray(M[cand], dtype=np.float32) @ q)
    top = np.argsort(s)[::-1][:k]
    return cand[top], s[top], len(cand)


def save_ivf(C, order, starts, rows):
    np.savez(IVF, C=C, order=order, starts=starts, rows=rows)


def load_ivf():
    z = np.load(IVF)
    rows = int(z["rows"]) if "rows" in z else 0
    return z["C"], z["order"], z["starts"], rows


def ivf_stale(rows):
    if not os.path.exists(IVF):
        return True
    with np.load(IVF) as z:
        return "rows" not in z or int(z["rows"]) != rows
