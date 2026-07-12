"""int8 embeddings: 4× smaller, and retrieval barely notices.

pipeline: float32 vectors ──► [quantize] ──► int8 + one scale ──► same top-k

Production vector stores rarely keep float32. The simplest compression is
scalar quantization: pick ONE scale so the largest value maps to ±127,
round everything to int8, and remember the scale.

    x  ≈  q · scale        (q is int8, scale is one float for the matrix)

Dot products then run on tiny integers — and because ranking only needs
ORDER, the rounding error usually changes nothing. Don't guess: measure
the damage (that's what __main__ does).

Run me:  python3 quantize.py --json docs/db.json   # committed data, no model
"""
import argparse
import os

import numpy as np

import db


def quantize(X):
    """float32 (n,d) → (int8 (n,d), scale). One shared scale, symmetric."""
    X = np.asarray(X, dtype=np.float32)
    scale = float(np.abs(X).max()) / 127.0 or 1.0
    return np.round(X / scale).astype(np.int8), scale


def dequantize(q, scale) -> np.ndarray:
    return q.astype(np.float32) * scale


def top_neighbors(sim, k=3):
    """Each row's k best columns, self excluded — just argsort, like tagger."""
    return [[j for j in np.argsort(row)[::-1] if j != i][:k]
            for i, row in enumerate(np.asarray(sim))]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="read the web export instead of the sqlite db")
    ap.add_argument("--db", default=db.DB_PATH)
    args = ap.parse_args()
    if args.json:
        items = db.load_json_gallery(args.json)
    else:
        if not os.path.exists(args.db):
            raise SystemExit(f"no database at {args.db} — "
                             "run ingest.py first, or try --json docs/db.json")
        items = db.all_images(db.connect(args.db))
        if not items:
            raise SystemExit(f"{args.db} is empty — run ingest.py first")

    X = np.asarray([it["image_emb"] for it in items])
    q, scale = quantize(X)
    print(f"{X.shape[0]} image embeddings, {X.shape[1]}-d:")
    print(f"  float32  {X.nbytes:>6} bytes")
    print(f"  int8     {q.nbytes:>6} bytes + one float scale ({scale:.6f}) — 4× smaller")

    # integer dot products (int32 accumulator), then compare the RANKINGS
    exact = top_neighbors(X @ X.T)
    approx = top_neighbors(q.astype(np.int32) @ q.astype(np.int32).T)
    agree = sum(a == b for a, b in zip(exact, approx))
    total = len(items)
    print(f"\ntop-3 nearest neighbors, float32 vs int8: "
          f"{agree}/{total} rows identical "
          f"({agree * 3}/{total * 3} neighbor slots)")
    print("ranking needs order, not precision — that is why this works.")


if __name__ == "__main__":
    main()
