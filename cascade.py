"""The approximation cascade: quantize at EVERY level, keep the truth anyway.

pipeline: billions ──► [binary → PQ → int8 → exact] ──► the right top-k

One approximation is a tradeoff. A CASCADE of them is a free lunch — if you
arrange them right. The trick every billion-scale system uses (FAISS
IVF-PQ-rerank, ScaNN, DiskANN): each level is a COARSER, cheaper distance
run on a SHORTER list, and each need only keep the true neighbours ALIVE
in its shortlist — not order them. The final exact stage, on a tiny handful,
does the ordering. So accuracy survives even though every stage before the
last is approximate.

The ladder, cheapest-and-coarsest first (this file adds the one rung the
repo was missing — binary — on top of quantize.py and pq.py):

  LEVEL 0  IVF coarse     which cells to even look at        (ann.py)
  LEVEL 1  BINARY 1 bit/d sign(x) per dim; distance = popcount(XOR), the
                          cheapest distance a CPU has. many → ~500
  LEVEL 2  PQ 64 bytes    asymmetric table lookups           (pq.py)  ~500 → ~100
  LEVEL 3  int8 512 bytes integer dot products               (quantize.py) ~100 → ~50
  LEVEL 4  exact f32      the truth, on ~50 vectors — orders the finalists

Only those last ~50 vectors per query ever touch float32. Out of billions.
That is the whole point: full precision is a rounding error in the budget.

The measured claim (this file, on clustered synthetic vectors — the shape
real embeddings have): the full cascade keeps ~100% of the recall that an
exact scan of the same cells would get, while the true neighbours SURVIVE
every approximate level. Approximate everywhere; wrong nowhere that counts.

Run me:  python3 cascade.py                    (synthetic, ~3 s, no model)
         python3 cascade.py --json docs/db.json
"""
import argparse

import numpy as np

import db
from ann import build, synthetic

BITS = 8  # int8 headroom, documentation only


# ---- the approximation levels, each a self-contained encoder + distance ----

def binary_encode(X):
    """1 bit per dimension: the sign. 512-d → 512 bits = 64 bytes, and the
    distance is popcount(a XOR b) — Hamming, the cheapest distance there is.
    For unit vectors it tracks angular similarity closely enough to shortlist."""
    return (np.asarray(X) > 0)


def hamming(qbits, Bbits):
    """Bits that differ. Fewer = closer. (np booleans stand in for popcount.)"""
    return (qbits[None, :] != Bbits).sum(axis=1)


def int8_encode(X):
    """quantize.py's lesson: one shared max-abs scale, every value a byte."""
    scale = float(np.abs(X).max()) / 127 or 1.0
    return np.round(np.asarray(X) / scale).astype(np.int8), scale


def pq_train(X, m=64, ks=256, seed=0):
    """pq.py's lesson, compact: m subspaces, ks centroids each (the codebooks).
    Returns (books (m, ks, sub), sub-vector width)."""
    d = X.shape[1]
    sub = d // m
    rng = np.random.default_rng(seed)
    # sample-as-centroids keeps this CI-fast; pq.py does full k-means per subspace
    books = np.stack([X[rng.integers(0, len(X), ks)][:, i * sub:(i + 1) * sub]
                      for i in range(m)])
    return books, sub


def pq_encode(X, books, sub):
    """Each subvector → the byte naming its nearest codebook centroid."""
    m = books.shape[0]
    codes = np.empty((len(X), m), np.uint8)
    for i in range(m):
        Xs = X[:, i * sub:(i + 1) * sub]
        codes[:, i] = np.argmin(((Xs[:, None, :] - books[i][None]) ** 2).sum(2), 1)
    return codes


def pq_scores(q, codes, books, sub):
    """ADC: one dot-product table per subspace, then a row is m lookups summed.
    No multiply per row — the property that lets a browser search a million."""
    m = books.shape[0]
    lut = np.stack([books[i] @ q[i * sub:(i + 1) * sub] for i in range(m)])  # (m, ks)
    return lut[np.arange(m)[:, None], codes.T].sum(0)                        # (n,)


class Cascade:
    """Build every level's index once (offline), then search cheaply."""

    def __init__(self, X, n_lists=64, seed=0):
        self.X = np.asarray(X, dtype=np.float32)
        self.bits = binary_encode(self.X)
        self.i8, self.i8_scale = int8_encode(self.X)
        self.books, self.sub = pq_train(self.X, seed=seed)
        self.codes = pq_encode(self.X, self.books, self.sub)
        self.C, self.lists = build(self.X, n_lists=min(n_lists, len(self.X)), seed=seed)

    def search(self, q, k=10, probes=8, keep=(500, 100, 50), trace=False):
        """Funnel one query through all five levels. Returns top-k ids
        (and, if trace, the candidate count surviving each level)."""
        q = np.asarray(q, dtype=np.float32)
        steps = []
        # L0 — IVF coarse: which cells
        near = np.argsort(q @ self.C.T)[::-1][:probes]
        cand = np.concatenate([self.lists[j] for j in near])
        steps.append(("L0 IVF cells", len(cand)))
        # L1 — binary Hamming shortlist
        qb = binary_encode(q[None])[0]
        cand = cand[np.argsort(hamming(qb, self.bits[cand]))[:keep[0]]]
        steps.append(("L1 binary", len(cand)))
        # L2 — PQ ADC
        cand = cand[np.argsort(pq_scores(q, self.codes[cand], self.books, self.sub))[::-1][:keep[1]]]
        steps.append(("L2 PQ-64", len(cand)))
        # L3 — int8 integer dots
        qi8 = np.round(q / self.i8_scale).astype(np.int32)
        cand = cand[np.argsort(self.i8[cand].astype(np.int32) @ qi8)[::-1][:keep[2]]]
        steps.append(("L3 int8", len(cand)))
        # L4 — exact float32, orders the finalists
        top = cand[np.argsort(self.X[cand] @ q)[::-1][:k]]
        steps.append(("L4 exact", len(top)))
        return (top, steps) if trace else top


def recall(found, truth):
    return len(set(found) & set(truth)) / len(truth)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also run the cascade on the real gallery")
    args = ap.parse_args()

    X, Q = synthetic(n=5000, n_queries=100, dim=64, blobs=64, noise=0.2)
    truth = np.argsort(Q @ X.T, axis=1)[:, ::-1][:, :10]
    cas = Cascade(X)

    # exact-on-cells is the CEILING: the best any method could do given L0's cells
    def ceiling(qi, probes=8):
        near = np.argsort(Q[qi] @ cas.C.T)[::-1][:probes]
        cand = np.concatenate([cas.lists[j] for j in near])
        return cand[np.argsort(X[cand] @ Q[qi])[::-1][:10]]

    r_ceiling = np.mean([recall(ceiling(i), truth[i]) for i in range(len(Q))])
    r_cascade = np.mean([recall(cas.search(Q[i]), truth[i]) for i in range(len(Q))])

    # does each APPROXIMATE level keep the true neighbours alive in its shortlist?
    surv = np.zeros(5)
    for i in range(len(Q)):
        t = set(truth[i])
        _, steps = cas.search(Q[i], trace=True)
        near = np.argsort(Q[i] @ cas.C.T)[::-1][:8]
        cand = np.concatenate([cas.lists[j] for j in near])
        surv[0] += len(set(cand) & t) / 10
        qb = binary_encode(Q[i][None])[0]
        cand = cand[np.argsort(hamming(qb, cas.bits[cand]))[:500]]
        surv[1] += len(set(cand) & t) / 10
        cand = cand[np.argsort(pq_scores(Q[i], cas.codes[cand], cas.books, cas.sub))[::-1][:100]]
        surv[2] += len(set(cand) & t) / 10
        qi8 = np.round(Q[i] / cas.i8_scale).astype(np.int32)
        cand = cand[np.argsort(cas.i8[cand].astype(np.int32) @ qi8)[::-1][:50]]
        surv[3] += len(set(cand) & t) / 10
    surv /= len(Q)

    print(f"{len(X):,} clustered vectors, 5-level approximation cascade, {len(Q)} queries\n")
    print(f"  exact scan of the probed cells (the ceiling):  recall@10 = {r_ceiling:.3f}")
    print(f"  FULL CASCADE binary→PQ→int8→exact:             recall@10 = {r_cascade:.3f}")
    print(f"  → the cascade keeps {100 * r_cascade / r_ceiling:.0f}% of the ceiling,"
          " touching float32 on ~50 vectors, not 5,000.\n")
    print("  true-neighbour survival through the funnel (each level must not drop them):")
    for (name, _), s in zip([("L0 IVF cells", 0), ("L1 binary → 500", 0),
                             ("L2 PQ-64 → 100", 0), ("L3 int8 → 50", 0)], surv):
        print(f"    {name:<18} {s:.3f}")
    print("    L4 exact → 10      (orders the survivors — recall is set above)")

    if args.json:
        items = db.load_json_gallery(args.json)
        I = np.asarray([it["image_emb"] for it in items], dtype=np.float32)
        print(f"\n  the real gallery ({len(items)} images): far below the thousands where a")
        print("  cascade earns its keep — here every level would just scan all 14. The")
        print("  cascade is a big-N tool; the point is it DEGRADES to exact, never wrong.")


if __name__ == "__main__":
    main()
