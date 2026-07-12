"""Approximate nearest neighbours: scan a sliver, keep most of the truth.

pipeline: many vectors ──► [ann] ──► coarse lists ──► probe a few ──► top-k

Exact search is one dot product per stored vector — fine at 14, painful at
14 million. Every vector database's answer is the same trick, IVF (the
"inverted file" index, the heart of FAISS):

    1. TRAIN   k-means the vectors into C coarse cells
    2. INDEX   every vector joins the list of its nearest centroid
    3. SEARCH  compare the query with the C centroids (cheap), then scan
               ONLY the P nearest lists — "probes"

Why it works: embeddings CLUSTER (look at the demo's 2-D map — animals
land near animals). A query's true neighbours almost all live in a few
nearby cells, so scanning those cells finds most of them. The dial is
probes P: more probes = more of the truth, more scanning. Measure it —
recall@k = the fraction of the TRUE top-k the shortcut kept.

Run me:  python3 ann.py            (synthetic clustered vectors, ~2 s, no model)
         python3 ann.py --json docs/db.json    (the real 14-image gallery too)
"""
import argparse

import numpy as np

import db


def kmeans(X, c, iters: int = 5, seed: int = 0):
    """Spherical k-means: farthest-point init, then Lloyd iterations.
    Returns (centroids (c,d) unit rows, assignment (n,))."""
    rng = np.random.default_rng(seed)
    cents = [X[rng.integers(len(X))]]
    for _ in range(c - 1):  # farthest-point init: spread the seeds out
        d = 1 - np.max(np.stack([X @ ct for ct in cents]), axis=0)
        cents.append(X[int(np.argmax(d))])
    C = np.stack(cents)
    for _ in range(iters):
        assign = np.argmax(X @ C.T, axis=1)
        for j in range(c):
            members = X[assign == j]
            if len(members):
                v = members.mean(axis=0)
                C[j] = v / np.linalg.norm(v)
    return C, np.argmax(X @ C.T, axis=1)


def build(X, n_lists: int = 64, seed: int = 0):
    """The index: centroids + one inverted list of row ids per centroid."""
    C, assign = kmeans(X, n_lists, seed=seed)
    return C, [np.where(assign == j)[0] for j in range(n_lists)]


def search(q, X, centroids, lists, k: int = 10, probes: int = 4):
    """Scan only the `probes` nearest lists. Returns (top-k ids, n scanned)."""
    near = np.argsort(q @ centroids.T)[::-1][:probes]
    cand = np.concatenate([lists[j] for j in near])
    order = np.argsort(q @ X[cand].T)[::-1][:k]
    return cand[order], len(cand)


def recall_at_k(found, truth) -> float:
    return len(set(found) & set(truth)) / len(truth)


def synthetic(n=5000, n_queries=100, dim=64, blobs=64, noise=0.2, seed=0):
    """Clustered unit vectors — the structure real embeddings actually have.
    (Uniformly random vectors are IVF's worst case; embeddings aren't that.)"""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(blobs, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    make = lambda m: (lambda V: V / np.linalg.norm(V, axis=1, keepdims=True))(
        centers[rng.integers(0, blobs, m)] + noise * rng.normal(size=(m, dim)))
    return make(n), make(n_queries)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also index the real gallery from the web export")
    args = ap.parse_args()

    X, Q = synthetic()
    C, lists = build(X)
    exact = np.argsort(Q @ X.T, axis=1)[:, ::-1][:, :10]
    print(f"{len(X)} clustered vectors, {len(lists)} lists, {len(Q)} queries — "
          "recall@10 vs how much was scanned:\n")
    print("  probes   recall@10   scanned")
    for probes in (1, 2, 4, 8):
        rec = scanned = 0
        for qi, q in enumerate(Q):
            found, n = search(q, X, C, lists, probes=probes)
            rec += recall_at_k(found, exact[qi])
            scanned += n
        print(f"  {probes:>6}   {rec / len(Q):>9.2f}   {scanned / len(Q) / len(X):>6.1%}")
    print("\nscan ~2% of the data, keep ~3/4 of the truth; ~13% keeps ~94%.")
    print("more probes = more truth, more work — recall is a DIAL, not a given.")

    if args.json:
        items = db.load_json_gallery(args.json)
        I = np.asarray([it["image_emb"] for it in items])
        C, lists = build(I, n_lists=4)
        sizes = sorted(len(l) for l in lists)
        print(f"\nthe real gallery, {len(items)} images in 4 lists {sizes}: at this "
              "size just scan everything — ANN pays off at thousands, not fourteen.")


if __name__ == "__main__":
    main()
