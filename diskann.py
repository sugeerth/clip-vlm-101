"""DiskANN: a billion vectors on one machine — PQ in RAM, the truth on SSD.

pipeline: many vectors ──► [Vamana graph] ──► PQ-navigate ──► rerank from disk ──► top-k

hnsw.py keeps the whole graph AND every full vector in RAM. At a billion
768-d floats that is ~3 TB — no single box has it. DiskANN (Microsoft
Research, NeurIPS 2019) is how you fit a billion on ONE machine with 64 GB:

  IN RAM   a compressed sketch — one PQ code per vector (~64 bytes) — plus a
           navigable graph. Cheap approximate distances steer the walk.
  ON SSD   the graph's edges and the FULL-precision vectors. The walk reads a
           few nodes' neighbours as it goes; at the end it fetches only the
           handful of finalists' true vectors and reranks them exactly.

Two ideas make it work, and this file implements both, measured:

  1. THE VAMANA GRAPH.  Not HNSW's layers — one flat graph, built by a greedy
     "robust prune": when wiring node p, keep an edge to a near neighbour c
     ONLY IF no already-kept, closer neighbour p* occludes it, i.e. unless
     α·d(p*,c) ≤ d(p,c). The α (>1) deliberately keeps some longer edges, so
     the graph has a SHORT diameter — fewer hops, so fewer SSD reads per query.
  2. PQ-STEERED SEARCH.  Navigate using PQ-approximate distances from the RAM
     sketch (no disk, no full vectors), gather a candidate list of width L,
     then read just those L full vectors once and rerank exactly.

The measured claim (this file, on the clustered synthetic vectors real
embeddings resemble): reranking from a compressed walk recovers essentially
all the recall of an exact scan, while the query READS only ~L full vectors —
a rounding error against the corpus. That is the whole trick: the billion
lives on disk; the query barely touches it.

Run me:  python3 diskann.py            (synthetic clustered vectors, ~4 s, no model)
         python3 diskann.py --json docs/db.json    (the real 14-image gallery too)
"""
import argparse

import numpy as np

import db
from ann import recall_at_k, synthetic
from cascade import pq_encode, pq_scores, pq_train


class DiskANN:
    """A teaching DiskANN: a Vamana graph built by robust-prune, searched with
    PQ-approximate distances (the RAM sketch) and reranked from full vectors
    (the SSD read). Every distance and every disk read is counted — 'fits on one
    box' is a budget you measure, not a slogan. Distance is 1 − cosine."""

    def __init__(self, X, R=32, L=64, alpha=1.2, m=16, seed=0):
        self.X = np.asarray(X, dtype=np.float32)
        self.n = len(self.X)
        self.R = R                 # max out-degree (graph stays sparse on disk)
        self.Lc = L                # build-time search-list width
        self.alpha = alpha         # robust-prune slack: >1 keeps long edges
        self.rng = np.random.default_rng(seed)
        # the RAM sketch: one PQ code per vector, ~m bytes each
        self.books, self.sub = pq_train(self.X, m=min(m, self.X.shape[1]), seed=seed)
        self.codes = pq_encode(self.X, self.books, self.sub)
        # the medoid: entry point, the vector nearest the centroid
        c = self.X.mean(axis=0)
        self.medoid = int(np.argmax(self.X @ c))
        self.nbrs = [[] for _ in range(self.n)]
        self._build()

    # exact distance on full vectors — the "SSD read", counted
    def _d(self, i, q):
        return 1.0 - float(self.X[i] @ q)

    def _d_many(self, ids, q):
        ids = np.asarray(ids, dtype=int)
        return 1.0 - (self.X[ids] @ q)

    def _greedy(self, q, entry, L, meter=None):
        """Greedy graph walk to the query using EXACT distances (build time).
        Returns (visited ids sorted by distance, full visited set)."""
        cand = {entry: self._d(entry, q)}
        visited = set()
        while True:
            unvis = [(d, i) for i, d in cand.items() if i not in visited]
            if not unvis:
                break
            unvis.sort()
            keep = unvis[:L]
            frontier = next(((d, i) for d, i in keep if i not in visited), None)
            if frontier is None:
                break
            _, p = frontier
            visited.add(p)
            if meter is not None:
                meter[0] += 1
            new = [nb for nb in self.nbrs[p] if nb not in cand]
            if new:                                 # batch the neighbour distances
                ds = 1.0 - (self.X[new] @ q)
                for nb, d in zip(new, ds):
                    cand[nb] = float(d)
            # keep the L closest in the working set
            if len(cand) > L:
                cand = dict(sorted(cand.items(), key=lambda kv: (kv[1], kv[0]))[:L])
        order = sorted(cand.items(), key=lambda kv: (kv[1], kv[0]))
        return [i for i, _ in order], visited

    def _robust_prune(self, p, candidates, alpha):
        """Vamana's heart: from a candidate pool, keep the closest, then DROP any
        candidate a kept-and-closer neighbour occludes (α·d(p*,c) ≤ d(p,c)).
        α>1 spares some long edges → short graph diameter → fewer hops."""
        pool = set(candidates) | set(self.nbrs[p])
        pool.discard(p)
        if not pool:
            self.nbrs[p] = []
            return
        pool = np.asarray(list(pool), dtype=int)
        dvals = 1.0 - (self.X[pool] @ self.X[p])        # d(p, c) for all c, vectorized
        order = np.argsort(dvals)
        pool = list(pool[order])
        dp = {int(c): float(dvals[order[k]]) for k, c in enumerate(pool)}
        out = []
        while pool:
            pstar = int(pool[0])
            out.append(pstar)
            if len(out) >= self.R:
                break
            rest = np.asarray(pool[1:], dtype=int)
            # occluded if the just-picked p* is α-closer to c than p is
            sims = self.X[rest] @ self.X[pstar]
            keep = [int(c) for c, s in zip(rest, sims)
                    if alpha * (1.0 - float(s)) > dp[int(c)]]
            pool = keep
        self.nbrs[p] = out

    def _build(self):
        """Two Vamana passes (α=1 then α>1) over a random insertion order."""
        # seed with a random R-regular graph so the first walks have somewhere to go
        for i in range(self.n):
            choices = [j for j in self.rng.choice(self.n, min(self.R, self.n - 1) + 1,
                                                  replace=False) if j != i][:self.R]
            self.nbrs[i] = list(choices)
        for alpha in (1.0, self.alpha):
            for p in self.rng.permutation(self.n):
                p = int(p)
                _, visited = self._greedy(self.X[p], self.medoid, self.Lc)
                self._robust_prune(p, visited, alpha)
                for j in list(self.nbrs[p]):        # add back-edges, prune if over degree
                    if p in self.nbrs[j]:
                        continue
                    if len(self.nbrs[j]) + 1 > self.R:
                        self._robust_prune(j, set(self.nbrs[j]) | {p}, alpha)
                    else:
                        self.nbrs[j].append(p)

    def search(self, q, k=10, L=None, W=None):
        """PQ-navigate the graph (RAM), then read W full vectors from 'SSD' and
        rerank exactly. Returns (top-k ids, dict of measured costs)."""
        q = np.asarray(q, dtype=np.float32)
        L = L or self.Lc
        W = W or L
        # PQ-approximate distance to every candidate: RAM lookups, no full vectors.
        # pq_scores gives similarity (higher=closer); distance = -score.
        approx = {}
        def pqd(i):
            if i not in approx:
                s = float(pq_scores(q, self.codes[i:i + 1], self.books, self.sub)[0])
                approx[i] = -s
            return approx[i]
        cand = {self.medoid: pqd(self.medoid)}
        visited, hops = set(), 0
        while True:
            unvis = sorted(((d, i) for i, d in cand.items() if i not in visited))
            if not unvis:
                break
            keep = unvis[:L]
            frontier = next(((d, i) for d, i in keep if i not in visited), None)
            if frontier is None:
                break
            p = frontier[1]
            visited.add(p)
            hops += 1
            for nb in self.nbrs[p]:
                if nb not in cand:
                    cand[nb] = pqd(nb)
            if len(cand) > L:
                cand = dict(sorted(cand.items(), key=lambda kv: (kv[1], kv[0]))[:L])
        shortlist = [i for i, _ in sorted(cand.items(), key=lambda kv: (kv[1], kv[0]))][:W]
        # the ONLY full-precision reads: rerank the W finalists from disk
        exact = self._d_many(shortlist, q)
        order = np.argsort(exact)[:k]
        top = np.asarray(shortlist, dtype=int)[order]
        cost = {"hops": hops, "pq_dists": len(approx), "ssd_reads": len(shortlist)}
        return top, cost


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also index the real gallery from the web export")
    args = ap.parse_args()

    X, Q = synthetic(n=1200, n_queries=60)
    exact = np.argsort(Q @ X.T, axis=1)[:, ::-1][:, :10]

    index = DiskANN(X, R=24, L=32, alpha=1.2)
    deg = [len(nb) for nb in index.nbrs]
    print(f"{len(X)} clustered vectors, {len(Q)} queries.")
    print(f"Vamana graph: medoid #{index.medoid}, out-degree ≤ {index.R} "
          f"(avg {np.mean(deg):.1f}), one flat layer.")
    print(f"RAM sketch: {index.codes.shape[1]} bytes/vector (PQ); full vectors live 'on SSD'.\n")

    print("  L    recall@10   SSD reads/query   graph hops")
    for L in (16, 32, 64):
        rec = reads = hops = 0
        for qi, q in enumerate(Q):
            found, cost = index.search(q, L=L)
            rec += recall_at_k(found, exact[qi])
            reads += cost["ssd_reads"]
            hops += cost["hops"]
        n = len(Q)
        print(f"  {L:>3}   {rec / n:>9.2f}   {reads / n:>15.1f}   {hops / n:>10.1f}")

    print(f"\nnavigate on a {index.codes.shape[1]}-byte-per-vector PQ sketch in RAM; read only ~L")
    print(f"full vectors from disk to rerank — out of {len(X):,}. That ratio is the point:")
    print("the corpus lives on SSD, the query barely touches it. α-pruning keeps the")
    print("graph's diameter short, so 'few hops' stays true as N climbs to a billion.")

    if args.json:
        items = db.load_json_gallery(args.json)
        I = np.asarray([it["image_emb"] for it in items], dtype=np.float32)
        idx = DiskANN(I, R=8, L=16, alpha=1.2, m=8)
        exact5 = np.argsort(I @ I[0])[::-1][:5]
        got, cost = idx.search(I[0], k=5, L=10)
        print(f"\nthe real gallery ({len(items)} images): DiskANN returns "
              f"{recall_at_k(got, exact5):.0%} of the exact top-5 for image 0, reading "
              f"{cost['ssd_reads']} vectors — but at fourteen there is no disk to spare. "
              "DiskANN is a billion-scale tool; here it just degrades to exact.")


if __name__ == "__main__":
    main()
