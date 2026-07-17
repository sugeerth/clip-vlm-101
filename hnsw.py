"""HNSW: don't sort the haystack, walk a map of it. O(log N), not O(√N).

pipeline: many vectors ──► [layered graph] ──► greedy-walk ──► top-k

ann.py's IVF asks "which cells?" and scans them — recall is a DIAL you pay
for in probes, and the work grows like √N. Every modern vector database
(Qdrant, Weaviate, pgvector, Vespa, FAISS-HNSW) reaches for a different
idea instead: Hierarchical Navigable Small World graphs.

The picture is an airline map. Build a graph where each vector links to its
nearest few neighbours, and stack it in LAYERS: layer 0 has everyone with
short local hops; each layer up keeps a sparse random sample with long
hops. To search, drop in at the top, greedily walk toward the query along
long hops (cross the continent in a few jumps), descend a layer, repeat.
By layer 0 you're already in the right neighbourhood and only refine
locally. The number of hops grows like log N, not √N — that is the whole
game at a billion vectors.

    BUILD   insert one at a time; each node gets a random top level
            L = ⌊-ln(u)·mL⌋ (geometric — most live only on layer 0).
            link it to the M nearest already-inserted, both directions,
            pruning everyone back to M so the graph stays sparse.
    SEARCH  greedy-descend the sparse upper layers to a good entry point,
            then a best-first beam (width ef) on layer 0. Bigger ef =
            more of the truth, more hops — recall is still a dial, but a
            far cheaper one than IVF's.

Measured here on the SAME clustered synthetic vectors ann.py uses, so the
comparison is apples to apples: at matched work (distance computations per
query) HNSW keeps more of the true top-k than IVF. Graph beats cells.

Run me:  python3 hnsw.py            (synthetic clustered vectors, ~3 s, no model)
         python3 hnsw.py --json docs/db.json    (the real 14-image gallery too)
"""
import argparse
import heapq
import math

import numpy as np

import db
from ann import build as ivf_build, recall_at_k, search as ivf_search, synthetic


class HNSW:
    """A teaching HNSW: correct in shape, deterministic, and instrumented so
    every distance computation is counted — because 'fast' is a measurement,
    not an adjective. Distance is 1 − cosine on unit vectors (smaller = nearer)."""

    def __init__(self, X, M=16, ef_construction=64, seed=0):
        self.X = np.asarray(X, dtype=np.float32)
        self.M = M                     # neighbours kept per node per layer
        self.M0 = 2 * M                # layer 0 is denser (the standard choice)
        self.efc = ef_construction
        self.mL = 1.0 / math.log(M)    # level multiplier: E[levels] ≈ 1/ln M
        self.rng = np.random.default_rng(seed)
        self.layers = []               # layers[l][node] -> list of neighbour ids
        self.entry = None              # top-layer entry point
        self.top = -1
        self.dist_calls = 0            # the honest cost meter
        for i in range(len(self.X)):
            self._insert(i)

    # --- distance: 1 - cosine, counted every time it is paid ---
    def _d(self, i, q):
        self.dist_calls += 1
        return 1.0 - float(self.X[i] @ q)

    def _d_many(self, ids, q):
        self.dist_calls += len(ids)
        return 1.0 - (self.X[ids] @ q)

    def _random_level(self):
        u = float(self.rng.random())
        u = min(max(u, 1e-12), 1 - 1e-12)   # keep the log finite
        return int(-math.log(u) * self.mL)

    def _search_layer(self, q, entries, ef, l):
        """Best-first beam on one layer. `entries` are seed ids; returns the ef
        closest ids found, staying within layer l's edges. The core walk."""
        visited = set(entries)
        # min-heap of (dist, id) as the frontier; max-heap (neg) of the ef best
        cand = []
        best = []
        for e in entries:
            de = self._d(e, q)
            heapq.heappush(cand, (de, e))
            heapq.heappush(best, (-de, e))
        while cand:
            d, c = heapq.heappop(cand)
            if best and -best[0][0] < d and len(best) >= ef:
                break                       # frontier is worse than our worst kept
            for nb in self.layers[l].get(c, ()):  # deterministic: insertion order
                if nb in visited:
                    continue
                visited.add(nb)
                dn = self._d(nb, q)
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(cand, (dn, nb))
                    heapq.heappush(best, (-dn, nb))
                    if len(best) > ef:
                        heapq.heappop(best)
        # closest-first, tie-broken by id so Python and any twin would agree
        return [i for _, i in sorted((-nd, i) for nd, i in best)]

    def _select_neighbours(self, q, cand_ids, m):
        """Keep the m closest of the candidates (simple, deterministic heuristic).
        Ties break by id so the graph is a pure function of the input order."""
        ds = self._d_many(cand_ids, q)
        order = sorted(range(len(cand_ids)), key=lambda k: (float(ds[k]), int(cand_ids[k])))
        return [int(cand_ids[k]) for k in order[:m]]

    def _insert(self, i):
        level = self._random_level()
        for l in range(len(self.layers), level + 1):
            self.layers.append({})      # grow the ladder as tall nodes arrive
        q = self.X[i]
        if self.entry is None:          # the very first node is the whole graph
            for l in range(level + 1):
                self.layers[l][i] = []
            self.entry, self.top = i, level
            return
        # 1) greedy-descend the layers ABOVE this node's level to find an entry
        ep = [self.entry]
        for l in range(self.top, level, -1):
            ep = self._search_layer(q, ep, ef=1, l=l)
        # 2) on each layer from this node's level down to 0, connect it up
        for l in range(min(level, self.top), -1, -1):
            found = self._search_layer(q, ep, ef=self.efc, l=l)
            m = self.M0 if l == 0 else self.M
            nbrs = self._select_neighbours(q, np.asarray(found, dtype=int), m)
            self.layers[l][i] = list(nbrs)
            for nb in nbrs:             # links are bidirectional; prune the far side
                lst = self.layers[l].setdefault(nb, [])
                if i not in lst:
                    lst.append(i)
                if len(lst) > m:
                    self.layers[l][nb] = self._select_neighbours(
                        self.X[nb], np.asarray(lst, dtype=int), m)
            ep = found
        if level > self.top:            # a taller node becomes the new entry
            for l in range(self.top + 1, level + 1):
                self.layers[l][i] = []  # sole node this high: present, just no peers yet
            self.entry, self.top = i, level

    def search(self, q, k=10, ef=32):
        """Descend the sparse upper layers to a good entry, then beam layer 0."""
        q = np.asarray(q, dtype=np.float32)
        ep = [self.entry]
        for l in range(self.top, 0, -1):
            ep = self._search_layer(q, ep, ef=1, l=l)
        found = self._search_layer(q, ep, ef=max(ef, k), l=0)
        return np.asarray(found[:k], dtype=int)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also index the real gallery from the web export")
    args = ap.parse_args()

    X, Q = synthetic(n=3000, n_queries=100)
    exact = np.argsort(Q @ X.T, axis=1)[:, ::-1][:, :10]

    # build once; the graph is the offline cost, amortised over every query
    index = HNSW(X, M=16, ef_construction=48)
    lvls = [len(l) for l in index.layers]
    print(f"{len(X)} clustered vectors, {len(Q)} queries.")
    print(f"HNSW graph: {len(index.layers)} layers, sizes {lvls} "
          f"(most nodes live only on layer 0).\n")

    print("  ef    recall@10   dist/query")
    for ef in (8, 16, 32, 64):
        rec = 0
        index.dist_calls = 0
        for qi, q in enumerate(Q):
            found = index.search(q, ef=ef)
            rec += recall_at_k(found, exact[qi])
        print(f"  {ef:>3}   {rec / len(Q):>9.2f}   {index.dist_calls / len(Q):>9.1f}")

    # apples-to-apples vs IVF on the SAME vectors: recall at matched work.
    C, lists = ivf_build(X, n_lists=64)
    print("\nIVF on the same data, for comparison:")
    print("  probes recall@10   dist/query")
    for probes in (1, 2, 4, 8):
        rec = scanned = 0
        for qi, q in enumerate(Q):
            found, n = ivf_search(q, X, C, lists, probes=probes)
            rec += recall_at_k(found, exact[qi])
            scanned += n + len(C)          # probed vectors + the centroid scan
        print(f"  {probes:>4}   {rec / len(Q):>9.2f}   {scanned / len(Q):>9.1f}")

    print("\nread the two tables at MATCHED dist/query: the graph keeps more of the")
    print("truth per distance computed. IVF's work grows like √N, the graph's like")
    print("log N — the gap only widens toward a billion.")

    if args.json:
        items = db.load_json_gallery(args.json)
        I = np.asarray([it["image_emb"] for it in items], dtype=np.float32)
        idx = HNSW(I, M=8, ef_construction=32)
        exact14 = np.argsort(I @ I[0])[::-1][:5]
        got = idx.search(I[0], k=5, ef=16)
        print(f"\nthe real gallery ({len(items)} images): HNSW returns "
              f"{recall_at_k(got, exact14):.0%} of the exact top-5 for image 0 — but at "
              "fourteen, just scan everything. A graph earns its keep at thousands.")


if __name__ == "__main__":
    main()
