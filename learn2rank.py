"""learn2rank.py — the ranker that LEARNS you, on your own device.

pipeline: retrieved candidates + your 👍/👎 ──► [learn] ──► re-ordered, personal

dcn.py showed the ranking-stage MECHANISM with hand-set weights — "untrained,
demonstrates the cross." This file closes that loop: it LEARNS the weights,
live, from a handful of thumbs-up/down, with no server and no framework.

The model is a linear scorer `s = w · x` over the same per-candidate features
[cos_image, cos_text, tag_overlap, rank_prior], trained by **pairwise RankNet**
(Burges et al., ICML 2005) — the learning-to-rank standard. For every
(👍 i, 👎 j) pair, push i's score above j's:

    o   = w · (x_i − x_j)                 # current score gap
    λ   = −σ / (1 + exp(σ·o))             # how wrong the pair is (→0 once fixed)
    w  ← w − lr · ( λ·(x_i − x_j) + l2·w )   # SGD step + L2 weight decay

Pairwise is the robust choice for sparse feedback: it learns only RELATIVE
order, so a satisfied pair stops pulling (λ→0), and there is no absolute
target to blow up on. Three safeguards keep 2 clicks from wrecking the order:

  - w starts at [1,0,0,0] → the untrained model IS the base (cos_image) ranking.
  - L2 weight decay shrinks weights toward that prior.
  - the learned score is BLENDED with the base score, capped at 50 %:
        final = (1−β)·base + β·learned,   β = 0.5 · n/(n+3)
    (both min-max normalized to [0,1] first). Retrieval always keeps ≥ half
    the vote; feedback only tilts. With one-sided feedback (all 👍 or all 👎,
    no pairs) it falls back to a gentle Rocchio nudge toward/away from what
    you marked — so a click always does something visible, but never wild.

Features are scaled to comparable ranges, so the learned weights stay
INTERPRETABLE: `importance` = |w| / Σ|w| reads as "you weight tag-overlap ~2×
raw cosine" (relative, on standardized features — the honest phrasing). The
whole model is four floats — small enough to live in localStorage as YOUR
personal ranker that never leaves your machine.
"""
import numpy as np

FEATURES = ("cos_image", "cos_text", "tag_overlap", "rank_prior")
SCALE = np.array([2.0, 2.0, 5.0, 1.0])          # cos→~[0,1] via (x+1)/2, tags/5
OFFSET = np.array([-0.5, -0.5, 0.0, 0.0])       # applied as x/SCALE... see _scale
W_INIT = np.array([1.0, 0.0, 0.0, 0.0])         # untrained == base cos_image order
SIGMA, LR, L2, EPOCHS = 1.0, 0.1, 0.1, 30
W_MAX, K_MIX = 0.5, 3.0                          # blend cap and saturation
ROCCHIO_POS, ROCCHIO_NEG = 0.75, 0.15


def _scale(x):
    """Map raw features to comparable ranges (identical in learn.js)."""
    x = np.asarray(x, dtype=np.float64)
    return x / SCALE + OFFSET                   # cos→[0,1], tag→[0,1], rank as-is


def _minmax(v):
    v = np.asarray(v, dtype=np.float64)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.full_like(v, 0.5)


class OnlineRanker:
    """A personal re-ranker learned online from relevance feedback."""

    def __init__(self):
        self.w = W_INIT.copy()
        self.buffer = []            # [(x_scaled, label)] — the click buffer

    # ---- persistence: the "personal model" as a few floats -----------------
    def to_state(self):
        return {"w": self.w.tolist(), "buffer": [[x.tolist(), y] for x, y in self.buffer]}

    def load_state(self, s):
        self.buffer = [(np.asarray(x, dtype=np.float64), int(y)) for x, y in s.get("buffer", [])]
        self._refit()
        return self

    @property
    def n(self):
        return len(self.buffer)

    def n_pairs(self):
        pos = sum(1 for _, y in self.buffer if y == 1)
        return pos * (self.n - pos)

    # ---- learning ----------------------------------------------------------
    def feedback(self, features, label):
        """Record one 👍(1)/👎(0) and refit from the whole buffer (stable, tiny data)."""
        self.buffer.append((_scale(features), int(label)))
        self._refit()

    def _refit(self):
        self.w = W_INIT.copy()
        pos = [x for x, y in self.buffer if y == 1]
        neg = [x for x, y in self.buffer if y == 0]
        if pos and neg:                                   # RankNet over all pairs
            for _ in range(EPOCHS):
                for xi in pos:
                    for xj in neg:
                        o = self.w @ (xi - xj)
                        lam = -SIGMA / (1 + np.exp(SIGMA * o))
                        self.w -= LR * (lam * (xi - xj) + L2 * self.w)
        elif pos or neg:                                  # one-sided → Rocchio nudge
            if pos:
                self.w = self.w + ROCCHIO_POS * np.mean(pos, axis=0)
            if neg:
                self.w = self.w - ROCCHIO_NEG * np.mean(neg, axis=0)

    def learned(self, features):
        return float(self.w @ _scale(features))

    # ---- serving -----------------------------------------------------------
    def rank(self, candidates, base_key="base_score", feat_key="features", k=None):
        """Re-rank by (1−β)·base + β·learned, β = 0.5·n/(n+3), both normalized."""
        if not candidates:
            return []
        beta = W_MAX * self.n / (self.n + K_MIX)
        base = _minmax([c[base_key] for c in candidates])
        learned = _minmax([self.learned(c[feat_key]) for c in candidates]) if self.n else base
        out = [{**c, "score": float((1 - beta) * base[i] + beta * learned[i]), "beta": beta}
               for i, c in enumerate(candidates)]
        out.sort(key=lambda c: c["score"], reverse=True)
        return out[:k] if k else out

    def importance(self):
        """Signed weight + relative importance per feature (on scaled features)."""
        total = float(np.abs(self.w).sum()) or 1.0
        return {name: {"weight": round(float(wi), 3), "importance": round(abs(float(wi)) / total, 3)}
                for name, wi in zip(FEATURES, self.w)}


if __name__ == "__main__":
    import argparse

    import db
    import dcn

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="images/004_cat.jpg")
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    items = db.load_json_gallery(args.json)
    query = [it for it in items if args.image in it["path"]][0]
    cand = dcn.candidates_from_gallery(query, items)
    for c in cand:
        c["base_score"] = c["cos_image"]
        c["features"] = [c["cos_image"], c["cos_text"], c["tag_overlap"], c["rank_prior"]]
    qtags = set(query["tags"])

    r = OnlineRanker()
    print(f"query: {query['path']}  tags: {', '.join(query['tags'])}\n")
    print("before feedback (pure retrieval, β=0):")
    for c in r.rank(cand, k=args.k):
        print(f"  {c['score']:.3f}  {c['item']['path']}  (shared tags {c['tag_overlap']})")

    for c in r.rank(cand, k=args.k):                       # 👍 tag-sharers, 👎 the rest
        r.feedback(c["features"], 1 if set(c["item"]["tags"]) & qtags else 0)
    print(f"\nafter {r.n} thumbs ({r.n_pairs()} pairs) — β={r.rank(cand)[0]['beta']:.2f}:")
    for c in r.rank(cand, k=args.k):
        print(f"  {c['score']:.3f}  {c['item']['path']}  (shared tags {c['tag_overlap']})")
    print(f"\nlearned importance: "
          + ", ".join(f"{k} {v['importance']:.2f}" for k, v in r.importance().items()))
    print("your clicks taught the ranker — on your machine, no server.")
