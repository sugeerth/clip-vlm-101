"""conformal.py — a result set with a GUARANTEE, or an honest abstain.

pipeline: calibration scores ──► [conformal] ──► threshold τ with coverage ≥ 1−α

Every other file returns a top-k and hopes. This one makes a promise you can
check: given a confidence level (say 90%), it returns the SMALLEST set of
results such that the thing you're looking for is inside it **at least 90% of
the time** — distribution-free, no assumption about CLIP or the data, only that
queries are exchangeable. When it can't keep that promise on a query, it says
so (abstains) instead of guessing.

This is **split conformal prediction** (Vovk et al. 2005; Angelopoulos & Bates,
arXiv:2107.07511), and for retrieval it collapses to picking one cosine
threshold, honestly:

  SCORE      nonconformity of a (query, relevant) pair = 1 − cos(query, relevant).
             Big score = the true match sat far from the query.
  CALIBRATE  on n labeled pairs, sort the scores and take the rank-corrected
             quantile:  k = ⌈(n+1)(1−α)⌉;  q̂ = kth smallest score (∞ if k>n).
  PREDICT    for a new query, return every item with score ≤ q̂, i.e. every item
             with  cos ≥ 1 − q̂.  That set covers the truth with prob ≥ 1−α.

The set is ADAPTIVE for free: a clear winner gives a set of one; a pile of
near-ties gives a big set — set SIZE is the per-query confidence signal. Empty
set → nothing cleared the bar → abstain ("no confident match"). Truncating the
set to look tidy would BREAK the guarantee, so we never do.

The guarantee is MARGINAL (coverage ≥ 1−α averaged over queries, not per query)
and finite-sample: 1−α ≤ coverage ≤ 1−α + 1/(n+1). On the 14-image gallery n is
tiny, so coverage moves in steps of 1/(n+1) ≈ 7% — α = 0.20 (80%) lands exactly;
90% rounds up to ~93%. Honest about the granularity is the whole point.

    python3 conformal.py --json docs/db.json            # coverage table, model-free
    python3 conformal.py --json docs/db.json --alpha 0.2 # one confidence level
"""
import numpy as np


def calibrate(scores, alpha: float) -> float:
    """Split-conformal quantile q̂ from calibration nonconformity scores.

    k = ⌈(n+1)(1−α)⌉ (the finite-sample correction); q̂ = kth smallest score.
    If k > n the level is too strict for this n → q̂ = ∞ (set = everything).
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n else float(s[k - 1])


def cosines(query_emb, items, key="image_emb"):
    q = np.asarray(query_emb, dtype=np.float64)
    return np.array([float(np.asarray(it[key], dtype=np.float64) @ q) for it in items])


def loo_scores(items, key="image_emb", query_key=None):
    """Leave-one-out calibration: each item is a held-out query, its same-tag
    siblings are the truth, and the score is 1 − (best same-tag cosine). Self is
    always excluded — a novel query at test time has no copy of itself in the
    gallery, so leaning on the trivial self-match would inflate the threshold.

    `query_key` (default = `key`) sets which modality the query uses. Calibrate
    in the SAME modality you query in: image→image cosines (~0.5–1.0) and
    text→image cosines (~0.15–0.30) live in different bands thanks to CLIP's
    modality gap, so a live text search must calibrate with query_key='text_emb'
    over key='image_emb' — an image-side threshold would never fire on it."""
    qkey = query_key or key
    scores = []
    for i, q in enumerate(items):
        cos = cosines(q[qkey], items, key)
        rel = [j for j, it in enumerate(items)
               if j != i and set(it["tags"]) & set(q["tags"])]
        if rel:
            scores.append(1.0 - max(cos[j] for j in rel))
        # singleton-tag queries have no positive → skip (can't be scored)
    return np.array(scores)


def predict(query_emb, items, qhat, key="image_emb"):
    """The conformal set: every item with cos ≥ 1 − q̂ (score ≤ q̂), best first."""
    cos = cosines(query_emb, items, key)
    tau = 1.0 - qhat
    idx = [i for i in np.argsort(cos)[::-1] if cos[i] >= tau]
    return idx, tau


def jackknife_coverage(scores, alpha: float) -> float:
    """Honest empirical coverage: for each calibration point, calibrate on the
    OTHERS and check the held-out point is covered. Should be ≈ 1−α."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    hits = 0
    for i in range(n):
        others = np.delete(s, i)
        qhat_i = calibrate(others, alpha)
        hits += s[i] <= qhat_i
    return hits / n if n else 0.0


def report(items, alphas=(0.4, 0.3, 0.2, 0.1, 0.05), key="image_emb") -> list:
    """A coverage/size table across confidence levels — the proof it works."""
    scores = loo_scores(items, key)
    n = len(scores)
    rows = []
    for a in alphas:
        qhat = calibrate(scores, a)
        sizes = [len(predict(q[key], items, qhat, key)[0]) - 1  # minus the query itself
                 for q in items]
        rows.append({
            "target": round(1 - a, 3),
            "coverage": round(jackknife_coverage(scores, a), 3),
            "avg_set": round(float(np.mean(sizes)), 2),
            "tau": round(1 - qhat, 3) if np.isfinite(qhat) else None,
            "n": n,
        })
    return rows


if __name__ == "__main__":
    import argparse

    import db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--alpha", type=float, help="one confidence level 1−α to detail")
    args = ap.parse_args()

    items = db.load_json_gallery(args.json) if args.json else db.all_images(db.connect(args.db))

    if args.alpha is not None:
        scores = loo_scores(items)
        qhat = calibrate(scores, args.alpha)
        cov = jackknife_coverage(scores, args.alpha)
        print(f"target coverage 1−α = {1 - args.alpha:.0%}  (n={len(scores)} calibration queries)")
        print(f"  q̂ = {qhat:.3f}  →  return items with cos ≥ {1 - qhat:.3f}")
        print(f"  empirical leave-one-out coverage = {cov:.1%}  (guarantee: ≥ {1 - args.alpha:.0%})")
        raise SystemExit(0)

    print(f"conformal coverage on {len(items)} images "
          "(relevant = shares a tag; leave-one-out):\n")
    print(f"  {'target':>7} {'coverage':>9} {'avg set':>8} {'cos ≥':>7}")
    for r in report(items):
        tau = f"{r['tau']:.3f}" if r["tau"] is not None else "  all"
        print(f"  {r['target']:>7.0%} {r['coverage']:>9.1%} {r['avg_set']:>8.1f} {tau:>7}")
    print("\ncoverage sits on or above the target (conformal is valid-or-"
          "conservative);\nthe set grows as you demand more confidence — that "
          "tradeoff is the guarantee.")
