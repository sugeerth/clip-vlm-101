"""rankquality.py — did the answer land in the top 1-3? measure it, don't hope.

pipeline: (relevance scores, user scroll) ──► [P@3 · MRR · nDCG · scroll penalty] ──► one 0-1 score

A ranking FAILED if the user has to scroll. Position bias is brutal — eye-tracking
and click logs agree almost nobody looks past the first few results — so a good
ranker puts the right image first, second, or third, and the honest way to know is
to MEASURE it, from two sides:

  RESULTS   are the relevant ones at the top? — precision@k, MRR (1/rank of the
            first relevant), nDCG@k (graded relevance, discounted by position).
  BEHAVIOR  how far did the user scroll before they engaged? Engaging within the
            top few is free; every result they had to scroll past is a penalty.

Combine the front-loading (nDCG) with the scroll penalty into one 0-1 "ranking
quality", plus a blunt verdict: did the top-3 answer it, or did they have to
scroll? Model-free and deterministic; js/rankquality.js is a byte-identical twin
that drives the live gauge on the search page.

    python3 rankquality.py            # the metrics on a few worked examples
"""
import math

FREE = 3          # engaging within the top-3 costs nothing — that's the target
NDCG_K = 5        # front-loading is judged over the first screenful


def reciprocal_rank(relevant):
    """MRR's per-query term: 1 / (rank of the first relevant result), else 0."""
    for i, r in enumerate(relevant, 1):
        if r:
            return 1.0 / i
    return 0.0


def precision_at_k(relevant, k=FREE):
    """Fraction of the top-k that are relevant (fixed denominator k)."""
    if k <= 0:
        return 0.0
    return sum(1 for r in relevant[:k] if r) / k


def dcg(gains):
    """Discounted cumulative gain: each gain divided by log2(rank+1)."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(gains, k=NDCG_K):
    """DCG@k over the ideal (sorted) DCG@k — 1.0 iff already best-ordered."""
    ideal = dcg(sorted(gains, reverse=True)[:k])
    return dcg(gains[:k]) / ideal if ideal > 0 else 0.0


def scroll_penalty(engaged_rank, n, free=FREE):
    """0 if the user engaged within the top `free`; grows toward 1 the deeper they
    had to scroll. No engagement at all is treated as scrolling to the bottom."""
    if n <= free:
        return 0.0
    rank = n if engaged_rank is None else engaged_rank
    return min(1.0, max(0, rank - free) / (n - free))


def ranking_quality(gains, relevant=None, engaged_rank=None, k=NDCG_K, free=FREE):
    """The whole verdict. `gains` are per-result relevance in [0,1] (rank order);
    `relevant` is an optional bool per result (defaults to gain >= 0.5). Returns
    the component metrics, one 0-1 `quality` (front-loading, halved at most by the
    scroll penalty), and a blunt top-3-or-scrolled verdict."""
    if relevant is None:
        relevant = [g >= 0.5 for g in gains]
    front = ndcg_at_k(gains, k)
    penalty = scroll_penalty(engaged_rank, len(gains), free)
    quality = front * (1.0 - 0.5 * penalty)     # scroll can halve, never zero, the score
    hit_top = (engaged_rank is None or engaged_rank <= free) and any(relevant[:free])
    return {
        "precision_at_3": precision_at_k(relevant, free),
        "mrr": reciprocal_rank(relevant),
        "ndcg": front,
        "scroll_penalty": penalty,
        "quality": quality,
        "verdict": "top-3" if hit_top else "scrolled",
    }


def _fmt(label, r):
    print(f"  {label:<28} P@3={r['precision_at_3']:.2f}  MRR={r['mrr']:.2f}  "
          f"nDCG={r['ndcg']:.2f}  scroll={r['scroll_penalty']:.2f}  "
          f"→ quality {r['quality']:.2f} ({r['verdict']})")


def main():
    print("ranking quality — the answer belongs in the top 1-3:\n")
    # a perfect ranking: the best result is first, user clicks it
    _fmt("perfect (click #1)",
         ranking_quality([1.0, 0.6, 0.5, 0.2, 0.1], engaged_rank=1))
    # relevant result buried at #5; user had to scroll to it
    _fmt("buried (scrolled to #5)",
         ranking_quality([0.2, 0.2, 0.3, 0.4, 1.0], engaged_rank=5))
    # decent front-loading, user engaged in the top-3
    _fmt("good enough (click #2)",
         ranking_quality([0.9, 0.8, 0.4, 0.3, 0.2], engaged_rank=2))
    # nothing relevant near the top, user scrolled and gave up
    _fmt("miss (scrolled, no click)",
         ranking_quality([0.1, 0.2, 0.1, 0.3, 0.2], engaged_rank=None))
    print("\nfront-loading (nDCG) is the ranker's job; the scroll penalty is the "
          "user's verdict.\nGood ranking = the right image up top, no scrolling required.")


if __name__ == "__main__":
    main()
