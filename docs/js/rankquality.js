// Mirror of rankquality.py — did the answer land in the top 1-3? measure it.
//
// A ranking FAILED if the user has to scroll. Position bias is brutal, so a good
// ranker puts the right image first/second/third. Score it from two sides: the
// RESULTS (precision@k, MRR, nDCG@k) and the BEHAVIOR (how far the user scrolled
// before engaging → a penalty). One 0-1 "quality" + a blunt top-3-or-scrolled
// verdict. Byte-identical to rankquality.py; drives the live gauge on the search page.
export const FREE = 3;      // engaging within the top-3 is free — that's the target
export const NDCG_K = 5;

// MRR's per-query term: 1 / (rank of the first relevant result), else 0.
export function reciprocalRank(relevant) {
  for (let i = 0; i < relevant.length; i++) if (relevant[i]) return 1 / (i + 1);
  return 0;
}

// fraction of the top-k that are relevant (fixed denominator k)
export function precisionAtK(relevant, k = FREE) {
  if (k <= 0) return 0;
  let hit = 0;
  for (let i = 0; i < Math.min(k, relevant.length); i++) if (relevant[i]) hit++;
  return hit / k;
}

// discounted cumulative gain: each gain / log2(rank+1)
export function dcg(gains) {
  return gains.reduce((s, g, i) => s + g / Math.log2(i + 2), 0);
}

// DCG@k over the ideal (sorted) DCG@k — 1.0 iff already best-ordered
export function ndcgAtK(gains, k = NDCG_K) {
  const ideal = dcg([...gains].sort((a, b) => b - a).slice(0, k));
  return ideal > 0 ? dcg(gains.slice(0, k)) / ideal : 0;
}

// 0 if the user engaged within the top `free`; grows toward 1 the deeper they had
// to scroll. No engagement is treated as scrolling to the bottom.
export function scrollPenalty(engagedRank, n, free = FREE) {
  if (n <= free) return 0;
  const rank = engagedRank == null ? n : engagedRank;
  return Math.min(1, Math.max(0, rank - free) / (n - free));
}

// the whole verdict. gains: per-result relevance in [0,1] (rank order); relevant:
// optional bool per result (defaults to gain >= 0.5). Returns the components, one
// 0-1 quality (front-loading, halved at most by the scroll penalty), and a verdict.
export function rankingQuality(gains, relevant = null, engagedRank = null, k = NDCG_K, free = FREE) {
  if (relevant == null) relevant = gains.map(g => g >= 0.5);
  const front = ndcgAtK(gains, k);
  const penalty = scrollPenalty(engagedRank, gains.length, free);
  const quality = front * (1 - 0.5 * penalty);
  const hitTop = (engagedRank == null || engagedRank <= free) && relevant.slice(0, free).some(Boolean);
  return {
    precision_at_3: precisionAtK(relevant, free),
    mrr: reciprocalRank(relevant),
    ndcg: front,
    scroll_penalty: penalty,
    quality,
    verdict: hitTop ? 'top-3' : 'scrolled',
  };
}
