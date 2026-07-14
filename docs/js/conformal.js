// Mirror of conformal.py — a result set with a coverage GUARANTEE, or an
// honest abstain. Split conformal prediction (Vovk 2005; Angelopoulos & Bates
// arXiv:2107.07511): for retrieval it collapses to one calibrated cosine
// threshold.
//   score      1 − cos(query, relevant)             (nonconformity)
//   calibrate  k = ⌈(n+1)(1−α)⌉;  q̂ = kth smallest score  (∞ if k>n)
//   predict    return items with cos ≥ 1 − q̂  → covers the truth ≥ 1−α of the time
// The set is adaptive (clear winner → set of 1; near-ties → big set); an empty
// set means abstain. The guarantee is MARGINAL and finite-sample (1−α ≤ cov ≤
// 1−α+1/(n+1)); on ~14 items it steps by 1/(n+1), so 80% lands exactly and 90%
// rounds up to ~93%. The rank-form quantile ports 1:1 from conformal.py.
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

// q̂ from calibration nonconformity scores at level alpha.
export function calibrate(scores, alpha) {
  const s = [...scores].sort((a, b) => a - b);
  const n = s.length;
  const k = Math.ceil((n + 1) * (1 - alpha));
  return k > n ? Infinity : s[k - 1];
}

export const cosines = (queryEmb, items, key = 'image_emb') =>
  items.map(it => dot(it[key], queryEmb));

// Leave-one-out calibration: each item a held-out query, same-tag siblings the
// truth, score = 1 − best same-tag cosine. Self is always excluded (a novel
// query has no copy of itself in the gallery). queryKey (default = key) picks
// the query modality — calibrate in the one you query in: the modality gap puts
// image→image (~0.5–1.0) and text→image (~0.15–0.30) cosines in different bands,
// so a live text search calibrates with queryKey='text_emb' over key='image_emb'.
export function looScores(items, key = 'image_emb', queryKey = null) {
  const qk = queryKey || key;
  const scores = [];
  items.forEach((q, i) => {
    const cos = cosines(q[qk], items, key);
    const rel = items.map((it, j) =>
      (j !== i && it.tags.some(t => q.tags.includes(t))) ? cos[j] : -Infinity);
    const best = Math.max(...rel);
    if (best > -Infinity) scores.push(1 - best);
  });
  return scores;
}

// The conformal set: item indices with cos ≥ 1 − q̂, best first, plus τ.
export function predict(queryEmb, items, qhat, key = 'image_emb') {
  const cos = cosines(queryEmb, items, key);
  const tau = 1 - qhat;
  const idx = cos.map((c, i) => [c, i]).sort((a, b) => b[0] - a[0])
    .filter(([c]) => c >= tau).map(([, i]) => i);
  return { idx, tau };
}

// Honest empirical coverage: calibrate on the others, check the held-out point.
export function jackknifeCoverage(scores, alpha) {
  const n = scores.length;
  if (!n) return 0;
  let hits = 0;
  for (let i = 0; i < n; i++) {
    const others = scores.filter((_, j) => j !== i);
    if (scores[i] <= calibrate(others, alpha)) hits++;
  }
  return hits / n;
}

export function report(items, alphas = [0.4, 0.3, 0.2, 0.1, 0.05], key = 'image_emb') {
  const scores = looScores(items, key);
  return alphas.map(a => {
    const qhat = calibrate(scores, a);
    const sizes = items.map(q => predict(q[key], items, qhat, key).idx.length - 1);
    return {
      target: +(1 - a).toFixed(3),
      coverage: +jackknifeCoverage(scores, a).toFixed(3),
      avgSet: +(sizes.reduce((s, x) => s + x, 0) / sizes.length).toFixed(2),
      tau: isFinite(qhat) ? +(1 - qhat).toFixed(3) : null,
      n: scores.length,
    };
  });
}
