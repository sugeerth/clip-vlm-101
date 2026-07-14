// Mirror of learn2rank.py — the ranker that LEARNS you, on your own device.
//
// dcn.js showed the ranking mechanism with hand-set weights; this LEARNS the
// weights live from your 👍/👎, with no server. Linear scorer s = w·x over the
// features [cos_image, cos_text, tag_overlap, rank_prior], trained by pairwise
// RankNet (Burges 2005): for every (👍 i, 👎 j) pair, push i above j.
//   o = w·(xi−xj);  λ = −σ/(1+exp(σo));  w ← w − lr·(λ·(xi−xj) + l2·w)
// Pairwise = robust with sparse feedback (no pos AND neg → no pair → no change).
// Safeguards: w starts [1,0,0,0] (untrained == base), L2 decay, and the learned
// score is BLENDED with the base and capped at 50%: final = (1−β)·base + β·learned,
// β = 0.5·n/(n+3). The model is four floats — it lives in localStorage as YOUR
// personal ranker that never leaves your machine. Constants match learn2rank.py.
const FEATURES = ['cos_image', 'cos_text', 'tag_overlap', 'rank_prior'];
const SCALE = [2.0, 2.0, 5.0, 1.0], OFFSET = [-0.5, -0.5, 0.0, 0.0];
const W_INIT = [1.0, 0.0, 0.0, 0.0];
const SIGMA = 1.0, LR = 0.1, L2 = 0.1, EPOCHS = 30;
const W_MAX = 0.5, K_MIX = 3.0, ROCCHIO_POS = 0.75, ROCCHIO_NEG = 0.15;

const scaleFeat = x => x.map((v, i) => v / SCALE[i] + OFFSET[i]);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const minmax = v => {
  const lo = Math.min(...v), hi = Math.max(...v);
  return hi > lo ? v.map(x => (x - lo) / (hi - lo)) : v.map(() => 0.5);
};

export class OnlineRanker {
  constructor() { this.w = [...W_INIT]; this.buffer = []; }

  toState() { return { w: this.w, buffer: this.buffer }; }
  loadState(s) {
    this.buffer = (s?.buffer || []).map(([x, y]) => [x, y | 0]);
    this._refit();
    return this;
  }
  get n() { return this.buffer.length; }
  nPairs() {
    const pos = this.buffer.filter(([, y]) => y === 1).length;
    return pos * (this.n - pos);
  }

  feedback(features, label) {
    this.buffer.push([scaleFeat(features), label | 0]);
    this._refit();
  }

  _refit() {
    this.w = [...W_INIT];
    const pos = this.buffer.filter(([, y]) => y === 1).map(([x]) => x);
    const neg = this.buffer.filter(([, y]) => y === 0).map(([x]) => x);
    if (pos.length && neg.length) {                       // RankNet over all pairs
      for (let e = 0; e < EPOCHS; e++)
        for (const xi of pos) for (const xj of neg) {
          const d = xi.map((v, i) => v - xj[i]);
          const o = dot(this.w, d);
          const lam = -SIGMA / (1 + Math.exp(SIGMA * o));
          this.w = this.w.map((wi, i) => wi - LR * (lam * d[i] + L2 * wi));
        }
    } else if (pos.length || neg.length) {                // one-sided → Rocchio nudge
      const mean = rows => rows[0].map((_, i) => rows.reduce((s, r) => s + r[i], 0) / rows.length);
      if (pos.length) { const m = mean(pos); this.w = this.w.map((wi, i) => wi + ROCCHIO_POS * m[i]); }
      if (neg.length) { const m = mean(neg); this.w = this.w.map((wi, i) => wi - ROCCHIO_NEG * m[i]); }
    }
  }

  learned(features) { return dot(this.w, scaleFeat(features)); }

  rank(candidates, { baseKey = 'base_score', featKey = 'features', k = null } = {}) {
    if (!candidates.length) return [];
    const beta = W_MAX * this.n / (this.n + K_MIX);
    const base = minmax(candidates.map(c => c[baseKey]));
    const learned = this.n ? minmax(candidates.map(c => this.learned(c[featKey]))) : base;
    const out = candidates.map((c, i) => ({ ...c, score: (1 - beta) * base[i] + beta * learned[i], beta }));
    out.sort((a, b) => b.score - a.score);
    return k ? out.slice(0, k) : out;
  }

  importance() {
    const total = this.w.reduce((s, wi) => s + Math.abs(wi), 0) || 1;
    return FEATURES.map((name, i) =>
      ({ name, weight: +this.w[i].toFixed(3), importance: +(Math.abs(this.w[i]) / total).toFixed(3) }));
  }
}
