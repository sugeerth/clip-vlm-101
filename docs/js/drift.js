// Mirror of drift.py — is the live stream still the world we calibrated for?
//
// Every guarantee here assumes the future looks like the past. This watches a
// stream of results and rules stable / shift / DRIFT with three distribution-
// free detectors on a monitored quality signal, plus the repo-native one:
//   PSI   Σ (l−r)·ln(l/r) over reference-quantile bins (<0.10 stable · >0.25 drift)
//   KS    the largest gap between the two CDFs — assumes nothing
//   COVERAGE  calibrate a conformal bar on the reference, measure it on the live
//             window; coverage falling below target IS exchangeability breaking.
// Then it sorts the window into POSITIVE (cleared the bar) and FAILURE cases.
// Constants and math match drift.py byte for byte.
import { calibrate, cosines } from './conformal.js';

export const PSI_SHIFT = 0.10, PSI_DRIFT = 0.25;
export const KS_ALPHA = 0.05;
export const COV_SLACK = 0.10;

const asc = a => [...a].sort((x, y) => x - y);
// numpy.quantile, linear interpolation
function quantile(sorted, q) {
  const n = sorted.length;
  if (n === 1) return sorted[0];
  const pos = q * (n - 1), lo = Math.floor(pos), hi = Math.ceil(pos), frac = pos - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}
// counts per bin with numpy.histogram's edge convention (searchsorted right − 1)
function histCounts(data, edges) {
  const counts = new Array(edges.length - 1).fill(0);
  for (const v of data) {
    let b = 0;                                // # edges <= v, via 'right'
    while (b < edges.length && edges[b] <= v) b++;
    b -= 1;
    if (b < 0) b = 0;
    if (b >= counts.length) b = counts.length - 1;
    counts[b]++;
  }
  return counts;
}
const countLE = (sorted, v) => {             // searchsorted(sorted, v, 'right')
  let lo = 0, hi = sorted.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] <= v) lo = mid + 1; else hi = mid; }
  return lo;
};

export function psi(ref, live, bins = 8) {
  const rs = asc(ref);
  const probs = Array.from({ length: bins + 1 }, (_, i) => i / bins);
  let edges = probs.map(q => quantile(rs, q));
  edges = edges.filter((e, i) => i === 0 || e !== edges[i - 1]);   // np.unique (already sorted)
  if (edges.length < 2) return 0.0;
  edges[0] = -Infinity; edges[edges.length - 1] = Infinity;
  const r = histCounts(ref, edges).map(c => c / ref.length);
  const l = histCounts(live, edges).map(c => c / live.length);
  const eps = 1e-6;
  let s = 0;
  for (let i = 0; i < r.length; i++) {
    const ri = Math.max(r[i], eps), li = Math.max(l[i], eps);
    s += (li - ri) * Math.log(li / ri);
  }
  return s;
}

export function ksStat(ref, live) {
  const rs = asc(ref), ls = asc(live);
  const grid = [...rs, ...ls];
  let m = 0;
  for (const g of grid) m = Math.max(m, Math.abs(countLE(rs, g) / rs.length - countLE(ls, g) / ls.length));
  return m;
}

export function ksCritical(n, m, alpha = KS_ALPHA) {
  const c = { 0.10: 1.22, 0.05: 1.36, 0.01: 1.63 }[alpha] ?? 1.36;
  return c * Math.sqrt((n + m) / (n * m));
}

export function coverage(refScores, liveScores, alpha, higherIsBetter = true) {
  const ref = refScores.map(s => (higherIsBetter ? 1 - s : s));
  const qhat = calibrate(ref, alpha);
  if (!isFinite(qhat)) return { cov: 1.0, bar: higherIsBetter ? -Infinity : Infinity };
  const bar = higherIsBetter ? 1 - qhat : qhat;
  const covered = liveScores.filter(s => (higherIsBetter ? s >= bar : s <= bar)).length;
  return { cov: covered / liveScores.length, bar };
}

export function monitor(ref, live, alpha = 0.2) {
  const p = psi(ref, live);
  const k = ksStat(ref, live);
  const kCrit = ksCritical(ref.length, live.length);
  const { cov, bar } = coverage(ref, live, alpha);
  const target = 1 - alpha;
  const reasons = [];
  if (p > PSI_DRIFT) reasons.push(`PSI ${p.toFixed(2)} > ${PSI_DRIFT} (population shifted)`);
  if (k > kCrit) reasons.push(`KS ${k.toFixed(2)} > ${kCrit.toFixed(2)} (distributions differ)`);
  if (cov < target - COV_SLACK) reasons.push(`coverage ${Math.round(cov * 100)}% < target ${Math.round(target * 100)}% (exchangeability broke)`);
  const level = reasons.length ? 'drift' : (p > PSI_SHIFT || k > kCrit * 0.75 ? 'shift' : 'stable');
  return { level, reasons, psi: p, ks: k, ksCritical: kCrit, coverage: cov, target,
    bar, failureRate: 1 - cov, nRef: ref.length, nLive: live.length };
}

export function classify(items, scores, bar) {
  const positive = [], failure = [];
  items.forEach((it, i) => (scores[i] >= bar ? positive : failure).push({ item: it, score: scores[i] }));
  return { positive, failure };
}

// the retrieval-quality population: every same-tag pair's similarity.
export function qualitySignal(items, key = 'image_emb') {
  const sig = [];
  items.forEach((q, i) => {
    const cos = cosines(q[key], items, key);
    items.forEach((it, j) => {
      if (j !== i && it.tags.some(t => q.tags.includes(t))) sig.push(cos[j]);
    });
  });
  return sig;
}

// per-item health: each image's best same-tag match (for the case list).
export function itemQuality(items, key = 'image_emb') {
  return items.map((q, i) => {
    const cos = cosines(q[key], items, key); cos[i] = -Infinity;
    const rel = items.map((it, j) =>
      (j !== i && it.tags.some(t => q.tags.includes(t))) ? cos[j] : -Infinity);
    const best = Math.max(...rel);
    return best > -Infinity ? best : 0.0;
  });
}

// deterministically CONTAMINATE a window: a `frac` fraction go off-distribution.
export function driftWindow(sig, frac, drop = 0.2) {
  const n = sig.length, k = Math.floor(frac * n + 0.5);   // explicit half-up, matches drift.py
  const live = [...sig];
  for (let i = 0; i < k; i++) { const idx = Math.floor((i * n) / k); live[idx] = Math.max(0, live[idx] - drop); }
  return live;
}
