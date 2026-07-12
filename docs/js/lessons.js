// Mirror of the five standalone Python lessons — the same math, live on the
// page's stored embeddings. No model, no DOM: pure functions, wired by app.js.
//   temperature.py    → softmax lives in rank.js (imported by the widget)
//   similarity.py     → modalityGap()
//   retrieval_eval.py → evalRetrieval()
//   arithmetic.py     → combine()
//   quantize.py       → quantizeInt8(), topNeighbors()
//   ann.py            → synthetic(), buildIVF(), searchIVF()
import { dot, scoreItem } from './rank.js';

// similarity.modality_gap: mean similarity within each tower and across them.
// The famous result (Liang et al. 2022): an image is ~2x MORE similar to
// other images than to the text embedding of its OWN caption.
export function modalityGap(items) {
  const n = items.length;
  let imgImg = 0, txtTxt = 0, own = 0, cross = 0, off = 0;
  for (let i = 0; i < n; i++) {
    own += dot(items[i].image_emb, items[i].text_emb);
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      imgImg += dot(items[i].image_emb, items[j].image_emb);
      txtTxt += dot(items[i].text_emb, items[j].text_emb);
      cross += dot(items[i].image_emb, items[j].text_emb);
      off++;
    }
  }
  return {
    'image · other images': imgImg / off,
    'text · other texts': txtTxt / off,
    'image · OWN caption': own / n,
    'image · other captions': cross / off,
  };
}

// retrieval_eval.evaluate: leave-one-out — each image queries the rest with
// its image embedding; a hit is RELEVANT if it shares >=1 tag with the query.
export function evalRetrieval(items, mode, ks = [1, 3, 5]) {
  const sums = Object.fromEntries(ks.map(k => [`P@${k}`, 0]));
  sums.MRR = 0;
  for (const q of items) {
    const rel = items.filter(it => it !== q)
      .map(it => ({ it, s: scoreItem(it, q.image_emb, mode) }))
      .sort((a, b) => b.s - a.s)
      .map(({ it }) => it.tags.some(t => q.tags.includes(t)));
    for (const k of ks) sums[`P@${k}`] += rel.slice(0, k).filter(Boolean).length / k;
    const first = rel.indexOf(true);
    sums.MRR += first === -1 ? 0 : 1 / (first + 1);
  }
  return Object.fromEntries(Object.entries(sums).map(([k, v]) => [k, v / items.length]));
}

// arithmetic.combine: Σ coeff·vector, renormalized back onto the unit sphere
// (the sum of unit vectors is NOT unit length — that's the one rule).
export function combine(vectors, coeffs) {
  const v = vectors[0].map((_, d) =>
    vectors.reduce((s, vec, i) => s + coeffs[i] * vec[d], 0));
  const n = Math.hypot(...v);
  if (n < 1e-9) return null; // the combination cancelled itself out
  return v.map(x => x / n);
}

// quantize.quantize: float32 → int8 with ONE shared scale (x ≈ q · scale).
export function quantizeInt8(vectors) {
  let m = 0;
  for (const v of vectors) for (const x of v) m = Math.max(m, Math.abs(x));
  const scale = (m / 127) || 1;
  return { q: vectors.map(v => v.map(x => Math.round(x / scale))), scale };
}

// quantize.top_neighbors: each row's k best columns, self excluded.
export function topNeighbors(vectors, k = 3) {
  return vectors.map((a, i) =>
    vectors.map((b, j) => ({ j, s: dot(a, b) }))
      .filter(({ j }) => j !== i)
      .sort((x, y) => y.s - x.s).slice(0, k).map(({ j }) => j));
}

// similarity.center: subtract the modality's mean direction, renormalize —
// the gap's simplest partial fix. On this gallery it widens the own-caption
// margin ~3x (test_lessons.mjs pins it).
export function centerRows(vectors) {
  const d = vectors[0].length;
  const mean = Array.from({ length: d }, (_, j) =>
    vectors.reduce((s, v) => s + v[j], 0) / vectors.length);
  return vectors.map(v => {
    const c = v.map((x, j) => x - mean[j]);
    const n = Math.hypot(...c) || 1;
    return c.map(x => x / n);
  });
}

// ---- ann.py mirrors: IVF on clustered synthetic vectors, in the browser --

// Deterministic RNG (mulberry32) + Box-Muller — Math.random has no seed,
// and the lesson's numbers must be the same on every visit.
export function rng32(seed) {
  let a = seed >>> 0;
  const uni = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(uni() || 1e-12)) * Math.cos(2 * Math.PI * uni());
}

const unitRow = v => { const n = Math.hypot(...v) || 1; return v.map(x => x / n); };

// Clustered unit vectors — the structure real embeddings actually have.
export function synthetic(n = 2000, dim = 32, blobs = 32, noise = 0.2, seed = 7) {
  const g = rng32(seed);
  const centers = Array.from({ length: blobs }, () =>
    unitRow(Array.from({ length: dim }, g)));
  const make = m => Array.from({ length: m }, () => {
    const c = centers[Math.floor(Math.abs(g()) * 7919) % blobs];
    return unitRow(c.map(x => x + noise * g()));
  });
  return { X: make(n), Q: make(50) };
}

// Spherical k-means: farthest-point init + Lloyd (ann.kmeans).
export function buildIVF(X, nLists = 32, iters = 4) {
  const cents = [X[0]];
  while (cents.length < nLists) {
    let best = 0, bestD = -Infinity;
    for (let i = 0; i < X.length; i++) {
      const d = 1 - Math.max(...cents.map(c => dot(X[i], c)));
      if (d > bestD) { bestD = d; best = i; }
    }
    cents.push(X[best]);
  }
  let C = cents, assign = [];
  for (let it = 0; it < iters; it++) {
    assign = X.map(v => C.reduce((b, c, j) => dot(v, c) > dot(v, C[b]) ? j : b, 0));
    C = C.map((c, j) => {
      const members = X.filter((_, i) => assign[i] === j);
      if (!members.length) return c;
      return unitRow(members[0].map((_, d2) =>
        members.reduce((s, m) => s + m[d2], 0) / members.length));
    });
  }
  const lists = Array.from({ length: nLists }, () => []);
  assign.forEach((j, i) => lists[j].push(i));
  return { C, lists };
}

// Scan only the `probes` nearest lists (ann.search).
export function searchIVF(q, X, C, lists, k = 10, probes = 4) {
  const near = C.map((c, j) => ({ j, s: dot(q, c) }))
    .sort((a, b) => b.s - a.s).slice(0, probes).map(({ j }) => j);
  const cand = near.flatMap(j => lists[j]);
  const found = cand.map(i => ({ i, s: dot(q, X[i]) }))
    .sort((a, b) => b.s - a.s).slice(0, k).map(({ i }) => i);
  return { found, scanned: cand.length };
}

export const recallAtK = (found, truth) =>
  found.filter(i => truth.includes(i)).length / truth.length;
