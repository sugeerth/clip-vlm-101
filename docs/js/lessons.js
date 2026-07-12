// Mirror of the five standalone Python lessons — the same math, live on the
// page's stored embeddings. No model, no DOM: pure functions, wired by app.js.
//   temperature.py    → softmax lives in rank.js (imported by the widget)
//   similarity.py     → modalityGap()
//   retrieval_eval.py → evalRetrieval()
//   arithmetic.py     → combine()
//   quantize.py       → quantizeInt8(), topNeighbors()
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
