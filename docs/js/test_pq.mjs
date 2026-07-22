// pq.js must agree with pq.py / opq.py — same rotation, same tables, same
// lookups, same two-stage re-rank. Synthetic pack, deterministic PRNG, no
// network, no model. Run: node docs/js/test_pq.mjs
import { adcTables, pqSearch, rerank, search } from './pq.js';

let failed = false;
const check = (cond, msg) => {
  console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`);
  failed ||= !cond;
};

// mulberry32 — the repo's stand-in for a seeded rng in tests
const rng = (seed => () => {
  seed |= 0; seed = seed + 0x6D2B79F5 | 0;
  let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
})(42);

const n = 2000, m = 8, ks = 16, sub = 4, d = m * sub;

// a real orthonormal rotation (Gram–Schmidt on random rows) — the thing OPQ
// learns and ships, applied query-side by pq.js.
function randOrtho(d) {
  const rows = [];
  for (let i = 0; i < d; i++) {
    const v = Float32Array.from({ length: d }, () => rng() * 2 - 1);
    for (const u of rows) {
      let dot = 0; for (let k = 0; k < d; k++) dot += v[k] * u[k];
      for (let k = 0; k < d; k++) v[k] -= dot * u[k];
    }
    let nrm = 0; for (let k = 0; k < d; k++) nrm += v[k] * v[k]; nrm = Math.sqrt(nrm);
    for (let k = 0; k < d; k++) v[k] /= nrm;
    rows.push(v);
  }
  const flat = new Float32Array(d * d);
  for (let i = 0; i < d; i++) for (let k = 0; k < d; k++) flat[i * d + k] = rows[i][k];
  return flat;
}

const rotation = randOrtho(d);
const books = Float32Array.from({ length: m * ks * sub }, () => rng() * 2 - 1);
const codes = Uint8Array.from({ length: n * m }, () => Math.floor(rng() * ks));
const pack = { books, codes, rotation, d, refine: null,
               manifest: { n, m, ks, sub, d, rerank_cand: 200 } };
const q = Float32Array.from({ length: d }, () => rng() * 2 - 1);

// reference rotation: qr = R · q
const qr = new Float32Array(d);
for (let i = 0; i < d; i++) { let s = 0; for (let k = 0; k < d; k++) s += rotation[i * d + k] * q[k]; qr[i] = s; }

// reference coarse score = dot(rotated query, reconstruction of the codes)
const refScore = i => {
  let s = 0;
  for (let j = 0; j < m; j++) {
    const o = (j * ks + codes[i * m + j]) * sub;
    for (let dd = 0; dd < sub; dd++) s += books[o + dd] * qr[j * sub + dd];
  }
  return s;
};

const T = adcTables(q, pack);         // must rotate internally, then build tables
let tablesOk = true;
for (let j = 0; j < m; j++) for (let c = 0; c < ks; c++) {
  let s = 0;
  for (let dd = 0; dd < sub; dd++) s += books[(j * ks + c) * sub + dd] * qr[j * sub + dd];
  tablesOk &&= Math.abs(T[j * ks + c] - s) < 1e-4;
}
check(tablesOk, 'adcTables rotates the query, then per-subspace centroid dots');

const found = pqSearch(pack, T, 10);
check(found.length === 10, 'pqSearch returns k results');
check(found.every(r => Math.abs(r.score - refScore(r.id)) < 1e-3),
  'every coarse score == dot with the rotated reconstruction');
check(found.every((r, i) => i === 0 || found[i - 1].score >= r.score),
  'coarse results come back sorted');

const allCoarse = Array.from({ length: n }, (_, i) => ({ id: i, score: refScore(i) }))
  .sort((a, b) => b.score - a.score).slice(0, 10);
check(JSON.stringify(allCoarse.map(r => r.id)) === JSON.stringify(found.map(r => r.id)),
  'coarse top-10 ids == full argsort of reference scores');

// ---- the int8 refine tier: two-stage re-rank ----
const i8 = Int8Array.from({ length: n * d }, () => Math.round(rng() * 254 - 127));
const i8scale = Float32Array.from({ length: d }, () => rng() * 0.02 + 0.001);
pack.refine = { i8, scale: i8scale };

// reference: score = i8_row · (scale ⊙ q)
const qs = new Float32Array(d);
for (let k = 0; k < d; k++) qs[k] = i8scale[k] * q[k];
const refRefine = id => { let s = 0; for (let k = 0; k < d; k++) s += i8[id * d + k] * qs[k]; return s; };

const cand = pqSearch(pack, T, 50);
const rr = rerank(pack, cand, q, 10);
check(rr.length === 10, 'rerank returns k results');
check(rr.every(r => Math.abs(r.score - refRefine(r.id)) < 1e-3),
  'every refined score == int8 · (scale ⊙ query)');
check(rr.every((r, i) => i === 0 || rr[i - 1].score >= r.score), 'refined results come back sorted');
const refTop = cand.map(c => ({ id: c.id, score: refRefine(c.id) }))
  .sort((a, b) => b.score - a.score).slice(0, 10);
check(JSON.stringify(refTop.map(r => r.id)) === JSON.stringify(rr.map(r => r.id)),
  'refined top-10 == exact re-rank of the coarse candidates');

// ---- search(): coarse without refine, two-stage with ----
pack.refine = null;
const coarseOnly = search(pack, q, 10);
check(JSON.stringify(coarseOnly.map(r => r.id)) === JSON.stringify(found.map(r => r.id)),
  'search() with no refine tier == the coarse scan');
pack.refine = { i8, scale: i8scale };
const twoStage = search(pack, q, 10, 50);
check(twoStage.length === 10 && twoStage.every((r, i) => i === 0 || twoStage[i - 1].score >= r.score),
  'search() with the refine tier runs the two-stage and stays sorted');

if (failed) { console.error('some pq.js checks FAILED'); process.exit(1); }
console.log('all pq.js checks passed');
