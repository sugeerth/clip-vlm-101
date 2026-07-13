// pq.js must agree with pq.py — same tables, same lookups, same top-k.
// Synthetic pack, deterministic PRNG, no network, no model.
// Run: node docs/js/test_pq.mjs
import { adcTables, pqSearch } from './pq.js';

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

const n = 2000, m = 8, ks = 16, sub = 4;
const books = Float32Array.from({ length: m * ks * sub }, () => rng() * 2 - 1);
const codes = Uint8Array.from({ length: n * m }, () => Math.floor(rng() * ks));
const pack = { books, codes, manifest: { n, m, ks, sub } };
const q = Float32Array.from({ length: m * sub }, () => rng() * 2 - 1);

// the reference: score = dot(query, reconstruction), computed the slow way
const refScore = i => {
  let s = 0;
  for (let j = 0; j < m; j++) {
    const o = (j * ks + codes[i * m + j]) * sub;
    for (let d = 0; d < sub; d++) s += books[o + d] * q[j * sub + d];
  }
  return s;
};

const T = adcTables(q, pack);
let tablesOk = true;
for (let j = 0; j < m; j++) for (let c = 0; c < ks; c++) {
  let s = 0;
  for (let d = 0; d < sub; d++) s += books[(j * ks + c) * sub + d] * q[j * sub + d];
  tablesOk &&= Math.abs(T[j * ks + c] - s) < 1e-5;
}
check(tablesOk, 'adcTables = per-subspace centroid · query dots');

const found = pqSearch(pack, T, 10);
check(found.length === 10, 'pqSearch returns k results');
check(found.every(r => Math.abs(r.score - refScore(r.id)) < 1e-4),
  'every returned score == dot with the reconstruction');
check(found.every((r, i) => i === 0 || found[i - 1].score >= r.score),
  'results come back sorted');

const all = Array.from({ length: n }, (_, i) => ({ id: i, score: refScore(i) }))
  .sort((a, b) => b.score - a.score).slice(0, 10);
check(JSON.stringify(all.map(r => r.id)) === JSON.stringify(found.map(r => r.id)),
  'top-10 ids == full argsort of reference scores');

if (failed) { console.error('some pq.js checks FAILED'); process.exit(1); }
console.log('all pq.js checks passed');
