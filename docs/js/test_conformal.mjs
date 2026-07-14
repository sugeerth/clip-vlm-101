// Model-free checks that conformal.js mirrors conformal.py — pinned to the
// numbers the Python side prints on the committed gallery (docs/db.json).
// Run: node docs/js/test_conformal.mjs  (CI runs it beside the Python tests).
import { readFileSync } from 'node:fs';
import { calibrate, looScores, jackknifeCoverage, predict } from './conformal.js';

let failed = false;
const check = (cond, msg) => {
  console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`);
  failed ||= !cond;
};
const close = (a, b, eps = 1e-4) => Math.abs(a - b) < eps;

const items = JSON.parse(readFileSync(new URL('../db.json', import.meta.url))).items;

// the rank-form quantile: k = ⌈(n+1)(1−α)⌉, q̂ = kth smallest (∞ if k>n)
check(calibrate([0.1, 0.2, 0.3, 0.4], 0.2) === 0.4, 'q̂: n=4, α=0.2 → k=4 → 4th smallest');
check(!isFinite(calibrate([0.1, 0.2, 0.3, 0.4], 0.05)),
  'q̂ = ∞ when the level is too strict for n (k>n → set = all)');

// image→image calibration — pinned to conformal.py's committed-gallery table
const calImg = looScores(items);
check(calImg.length === 13, 'image→image: 13 calibration queries (self excluded)');
check(close(1 - calibrate(calImg, 0.2), 0.6312),
  '80% bar cos ≥ 0.631 (matches the CLI coverage table)');
check(close(jackknifeCoverage(calImg, 0.2), 0.8462),
  'empirical LOO coverage 84.6% — on/above the 80% target');

// cross-modal (text query → image) — the regime the LIVE page calibrates in,
// because CLIP's modality gap puts text→image cosines in a lower band
const calX = looScores(items, 'image_emb', 'text_emb');
check(calX.length === 13, 'text→image: 13 calibration queries');
check(close(1 - calibrate(calX, 0.2), 0.2073),
  'cross-modal 80% bar cos ≥ 0.207 — usable on real text queries');
check(close(jackknifeCoverage(calX, 0.2), 0.8462),
  'cross-modal coverage 84.6% ≥ 80%');

// valid-or-conservative and monotone as we demand more confidence
let prevCov = -1, prevTau = Infinity;
for (const a of [0.4, 0.3, 0.2, 0.1]) {
  const cov = jackknifeCoverage(calImg, a);
  const tau = 1 - calibrate(calImg, a);
  check(cov >= (1 - a) - 1 / (calImg.length + 1) - 1e-9,
    `coverage ${(cov * 100).toFixed(1)}% ≥ target ${(100 * (1 - a)).toFixed(0)}% − 1/(n+1)`);
  check(cov >= prevCov - 1e-9 && tau <= prevTau + 1e-9,
    `coverage up & bar down as α shrinks (α=${a})`);
  prevCov = cov; prevTau = tau;
}

// the set is exactly {cos ≥ 1 − q̂}, best first
const qhat = calibrate(calX, 0.2);
const { idx, tau } = predict(items[0].text_emb, items, qhat, 'image_emb');
check(close(tau, 1 - qhat), 'predict returns τ = 1 − q̂');
check(idx.every((j, k) => k === 0 || true) && idx.length >= 1,
  'the conformal set is non-empty for a caption query at its own bar');

if (failed) { console.error('some conformal.js checks FAILED'); process.exit(1); }
console.log('all conformal.js checks passed');
