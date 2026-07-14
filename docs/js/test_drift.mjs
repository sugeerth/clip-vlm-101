// Model-free checks that drift.js mirrors drift.py — the three detectors and the
// stream verdicts, pinned to the Python numbers on the committed gallery.
// Run: node docs/js/test_drift.mjs   (CI runs it too).
import { readFileSync } from 'node:fs';
import { psi, ksStat, ksCritical, coverage, monitor, classify,
  qualitySignal, itemQuality, driftWindow, PSI_SHIFT, PSI_DRIFT, COV_SLACK } from './drift.js';

let failed = false;
const check = (cond, msg) => { console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`); failed ||= !cond; };
const close = (a, b, eps = 1e-4) => Math.abs(a - b) < eps;

check(PSI_SHIFT === 0.10 && PSI_DRIFT === 0.25 && COV_SLACK === 0.10, 'constants match drift.py');

// PSI/KS basics
check(psi([1, 2, 3, 4], [1, 2, 3, 4]) === 0, 'PSI of a window against itself = 0');
check(psi([1, 1, 1], [2, 2, 2]) === 0, 'PSI of a constant reference = 0 (no bins)');
check(ksStat([1, 2, 3, 4], [1, 2, 3, 4]) === 0, 'KS of identical samples = 0');
check(close(ksStat([0, 0, 0, 0], [1, 1, 1, 1]), 1.0), 'KS of fully-separated samples = 1');
check(close(ksCritical(60, 60, 0.05), 1.36 * Math.sqrt(120 / 3600)), 'KS critical value formula');

// coverage: reference calibrates the bar, live is measured against it
const covSelf = coverage([0.6, 0.7, 0.8, 0.9, 0.5], [0.6, 0.7, 0.8, 0.9, 0.5], 0.2);
check(covSelf.cov >= 0.79, 'coverage of the reference against its own bar ≈ target');

// the gallery stream — pinned to drift.py
const items = JSON.parse(readFileSync(new URL('../db.json', import.meta.url))).items;
const ref = qualitySignal(items);
check(ref.length === 60, 'quality signal: 60 same-tag pair similarities');

const t0 = monitor(ref, driftWindow(ref, 0.0), 0.2);
check(t0.level === 'stable' && close(t0.psi, 0) && close(t0.coverage, 0.8333, 1e-3),
  't0 (0% off): stable, PSI 0, coverage 83%');
const t1 = monitor(ref, driftWindow(ref, 0.15), 0.2);
check(t1.level === 'shift' && close(t1.psi, 0.1327, 1e-3) && close(t1.ks, 0.1333, 1e-3),
  't1 (15% off): SHIFT — PSI 0.13, KS 0.13 (matches drift.py)');
const t2 = monitor(ref, driftWindow(ref, 0.35), 0.2);
check(t2.level === 'drift' && close(t2.psi, 0.5666, 1e-3) && close(t2.coverage, 0.5667, 1e-3),
  't2 (35% off): DRIFT — PSI 0.57, coverage 57%');
const t3 = monitor(ref, driftWindow(ref, 0.60), 0.2);
check(t3.level === 'drift' && t3.reasons.length === 3 && close(t3.psi, 1.0722, 1e-3),
  't3 (60% off): DRIFT on all three detectors — PSI 1.07');

// drift_window's k uses explicit half-up rounding (not banker's) so it matches
// drift.py at an exact half: frac*n = 2.5 → k = 3, index 0 contaminated first
check(driftWindow([1, 1, 1, 1, 1], 0.5).filter(x => x < 1).length === 3,
  'driftWindow half-boundary: frac*n=2.5 → k=3 (half-up, matches drift.py)');
check(driftWindow([0.5], 0.5)[0] === 0.3, 'driftWindow([0.5],0.5) → [0.3] (k=1)');

// PSI is monotone in the contamination fraction
const psis = [0, 0.15, 0.35, 0.6].map(f => psi(ref, driftWindow(ref, f)));
check(psis.every((p, i) => i === 0 || p >= psis[i - 1] - 1e-12), 'PSI rises monotonically with contamination');

// positive vs failure cases against the calibrated bar
const { bar } = coverage(ref, driftWindow(ref, 0.6), 0.2);
const { positive, failure } = classify(items, itemQuality(items), bar);
check(positive.length === 13 && failure.length === 1, '13 positive · 1 failure case (matches drift.py)');
check(failure[0].item.tags.includes('bicycle'), 'the failure is the bicycle (no same-tag sibling)');

if (failed) { console.error('some drift.js checks FAILED'); process.exit(1); }
console.log('all drift.js checks passed');
