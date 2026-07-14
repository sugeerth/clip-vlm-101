// Model-free checks that hermes.js mirrors hermes.py's gates — pinned to the
// numbers the Python `search()` prints on the SAME synthetic gallery + encoder.
// Run: node docs/js/test_hermes.mjs   (CI runs it beside the Python tests).
//
// Hermes has one gate here: the MARGIN critique. If the best phrasing separates
// its top hit from the pack by ≥ MIN_MARGIN it is "decisive" and published as-is;
// otherwise no phrasing is trusted and all four are ensembled. This test drives
// both sides of that gate, and checks the returned `qvec` is exactly the vector
// that produced the shown ranking (so the "🪽 hermes chose …" trace can't lie).
import { hermesSearch, margin, MIN_MARGIN, QUERY_TEMPLATES } from './hermes.js';
import { rank } from './rank.js';

let failed = false;
const check = (cond, msg) => {
  console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`);
  failed ||= !cond;
};
const close = (a, b, eps = 1e-5) => Math.abs(a - b) < eps;
const norm = v => { const n = Math.hypot(...v) || 1; return v.map(x => x / n); };
const mk = (path, v) => ({ path, image_emb: norm(v), text_emb: norm(v), tags: [path] });
// a fake encoder: map each filled phrasing to its query vector, then unit-scale
const encoder = table => async phr => phr.map(p => norm(table[p]));

check(MIN_MARGIN === 0.03, 'MIN_MARGIN = 0.03 (matches hermes.py)');

// margin() edge cases — must mirror the Python helper exactly
check(margin([{ score: 0.9 }, { score: 0.4 }, { score: 0.2 }]) === 0.9 - (0.4 + 0.2) / 2,
  'margin = top1 − mean(rest)');
check(margin([{ score: 0.7 }]) === 0.7, 'margin of a single hit = its score');
check(margin([]) === 0, 'margin of nothing = 0');

// CASE A — a DECISIVE phrasing: "a photo of cat" points sharply at item a.
// Pinned to hermes.py: margins, chose, satisfied, order, and qvec all match.
const itemsA = [mk('a', [1, 0, 0, 0]), mk('b', [0, 1, 0, 0]), mk('c', [0, 0, 1, 0]), mk('d', [0.7, 0.7, 0, 0])];
const encA = encoder({
  'cat': [0.5, 0.5, 0.4, 0.3], 'a photo of cat': [1, 0, 0, 0],
  'a close-up photo of cat': [0.4, 0.4, 0.4, 0.4], 'an image showing cat': [0.3, 0.3, 0.3, 0.3],
});
const A = await hermesSearch('cat', encA, itemsA, 4);
check(A.rounds.map(r => +r.margin.toFixed(6)).join() === [0.277636, 0.764298, 0.207107, 0.207107].join(),
  'CASE A margins match Python');
check(A.satisfied === true && A.chose === 'a photo of cat', 'CASE A: the decisive phrasing is chosen');
check(A.ranked.map(r => r.item.path).join() === 'a,d,b,c', 'CASE A order matches Python');
check(close(A.qvec[0], 1) && close(A.qvec[1], 0), 'CASE A qvec = the chosen phrasing embedding');

// CASE B — NO phrasing decisive (near-identical items → every margin < MIN_MARGIN):
// the gate refuses to trust any single phrasing and ensembles all four.
const itemsB = ['p', 'q', 'r', 's'].map((c, i) => mk(c, [1, 0.02 + 0.001 * i, 0.01, 0]));
const encB = encoder({
  'cat': [1, 0.02, 0.01, 0], 'a photo of cat': [1, 0.021, 0.01, 0],
  'a close-up photo of cat': [1, 0.019, 0.011, 0], 'an image showing cat': [1, 0.02, 0.009, 0],
});
const B = await hermesSearch('cat', encB, itemsB, 4);
check(B.rounds.every(r => r.margin < MIN_MARGIN), 'CASE B: every phrasing is below the margin gate');
check(B.satisfied === false && B.chose === 'an ensemble of all phrasings',
  'CASE B: falls back to the ensemble');
check(close(B.qvec[0], 0.99975) && close(B.qvec[1], 0.019995, 1e-4),
  'CASE B qvec = unit(mean of all four phrasings), matches Python');

// the property the live wiring depends on: qvec is EXACTLY what produced the
// shown ranking, so re-ranking on it reproduces the same order (both cases).
for (const [name, r] of [['A', A], ['B', B]]) {
  const reRanked = rank(name === 'A' ? itemsA : itemsB, r.qvec, 'fused', 4).map(x => x.item.path);
  check(reRanked.join() === r.ranked.map(x => x.item.path).join(),
    `CASE ${name}: qvec reproduces the published ranking (trace can't lie)`);
}

// phrasing templates stay in lock-step with hermes.py's QUERY_TEMPLATES
check(QUERY_TEMPLATES.join('|') === '{q}|a photo of {q}|a close-up photo of {q}|an image showing {q}',
  'QUERY_TEMPLATES mirror hermes.py');

if (failed) { console.error('some hermes.js checks FAILED'); process.exit(1); }
console.log('all hermes.js checks passed');
