// Model-free checks that learn.js mirrors learn2rank.py — pinned to numbers the
// Python side prints on the SAME fixture. Run: node docs/js/test_learn.mjs
// (CI runs it next to the Python smoke tests, so the twins can never drift).
import { OnlineRanker } from './learn.js';

let failed = false;
const check = (cond, msg) => {
  console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`);
  failed ||= !cond;
};
const close = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;
const vclose = (a, b, eps = 1e-6) => a.length === b.length && a.every((x, i) => close(x, b[i], eps));

// features are [cos_image, cos_text, tag_overlap, rank_prior]
const FIX = [
  [[0.30, 0.28, 3, 1.00], 1],
  [[0.25, 0.20, 0, 0.50], 0],
  [[0.22, 0.24, 2, 0.33], 1],
  [[0.20, 0.10, 0, 0.25], 0],
];

// untrained ranker IS the base order (w = [1,0,0,0]) — the core safety property
const cold = new OnlineRanker();
check(vclose(cold.w, [1, 0, 0, 0]), 'cold start w = [1,0,0,0] (untrained == base)');
check(cold.rank([{ base_score: 0.6, features: [0, 0, 0, 0] }])[0].beta === 0,
  'beta = 0 with no feedback');

const r = new OnlineRanker();
for (const [f, l] of FIX) r.feedback(f, l);

// pinned to `python3 -c` on learn2rank.py with this exact fixture
check(vclose(r.w, [0.333747, 0.131501, 1.213983, 0.576071], 1e-5),
  'RankNet weights match Python to 5 dp');
check(r.n === 4 && r.nPairs() === 4, 'n = 4, n_pairs = 2 pos × 2 neg = 4');

const imp = Object.fromEntries(r.importance().map(f => [f.name, f.importance]));
check(close(imp.tag_overlap, 0.538, 1e-3) && imp.tag_overlap > imp.rank_prior
  && imp.rank_prior > imp.cos_image,
  'importance: tag_overlap 0.538 dominant, matches Python');

// blend: beta = 0.5·n/(n+3) = 2/7, and the top item's normalized score is 1.0
const cand = [
  { base_score: 0.60, features: [0.30, 0.28, 3, 1.00] },
  { base_score: 0.55, features: [0.20, 0.10, 0, 0.50] },
  { base_score: 0.50, features: [0.24, 0.26, 2, 0.33] },
];
const out = r.rank(cand);
check(close(out[0].beta, 2 / 7), 'beta = 0.5·n/(n+3) = 2/7');
check(vclose(out.map(o => o.score), [1.0, 0.357143, 0.110698], 1e-5),
  'blended scores match Python');
check(out[0].features[2] === 3, 'the tag-sharer the fixture liked ranks first');

// the 50% cap: even a lopsided feedback stream can't hand the learned score
// more than half the vote
const big = new OnlineRanker();
for (let i = 0; i < 50; i++) { big.feedback([1, 1, 5, 1], 1); big.feedback([0, 0, 0, 0], 0); }
check(big.rank(cand)[0].beta <= 0.5 + 1e-12, 'blend weight capped at 0.5');

// one-sided feedback → Rocchio nudge (no degenerate pairwise gradient)
const one = new OnlineRanker();
one.feedback([0.3, 0.3, 2, 1.0], 1);
check(vclose(one.w, [0.7375, -0.2625, 0.3, 0.75], 1e-6),
  'pos-only feedback = Rocchio nudge, matches Python');
check(one.nPairs() === 0, 'one-sided feedback forms no pairs');

// state round-trips: reload from buffer reproduces the same weights
const round = new OnlineRanker().loadState(r.toState());
check(vclose(round.w, r.w, 1e-12), 'toState → loadState reproduces the weights');

if (failed) { console.error('some learn.js checks FAILED'); process.exit(1); }
console.log('all learn.js checks passed');
