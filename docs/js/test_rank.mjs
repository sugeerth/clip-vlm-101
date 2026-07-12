// Model-free checks that the JS math mirrors the Python — test_smoke.py's twin.
// Run: node docs/js/test_rank.mjs   (CI runs it next to the Python smoke tests)
import { dot, unit, scoreItem, rank, topTags, fuse, softmax } from './rank.js';

let failed = false;
const check = (cond, msg) => {
  console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`);
  failed ||= !cond;
};
const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

// dot and unit — the two primitives everything else is built on
check(close(dot([1, 2, 3], [4, 5, 6]), 32), 'dot product');
const u = unit([3, 4]);
check(close(u[0], 0.6) && close(Math.hypot(...u), 1), 'unit re-scales to length 1');
check(unit([0, 0]).every(x => x === 0), 'unit of a zero vector stays finite');

// fuse — [a ; b] / √2 keeps unit length (fusion.py's test_fusion_math twin)
const a = unit([1, 2, -1, 0.5]), b = unit([-2, 0.3, 1, 1]);
const f = fuse(a, b);
check(f.length === 8 && close(Math.hypot(...f), 1, 1e-12), 'fuse keeps unit length');

// scoreItem — fused equals the AVERAGE of image and text similarity
const item = { image_emb: a, text_emb: b };
const q = unit([0.2, -1, 0.4, 2]);
check(close(scoreItem(item, q, 'image'), dot(a, q)), 'image mode = image dot');
check(close(scoreItem(item, q, 'text'), dot(b, q)), 'text mode = text dot');
check(close(scoreItem(item, q, 'fused'), (dot(a, q) + dot(b, q)) / 2, 1e-12),
  'fused = mean(image, text) — same identity search.py prints');

// topTags — test_smoke.test_top_tags with the same fixtures
const vocab = ['cat', 'dog', 'car'];
const tagEmbs = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];
const best = topTags([0.1, 0.9, 0, 0], tagEmbs, vocab, 2).map(s => s.tag);
check(best[0] === 'dog' && best[1] === 'cat', 'topTags orders by dot product');

// rank — descending order, sliced to k
const items = tagEmbs.map((e, i) => ({ image_emb: e, text_emb: e, name: vocab[i] }));
const ranked = rank(items, [0.1, 0.9, 0, 0], 'image', 2);
check(ranked.length === 2 && ranked[0].item.name === 'dog' && ranked[0].score >= ranked[1].score,
  'rank sorts descending and keeps top k');

// softmax — temperature.py's twin
const p = softmax([0.3, 0.2, 0.1]);
check(close(p.reduce((s, x) => s + x, 0), 1, 1e-12) && p[0] > p[1] && p[1] > p[2],
  'softmax sums to 1 and keeps order');
check(softmax([0.3, 0.2, 0.1], 0).every(x => close(x, 1 / 3, 1e-12)), 'scale 0 → uniform');
check(softmax([0.3, 0.2, 0.1], 1000)[0] > 0.999, 'huge scale → one-hot');

if (failed) { console.error('some rank.js checks FAILED'); process.exit(1); }
console.log('all rank.js checks passed');
