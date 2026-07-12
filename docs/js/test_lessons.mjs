// The JS lessons must reproduce the PYTHON-pinned numbers on the same
// committed data (README's "reproduce these numbers" table) — run in CI
// right after the Python versions print them.
// Run: node docs/js/test_lessons.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { modalityGap, evalRetrieval, combine, quantizeInt8, topNeighbors, centerRows, synthetic, buildIVF, searchIVF, recallAtK } from './lessons.js';
import { softmax, dot } from './rank.js';

const DB = JSON.parse(readFileSync(new URL('../db.json', import.meta.url)));
const items = DB.items;

let failed = false;
const check = (cond, msg) => {
  console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`);
  failed ||= !cond;
};
const close = (a, b, eps) => Math.abs(a - b) < eps;

// temperature.py: cat's caption vs all images — 7.9% at scale 1, 99.7% at 100
const cat = items.find(it => it.file.includes('cat'));
const scores = items.map(it => dot(it.image_emb, cat.text_emb));
check(close(Math.max(...softmax(scores, 1)) , 0.079, 0.002), 'temperature: scale 1 → top ≈ 7.9%');
check(Math.max(...softmax(scores, 100)) > 0.99, 'temperature: scale 100 → top > 99%');

// similarity.py: the modality gap, same means to 3 decimals
const gap = modalityGap(items);
check(close(gap['image · other images'], 0.569, 0.002), 'gap: image·images ≈ +0.569');
check(close(gap['image · OWN caption'], 0.293, 0.002), 'gap: image·own caption ≈ +0.293');
check(gap['image · OWN caption'] + 0.1 < gap['image · other images'],
  'gap ordering: own caption is FAR below other images');

// retrieval_eval.py: image mode P@1 = 0.857, MRR ≈ 0.875
const ev = evalRetrieval(items, 'image');
check(close(ev['P@1'], 0.857, 0.01), 'eval: image P@1 ≈ 0.857');
check(close(ev.MRR, 0.875, 0.01), 'eval: image MRR ≈ 0.875');

// arithmetic.py: the 'animal' centroid retrieves exactly the 4 animals
const animals = items.filter(it => it.tags.includes('animal'));
const q = combine(animals.map(it => it.image_emb), animals.map(() => 1));
const top4 = items.map(it => ({ it, s: dot(it.image_emb, q) }))
  .sort((a, b) => b.s - a.s).slice(0, 4).map(({ it }) => it.file);
check(animals.every(a => top4.includes(a.file)), 'arithmetic: animal centroid → the 4 animals');
check(combine([[1, 0], [1, 0]], [1, -1]) === null, 'arithmetic: cancellation → null');

// quantize.py: 4x smaller, 13/14 rows of top-3 neighbors identical
const vecs = items.map(it => it.image_emb);
const { q: qv, scale } = quantizeInt8(vecs);
check(scale > 0 && qv.every(v => v.every(x => Number.isInteger(x) && Math.abs(x) <= 127)),
  'quantize: int8 range with one shared scale');
const exact = topNeighbors(vecs), approx = topNeighbors(qv);
const agree = exact.filter((r, i) => r.join() === approx[i].join()).length;
check(agree >= 13, `quantize: ${agree}/14 top-3 rows identical (>=13)`);

// similarity.center: the gap fix widens the own-caption margin ~3x (Python-pinned)
const Ic = centerRows(items.map(it => it.image_emb));
const Tc = centerRows(items.map(it => it.text_emb));
const centered = items.map((it, i) => ({ ...it, image_emb: Ic[i], text_emb: Tc[i] }));
const g1 = modalityGap(centered);
check(close(g1['image · OWN caption'], 0.360, 0.005), 'center: own caption ≈ +0.360');
check((g1['image · OWN caption'] - g1['image · other captions'])
  > 2.5 * (gap['image · OWN caption'] - gap['image · other captions']),
  'center: own-caption margin widens ~3x');

// ann.py mirror: IVF on clustered synthetic data — the recall dial works
const { X, Q } = synthetic();
const { C, lists } = buildIVF(X);
check(lists.reduce((s, l) => s + l.length, 0) === X.length, 'ann: every vector in one list');
const exactNN = Q.map(q => X.map((v, i) => ({ i, s: dot(v, q) }))
  .sort((a, b) => b.s - a.s).slice(0, 10).map(({ i }) => i));
const run = probes => {
  let rec = 0, scanned = 0;
  Q.forEach((q, qi) => {
    const r = searchIVF(q, X, C, lists, 10, probes);
    rec += recallAtK(r.found, exactNN[qi]); scanned += r.scanned;
  });
  return [rec / Q.length, scanned / Q.length / X.length];
};
const [r1, s1] = run(1), [r8, s8] = run(8);
check(r1 > 0.7 && s1 < 0.05, `ann: probes 1 → recall ${r1.toFixed(2)} scanning ${(100 * s1).toFixed(1)}%`);
check(r8 > 0.9 && r8 >= r1 && s8 > s1, 'ann: more probes → more truth, more work');

if (failed) { console.error('some lessons.js checks FAILED'); process.exit(1); }
console.log('all lessons.js checks passed — JS reproduces the Python-pinned numbers');
