// Model-free checks that trust.js mirrors trust.py — the composer (quorum,
// split-decision, participation cap) and the four lenses, pinned to Python, plus
// a gallery-level end-to-end compose. Run: node docs/js/test_trust.mjs (CI too).
import { readFileSync } from 'node:fs';
import { compose, gateTrust, conformalTrust, councilTrust, marginTrust,
  QUORUM, SPLIT, HIGH, MED, MIN_FOR_HIGH, WEIGHTS } from './trust.js';
import { council } from './judge.js';
import { calibrate, looScores } from './conformal.js';

let failed = false;
const check = (cond, msg) => { console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`); failed ||= !cond; };
const close = (a, b, eps = 1e-4) => Math.abs(a - b) < eps;

check(QUORUM === 2 && SPLIT === 0.5 && HIGH === 0.66 && MED === 0.40 && MIN_FOR_HIGH === 3,
  'constants match trust.py');

// composer — pinned to trust.py on identical signal inputs
const c1 = compose([{ name: 'a', trust: 1.0, weight: 1.0 }, { name: 'b', trust: 0.71, weight: 1.0 },
  { name: 'c', trust: 0.73, weight: 1.2 }, { name: 'd', trust: 1.0, weight: 0.7 }]);
check(c1.level === 'high' && close(c1.score, 0.8426) && close(c1.consensus, 0.71),
  'four strong signals → high, weighted score 0.8426');

const c2 = compose([{ name: 'a', trust: 0.70, weight: 1 }, { name: 'b', trust: null, weight: 1 },
  { name: 'c', trust: null, weight: 1.2 }, { name: 'd', trust: 0.90, weight: 0.7 }]);
check(c2.level === 'medium' && c2.reason.startsWith('capped') && close(c2.score, 0.7824),
  'two lenses abstained → high is CAPPED to medium (participation)');

check(compose([{ name: 'a', trust: 0.2, weight: 1 }, { name: 'b', trust: 0.9, weight: 1 }]).level === 'abstain',
  'spread > SPLIT → abstain (split decision)');
check(compose([{ name: 'a', trust: 0.4, weight: 1 }, { name: 'b', trust: 0.9, weight: 1 }]).reason === 'composed',
  'spread == SPLIT (0.5) is NOT a split — strict >');
check(compose([{ name: 'a', trust: 0.9, weight: 1 }, { name: 'b', trust: null, weight: 1 }]).level === 'abstain',
  'fewer than QUORUM voting → abstain (not enough signals)');

// the four lenses
check(gateTrust(0.85, 0.8, 0.72, 0.66) === 1.0 && gateTrust(0.75, 0.8, 0.72, 0.66) === 0.7
  && gateTrust(0.68, 0.8, 0.72, 0.66) === 0.4 && gateTrust(0.5, 0.8, 0.72, 0.66) === 0.1,
  'gate buckets: strong/moderate/weak/very-weak → 1/0.7/0.4/0.1');
check(conformalTrust(0.63, 0.63) === 0.5 && close(conformalTrust(0.8, 0.6), 0.75)
  && conformalTrust(0.5, 0.6) === null,
  'conformal: just-cleared 0.5, graded above, below-bar → null (abstain)');
check(councilTrust({ decision: 'abstain' }) === null && councilTrust({ decision: 'relevant', mean: 0.7 }) === 0.7,
  'council: abstain → null, ruled → its mean');
check(close(marginTrust([0.8, 0.6, 0.5]), 1.0) && marginTrust([0.8]) === null,
  'margin: decisive winner → 1, single result → null');

// ── gallery end-to-end: compose the real lenses, pinned to trust.py's verdicts ──
const items = JSON.parse(readFileSync(new URL('../db.json', import.meta.url))).items;
const by = s => items.find(it => (it.file || it.path || '').includes(s));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const fused = it => { const v = [...it.image_emb, ...it.text_emb]; return v.map(x => x / Math.SQRT2); };
const tau = 1 - calibrate(looScores(items), 0.2);
const STRONG = 0.80, MODERATE = 0.72, WEAK = 0.66;

function verdictFor(qName, rName) {
  const q = by(qName);
  const ranked = items.filter(it => it !== q).sort((a, b) => dot(fused(b), fused(q)) - dot(fused(a), fused(q)));
  const scores = ranked.map(it => dot(fused(it), fused(q)));
  const i = ranked.indexOf(by(rName));
  const cos = scores[i], rest = scores.slice(0, i).concat(scores.slice(i + 1));
  const icos = dot(by(rName).image_emb, q.image_emb);
  return compose([
    { name: 'gate', trust: gateTrust(cos, STRONG, MODERATE, WEAK), weight: WEIGHTS.gate },
    { name: 'conformal', trust: conformalTrust(icos, tau), weight: WEIGHTS.conformal },
    { name: 'council', trust: councilTrust(council(q, by(rName))), weight: WEIGHTS.council },
    { name: 'margin', trust: marginTrust([cos, ...rest]), weight: WEIGHTS.margin },
  ]);
}
const dog = verdictFor('004_cat', '005_dog');
check(dog.level === 'high' && dog.reason === 'composed',
  'cat→dog: all four lenses agree → HIGH trust (matches trust.py)');
const pizza = verdictFor('000_apple', '010_pizza');
check(pizza.level === 'medium' && pizza.reason === 'composed',
  'apple→pizza: council abstains (hung) but gate+conformal+margin → medium (matches trust.py)');
const bike = verdictFor('000_apple', '002_bicycle');
check(bike.level === 'abstain' && bike.reason === 'split decision',
  'apple→bicycle: the lenses split → abstain (matches trust.py)');

if (failed) { console.error('some trust.js checks FAILED'); process.exit(1); }
console.log('all trust.js checks passed');
