// Model-free checks that judge.js mirrors judge.py — the score GATE and the
// COUNCIL's quorum / hung-jury / ruling, pinned to the Python numbers on the
// committed gallery. Run: node docs/js/test_judge.mjs  (CI runs it too).
import { readFileSync } from 'node:fs';
import { parseScore, aggregate, majority, council, heuristicVotes,
  QUORUM, ACCEPT, HUNG_SPREAD, RUBRICS } from './judge.js';

let failed = false;
const check = (cond, msg) => {
  console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`);
  failed ||= !cond;
};
const close = (a, b, eps = 1e-4) => Math.abs(a - b) < eps;

check(QUORUM === 2 && ACCEPT === 0.5 && HUNG_SPREAD === 0.5, 'constants match judge.py');
check(RUBRICS.length === 3, 'three rubrics (relevance / specificity / faithfulness)');

// the gate — accept what a small model emits, reject out of range / no number
for (const [t, want] of [['0.7', 0.7], ['.7', 0.7], ['7/10', 0.7], ['8 out of 10', 0.8],
  ['70%', 0.7], ['score: 0.9', 0.9], ['relevant', null], ['2.5', null], ['150%', null]])
  check(parseScore(t) === want, `gate "${t}" → ${want}`);

// a judge with no parseable score ABSTAINS; the mean is confidence-weighted
const votes = [{ name: 'a', score: 0.8, confidence: 0.9 },
  { name: 'b', score: null, confidence: 0.7 }, { name: 'c', score: 0.7, confidence: 0.6 }];
const v = aggregate(votes);
check(v.nValid === 2 && v.abstained.join() === 'b', 'the scoreless judge abstains');
check(close(v.mean, (0.8 * 0.9 + 0.7 * 0.6) / (0.9 + 0.6)), 'confidence-weighted mean, not a plain average');
check(v.decision === 'relevant', 'mean ≥ ACCEPT → relevant');

// too few valid votes → no quorum
check(aggregate([{ name: 'a', score: 0.9 }, { name: 'b', score: null }]).decision === 'abstain',
  'fewer than QUORUM valid votes → abstain (no quorum)');

// a split panel is a HUNG JURY — abstain rather than average over a coin flip
const hung = aggregate([{ name: 'a', score: 0.1 }, { name: 'b', score: 0.9 }]);
check(hung.decision === 'abstain' && hung.reason === 'hung jury', 'spread > HUNG_SPREAD → hung jury');
check(close(hung.consensus, 0.2), 'consensus = 1 − spread');

// majority: yes/no votes, ties abstain
check(majority(votes).decision === 'relevant', 'majority: 2 yes → relevant');
const tie = majority([{ name: 'a', score: 0.9 }, { name: 'b', score: 0.1 }]);
check(tie.decision === 'abstain' && tie.reason === 'tie', 'majority tie → abstain');

// the model-free council on the real gallery — pinned to judge.py's numbers
const items = JSON.parse(readFileSync(new URL('../db.json', import.meta.url))).items;
const by = s => items.find(it => (it.file || it.path || '').includes(s));
const strong = council(by('004_cat'), by('005_dog'));       // cat → dog: clear match
check(strong.decision === 'relevant' && strong.nValid === 3, 'cat→dog: council rules relevant');
check(close(strong.mean, 0.7338) && close(strong.consensus, 0.6618), 'cat→dog numbers match judge.py');
const split = council(by('000_apple'), by('011_pluto'));    // shares "apple" by a tagging fluke
check(split.decision === 'abstain' && split.reason === 'hung jury', 'apple→pluto: tag says yes, vision says no → hung');
check(close(split.mean, 0.4248) && close(split.consensus, 0.2), 'apple→pluto numbers match judge.py');

// each rubric reads a DIFFERENT signal — which is why they can disagree
const hv = heuristicVotes(by('000_apple'), by('011_pluto'));
check(hv.find(j => j.name === 'faithfulness').score === 1
  && hv.find(j => j.name === 'relevance').score < 0.5,
  'faithfulness (shared top tag) high while relevance (vision) low — the split is real');

if (failed) { console.error('some judge.js checks FAILED'); process.exit(1); }
console.log('all judge.js checks passed');
